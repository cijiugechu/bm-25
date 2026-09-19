use bm_25::{
    Embedding, Query, Scorer, ScoringMode, SearchScratch, TokenEmbedding as Token, TraversalMode,
};

fn reference(docs: &[Embedding], query: &Query) -> Vec<(u32, f32)> {
    let coefficients: Vec<_> = query
        .terms()
        .iter()
        .map(|term| {
            let df = docs
                .iter()
                .filter(|d| d.iter().any(|t| t.index == term.index))
                .count() as f32;
            let idf = (1.0 + (docs.len() as f32 - df + 0.5) / (df + 0.5)).ln();
            (term.index, idf * term.value)
        })
        .collect();
    let mut scores: Vec<_> = docs
        .iter()
        .enumerate()
        .filter_map(|(id, doc)| {
            let mut score = 0.0;
            for &(term, factor) in &coefficients {
                score += doc
                    .iter()
                    .find(|t| t.index == term)
                    .map_or(0.0, |t| t.value)
                    * factor;
            }
            (score > 0.0).then_some((id as u32, score))
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores
}

#[test]
fn blocked_topk_matches_exhaustive_scoring_across_densities_and_tails() {
    let mut seed = 42_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 32) as u32
    };
    for n in [0, 1, 63, 64, 127, 128, 129, 513] {
        let docs: Vec<_> = (0..n)
            .map(|doc| {
                Embedding::new((0..24).filter_map(|term| {
                    let include = term < 2 || random() % 128 < term * 4;
                    include.then(|| {
                        Token::new(
                            term,
                            ((random() % 1000 + 1) as f32 / 71.0)
                                * if doc < 128 { 8.0 } else { 1.0 },
                        )
                    })
                }))
            })
            .collect();
        let mut scorer = Scorer::new();
        for (id, doc) in docs.iter().enumerate() {
            scorer.upsert(&(id as u32), doc.clone());
        }
        let index = scorer.into_frozen();
        let mut scratch = SearchScratch::default();
        let mut out = Vec::new();
        for q in 0..20 {
            let query =
                Query::new((0..q).map(|_| Token::new(random() % 30, (random() % 5 + 1) as f32)));
            let expected = reference(&docs, &query);
            let prepared = index.prepare(&query);
            for k in [0, 1, 7, 128, n, usize::MAX] {
                index.search_into(&prepared, k, &mut scratch, &mut out);
                let actual: Vec<_> = out
                    .iter()
                    .map(|hit| (*index.document_id(hit.doc_id).unwrap(), hit.score.to_bits()))
                    .collect();
                let expected: Vec<_> = expected
                    .iter()
                    .take(k)
                    .map(|&(id, score)| (id, score.to_bits()))
                    .collect();
                assert_eq!(actual, expected, "n={n} q={q} k={k}");
            }
        }
    }
}

#[test]
fn pruning_skips_low_score_blocks_but_preserves_ties() {
    let mut scorer = Scorer::new();
    for id in 0..1025_u32 {
        scorer.upsert(
            &id,
            Embedding::new([Token::new(1, if id < 128 { 10.0 } else { 1.0 })]),
        );
    }
    let index = scorer.into_frozen();
    let query = index.prepare(&Query::new([Token::new(1, 2.0)]));
    let mut scratch = SearchScratch::default();
    let mut out = Vec::new();
    index.search_into(&query, 3, &mut scratch, &mut out);
    assert_eq!(
        out.iter()
            .map(|h| *index.document_id(h.doc_id).unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(scratch.stats().evaluated_blocks, 1);
    assert_eq!(scratch.stats().skipped_blocks, 8);
    assert!(scratch.stats().dense_blocks > 0);
    scratch.set_mode(ScoringMode::Relaxed);
    index.search_into(&query, 3, &mut scratch, &mut out);
    assert_eq!(scratch.stats().skipped_blocks, 0);
    assert_eq!(scratch.stats().evaluated_blocks, 9);
}

#[test]
fn snapshots_reject_stale_queries_and_updates_recompute_df() {
    let mut scorer = Scorer::new();
    scorer.upsert(&"a", Embedding::new([Token::new(1, 1.0)]));
    let query = Query::new([Token::new(1, 1.0)]);
    let old = scorer.snapshot().prepare(&query);
    scorer.upsert(&"b", Embedding::new([Token::new(2, 1.0)]));
    let score = scorer.score_query(&"a", &query).unwrap();
    assert_eq!(score, 2.0_f32.ln());
    scorer.remove(&"a");
    assert!(scorer.matches_query(&query, 10).is_empty());
    scorer.upsert(&"c", Embedding::new([Token::new(1, 2.0)]));
    assert_eq!(scorer.matches_query(&query, 1)[0].id, "c");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scorer
            .snapshot()
            .search_into(&old, 0, &mut SearchScratch::default(), &mut Vec::new());
    }));
    assert!(result.is_err());
}

#[test]
fn zero_empty_duplicate_and_invalid_inputs_have_explicit_semantics() {
    let doc = Embedding::new([Token::new(1, 2.0), Token::new(1, 99.0), Token::new(2, 0.0)]);
    assert_eq!(doc.len(), 1);
    assert_eq!(doc[0].value, 2.0);
    let query = Query::new([Token::new(1, 2.0), Token::new(1, 3.0), Token::new(2, 0.0)]);
    assert_eq!(query.terms(), [Token::new(1, 5.0)]);
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(std::panic::catch_unwind(|| Embedding::new([Token::new(1, value)])).is_err());
        assert!(std::panic::catch_unwind(|| Query::new([Token::new(1, value)])).is_err());
    }
    let mut scorer = Scorer::new();
    scorer.upsert(&0, doc);
    scorer.upsert(&1, Embedding::new([]));
    assert_eq!(
        scorer.score_query(&0, &Query::new([Token::new(1, 1.0)])),
        Some(2.0 * 2.0_f32.ln())
    );
    assert_eq!(scorer.score_query(&1, &query), Some(0.0));
}

#[test]
fn overflow_and_subnormal_scores_do_not_break_strict_pruning() {
    let docs: Vec<_> = (0..385)
        .map(|id| {
            Embedding::new([
                Token::new(
                    0,
                    if id < 128 {
                        f32::MAX
                    } else {
                        f32::from_bits(1)
                    },
                ),
                Token::new(1, if id % 3 == 0 { f32::MAX } else { 1.0 }),
            ])
        })
        .collect();
    let mut scorer = Scorer::new();
    for (id, doc) in docs.iter().enumerate() {
        scorer.upsert(&(id as u32), doc.clone());
    }
    let index = scorer.into_frozen();
    let query = Query::new([Token::new(0, 1024.0), Token::new(1, 1024.0)]);
    let mut out = Vec::new();
    index.search_into(
        &index.prepare(&query),
        150,
        &mut SearchScratch::default(),
        &mut out,
    );
    let actual: Vec<_> = out
        .iter()
        .map(|h| (*index.document_id(h.doc_id).unwrap(), h.score))
        .collect();
    assert_eq!(actual, reference(&docs, &query)[..150]);
}

#[test]
fn frozen_index_is_shareable_with_independent_workspaces() {
    let mut scorer = Scorer::new();
    scorer.upsert(&0_u32, Embedding::new([Token::new(1, 1.0)]));
    let index = scorer.into_frozen();
    let query = index.prepare(&Query::new([Token::new(1, 1.0)]));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                let mut out = Vec::new();
                index.search_into(&query, 1, &mut SearchScratch::default(), &mut out);
                assert_eq!(out.len(), 1);
            });
        }
    });
}

#[test]
fn relaxed_scores_are_checked_against_strict_on_representative_finite_inputs() {
    let docs: Vec<_> = (0..389)
        .map(|doc| {
            Embedding::new((0..19).filter_map(|term| {
                (term < 3 || (doc + term) % 5 == 0)
                    .then_some(Token::new(term, (doc % 101 + term + 1) as f32 / 73.0))
            }))
        })
        .collect();
    let mut scorer = Scorer::new();
    for (id, doc) in docs.iter().enumerate() {
        scorer.upsert(&(id as u32), doc.clone());
    }
    let index = scorer.into_frozen();
    let query = Query::new((0..19).map(|term| Token::new(term, (term + 1) as f32 / 7.0)));
    let expected = reference(&docs, &query);
    let mut scratch = SearchScratch::default();
    scratch.set_mode(ScoringMode::Relaxed);
    let mut out = Vec::new();
    index.search_into(&index.prepare(&query), usize::MAX, &mut scratch, &mut out);
    assert_eq!(out.len(), expected.len());
    for hit in out {
        let id = *index.document_id(hit.doc_id).unwrap();
        let score = expected.iter().find(|&&(key, _)| key == id).unwrap().1;
        assert!((hit.score - score).abs() <= 1e-5 * (1.0 + score));
    }
}

#[test]
fn long_queries_match_reference_with_every_traversal_and_reused_scratch() {
    // Dense common terms, sparse disjoint streams, exhausted streams and a tail.
    // Query order differs from index order and magnitudes make addition order matter.
    let docs: Vec<_> = (0..641_u32)
        .map(|doc| {
            Embedding::new(
                (0..160_u32)
                    .filter(|&term| term < 3 || (doc / 128 + term) % 5 == 0 || doc == term)
                    .map(|term| {
                        Token::new(
                            term,
                            if term % 7 == 0 {
                                1e8
                            } else {
                                (doc % 13 + term + 1) as f32 / 31.0
                            },
                        )
                    }),
            )
        })
        .collect();
    let mut scorer = Scorer::new();
    for (id, doc) in docs.iter().enumerate() {
        scorer.upsert(&(id as u32), doc.clone());
    }
    let index = scorer.into_frozen();
    let mut scratch = SearchScratch::default();
    let mut out = Vec::new();
    let mut prepared = bm_25::PreparedQuery::default();
    for q in [160, 2, 17, 0, 33, 159] {
        let query = Query::new((0..q).map(|i| Token::new((i * 73) % 163, (i % 5 + 1) as f32)));
        let expected = reference(&docs, &query);
        index.prepare_into(&query, &mut prepared);
        for mode in [
            TraversalMode::Heap,
            TraversalMode::Scan,
            TraversalMode::Auto,
        ] {
            scratch.set_traversal_mode(mode);
            for k in [0, 1, 20, 128, 700, usize::MAX] {
                index.search_into(&prepared, k, &mut scratch, &mut out);
                let actual: Vec<_> = out
                    .iter()
                    .map(|h| (h.doc_id.ordinal(), h.score.to_bits()))
                    .collect();
                let expected: Vec<_> = expected
                    .iter()
                    .take(k)
                    .map(|&(id, s)| (id, s.to_bits()))
                    .collect();
                assert_eq!(actual, expected, "q={q}, mode={mode:?}, k={k}");
            }
        }
    }
}

#[test]
fn heap_pruning_handles_ties_overflow_and_exhausted_streams() {
    for weight in [1.0, f32::MAX, f32::from_bits(1)] {
        let docs: Vec<_> = (0..513)
            .map(|doc| {
                Embedding::new((0..40).filter_map(|term| {
                    (term < 2 || doc < 256).then_some(Token::new(
                        term,
                        if doc < 128 { weight } else { f32::from_bits(1) },
                    ))
                }))
            })
            .collect();
        let mut scorer = Scorer::new();
        for (id, doc) in docs.iter().enumerate() {
            scorer.upsert(&(id as u32), doc.clone());
        }
        let index = scorer.into_frozen();
        let query = Query::new((0..40).rev().map(|term| Token::new(term, 2.0)));
        let expected = reference(&docs, &query);
        let mut scratch = SearchScratch::default();
        scratch.set_traversal_mode(TraversalMode::Heap);
        let mut out = Vec::new();
        index.search_into(&index.prepare(&query), 3, &mut scratch, &mut out);
        assert_eq!(
            out.iter()
                .map(|h| (h.doc_id.ordinal(), h.score))
                .collect::<Vec<_>>(),
            expected[..3]
        );
        if weight != f32::from_bits(1) {
            assert!(scratch.stats().skipped_blocks > 0);
        }
    }
}

#[test]
fn automatic_traversal_distinguishes_disjoint_and_overlapping_streams() {
    for overlap in [false, true] {
        let mut scorer = Scorer::new();
        for id in 0..8192_u32 {
            let terms = (0..64).filter(|&term| id % 128 == 0 && (overlap || term == id / 128));
            scorer.upsert(&id, Embedding::new(terms.map(|t| Token::new(t, 1.0))));
        }
        let index = scorer.into_frozen();
        let query = index.prepare(&Query::new((0..64).map(|t| Token::new(t, 1.0))));
        let mut scratch = SearchScratch::default();
        let mut out = Vec::new();
        index.search_into(&query, 20, &mut scratch, &mut out);
        assert_eq!(
            scratch.stats().traversal,
            if overlap {
                TraversalMode::Scan
            } else {
                TraversalMode::Heap
            }
        );
        assert_eq!(
            scratch.stats().matched_term_blocks,
            if overlap { 64 * 64 } else { 64 }
        );
        assert_eq!(
            out.iter().map(|h| h.doc_id.ordinal()).collect::<Vec<_>>(),
            (0..20).map(|i| i * 128).collect::<Vec<_>>()
        );
    }
}
