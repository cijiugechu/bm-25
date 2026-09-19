//! Standalone diagnostic: fully overlapping posting blocks, variable query length.
//! Build the release library, then link this file using rustc (see findings.md).
use bm_25::{Embedding, Query, Scorer, SearchScratch, TokenEmbedding};
use std::{hint::black_box, time::Instant};

fn median_ns(mut operation: impl FnMut(), iterations: usize) -> f64 {
    for _ in 0..3 {
        operation();
    }
    let mut samples = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    const BLOCKS: u32 = 64;
    println!("terms,query_build_us,prepare_us,search_all_us,search_top20_us,evaluated_blocks,skipped_blocks");
    for terms in [2_u32, 8, 32, 128, 512, 1024] {
        let mut scorer = Scorer::<u32>::new();
        for doc in 0..BLOCKS * 128 {
            let embedding = if doc % 128 == 0 {
                Embedding::new((0..terms).map(|term| TokenEmbedding::new(term, 1.0)))
            } else {
                Embedding::new([])
            };
            scorer.upsert(&doc, embedding);
        }
        let index = scorer.into_frozen();
        let make_query = || Query::new((0..terms).map(|t| TokenEmbedding::new(black_box(t), 1.0)));
        let query = make_query();
        let query_time = median_ns(
            || {
                black_box(make_query());
            },
            (8192 / terms as usize).max(8),
        );
        let mut prepared = index.prepare(&query);
        let prepare_time = median_ns(
            || {
                index.prepare_into(black_box(&query), &mut prepared);
                black_box(&prepared);
            },
            (8192 / terms as usize).max(8),
        );
        let mut scratch = SearchScratch::default();
        let mut hits = Vec::new();
        let mut times = [0.0; 2];
        for (i, limit) in [usize::MAX, 20].into_iter().enumerate() {
            index.search_into(&prepared, limit, &mut scratch, &mut hits);
            assert_eq!(hits.len(), (BLOCKS as usize).min(limit));
            assert_eq!(scratch.stats().evaluated_blocks, BLOCKS as usize);
            assert_eq!(scratch.stats().skipped_blocks, 0);
            times[i] = median_ns(
                || {
                    index.search_into(black_box(&prepared), limit, &mut scratch, &mut hits);
                    black_box(&hits);
                },
                (512 / terms as usize).max(4),
            );
        }
        println!(
            "{terms},{:.3},{:.3},{:.3},{:.3},{},{}",
            query_time / 1000.0,
            prepare_time / 1000.0,
            times[0] / 1000.0,
            times[1] / 1000.0,
            scratch.stats().evaluated_blocks,
            scratch.stats().skipped_blocks
        );
    }
}
