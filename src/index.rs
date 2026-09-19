use crate::{
    embedder::DefaultEmbeddingSpace,
    kernels,
    ranking::{self, Candidate},
    traversal::{Traversal, TraversalMode},
    Embedding, Query,
};
use std::{
    collections::{BinaryHeap, HashMap},
    hash::Hash,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};

const BLOCK_SIZE: usize = 128;
// At half occupancy we trade up to 192 extra payload bytes for contiguous SIMD
// lanes. Keep the speed/space crossover visible in the density benchmark.
const DENSE_THRESHOLD: usize = BLOCK_SIZE / 2;
static NEXT_SNAPSHOT: AtomicU64 = AtomicU64::new(1);

/// A compact document handle, valid only in the snapshot that returned it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DocId(u32);

impl DocId {
    /// Returns the snapshot-local ordinal.
    pub fn ordinal(self) -> u32 {
        self.0
    }
}

/// A scored snapshot-local document. Resolve through `FrozenIndex::document_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Snapshot-local document handle.
    pub doc_id: DocId,
    /// Relevance score.
    pub score: f32,
}

/// Floating-point policy. Strict mode also enables conservative block pruning.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScoringMode {
    /// Separate multiply/add in query term order; deterministic within a snapshot/toolchain.
    #[default]
    Strict,
    /// Allows algebraic rewrites, with unspecified precision. Disables block pruning.
    Relaxed,
}

#[derive(Clone, Copy, Debug)]
struct TermId(u32);

#[derive(Clone, Debug)]
pub(crate) struct PreparedTerm {
    pub(crate) blocks: Range<usize>,
    pub(crate) factor: f32,
}

/// Query coefficients bound to an immutable index snapshot.
#[derive(Default, Debug)]
pub struct PreparedQuery {
    snapshot: u64,
    terms: Vec<PreparedTerm>,
}

/// Counters for the most recent search; useful for measuring pruning and block density.
#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    /// Selected traversal for this search; Auto for an empty/zero-limit search.
    pub traversal: TraversalMode,
    /// Matching term blocks visited, including those skipped by score bounds.
    pub matched_term_blocks: usize,
    /// Number of document blocks evaluated.
    pub evaluated_blocks: usize,
    /// Number of document blocks skipped using conservative score bounds.
    pub skipped_blocks: usize,
    /// Number of dense term blocks evaluated.
    pub dense_blocks: usize,
    /// Number of sparse term blocks evaluated.
    pub sparse_blocks: usize,
}

/// Reusable per-query workspace. Use a separate workspace for each concurrent search.
#[derive(Debug)]
pub struct SearchScratch {
    scores: [f32; BLOCK_SIZE],
    traversal: Traversal,
    traversal_mode: TraversalMode,
    heap: BinaryHeap<Candidate>,
    stats: SearchStats,
    mode: ScoringMode,
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self {
            scores: [0.0; BLOCK_SIZE],
            traversal: Traversal::default(),
            traversal_mode: TraversalMode::Auto,
            heap: BinaryHeap::new(),
            stats: SearchStats::default(),
            mode: ScoringMode::Strict,
        }
    }
}

impl SearchScratch {
    /// Chooses a floating-point policy for subsequent searches.
    pub fn set_mode(&mut self, mode: ScoringMode) {
        self.mode = mode;
    }
    /// Overrides automatic block scheduling for subsequent searches.
    /// Useful for workload-specific tuning; all modes preserve strict scores and ties.
    pub fn set_traversal_mode(&mut self, mode: TraversalMode) {
        self.traversal_mode = mode;
    }
    /// Returns counters for the last search.
    pub fn stats(&self) -> SearchStats {
        self.stats
    }
}

struct Term {
    blocks: Range<usize>,
    idf: f32,
}
pub(crate) struct Block {
    pub(crate) doc_block: u32,
    weights: Range<usize>,
    offsets: usize,
    max_weight: f32,
    dense: bool,
}

/// Immutable, compact, read-concurrent index with sparse/dense document-range blocks.
/// Parameters and document weights are fixed at construction; rebuild to change them.
pub struct FrozenIndex<K, D = DefaultEmbeddingSpace> {
    snapshot: u64,
    keys: Vec<K>,
    dictionary: HashMap<D, TermId>,
    terms: Vec<Term>,
    blocks: Vec<Block>,
    weights: Vec<f32>,
    // Sparse positions are delta-coded relative to a 128-document range (one byte).
    offsets: Vec<u8>,
}

impl<K, D> FrozenIndex<K, D> {
    /// Returns the number of documents, including documents with no searchable terms.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    /// Whether there are no documents.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Resolves a handle from this snapshot. Handles must not be mixed across snapshots.
    pub fn document_id(&self, id: DocId) -> Option<&K> {
        self.keys.get(id.0 as usize)
    }

    /// Searches a prepared query into a reusable output buffer.
    /// Panics if the query belongs to a different snapshot, even for limit zero.
    /// Ties are resolved by ascending snapshot-local document ID.
    pub fn search_into(
        &self,
        query: &PreparedQuery,
        limit: usize,
        scratch: &mut SearchScratch,
        out: &mut Vec<Hit>,
    ) {
        assert_eq!(
            query.snapshot, self.snapshot,
            "prepared query belongs to another index snapshot"
        );
        out.clear();
        scratch.heap.clear();
        scratch.stats = SearchStats::default();
        if limit == 0 || query.terms.is_empty() {
            return;
        }
        let all = limit >= self.len();
        scratch.stats.traversal =
            scratch
                .traversal
                .reset(&query.terms, &self.blocks, scratch.traversal_mode);
        while let Some(doc_block) = scratch.traversal.next(&self.blocks) {
            scratch.stats.matched_term_blocks += scratch.traversal.matches.len();
            // Same nonnegative multiplication/addition order as scoring: monotonic IEEE
            // operations bound every lane, including overflow to +inf. Strict '<' keeps ties.
            let mut upper = 0.0_f32;
            if !all && scratch.mode == ScoringMode::Strict && scratch.heap.len() == limit {
                for matched in &scratch.traversal.matches {
                    upper += self.blocks[matched.block].max_weight * matched.factor;
                }
            } else {
                upper = f32::INFINITY;
            }
            let skip = scratch
                .heap
                .peek()
                .is_some_and(|worst| upper < worst.0.score);
            if skip {
                scratch.stats.skipped_blocks += 1;
            } else {
                scratch.stats.evaluated_blocks += 1;
                scratch.scores.fill(0.0);
            }
            let mut touched = [0_u64; 2];
            if skip {
                continue;
            }
            for matched in &scratch.traversal.matches {
                let block = &self.blocks[matched.block];
                let weights = &self.weights[block.weights.clone()];
                if block.dense {
                    scratch.stats.dense_blocks += 1;
                    touched = [u64::MAX; 2];
                    match scratch.mode {
                        ScoringMode::Strict => {
                            kernels::accumulate(&mut scratch.scores, weights, matched.factor)
                        }
                        ScoringMode::Relaxed => kernels::accumulate_relaxed(
                            &mut scratch.scores,
                            weights,
                            matched.factor,
                        ),
                    }
                } else {
                    scratch.stats.sparse_blocks += 1;
                    let offsets = &self.offsets[block.offsets..block.offsets + weights.len()];
                    for (&offset, &weight) in offsets.iter().zip(weights) {
                        touched[offset as usize / 64] |= 1_u64 << (offset % 64);
                        let score = &mut scratch.scores[offset as usize];
                        *score = match scratch.mode {
                            ScoringMode::Strict => *score + weight * matched.factor,
                            ScoringMode::Relaxed => score.algebraic_add(weight * matched.factor),
                        };
                    }
                }
            }
            let base = doc_block as usize * BLOCK_SIZE;
            let len = BLOCK_SIZE.min(self.len() - base);
            for (word, mut bits) in touched.into_iter().enumerate() {
                while bits != 0 {
                    let offset = word * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if offset >= len {
                        break;
                    }
                    let score = scratch.scores[offset];
                    if score <= 0.0 {
                        continue;
                    }
                    let hit = Hit {
                        doc_id: DocId((base + offset) as u32),
                        score,
                    };
                    if all {
                        out.push(hit);
                    } else {
                        ranking::retain(&mut scratch.heap, hit, limit);
                    }
                }
            }
        }
        if !all {
            out.extend(scratch.heap.drain().map(|c| c.0));
        }
        ranking::sort(out);
    }
}

impl<K: Clone, D: Eq + Hash + Clone> FrozenIndex<K, D> {
    pub(crate) fn build<'a>(documents: impl Iterator<Item = (&'a K, &'a Embedding<D>)>) -> Self
    where
        K: 'a,
        D: 'a,
    {
        let snapshot = NEXT_SNAPSHOT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .expect("snapshot ID exhaustion");
        let mut index = Self {
            snapshot,
            keys: Vec::new(),
            dictionary: HashMap::new(),
            terms: Vec::new(),
            blocks: Vec::new(),
            weights: Vec::new(),
            offsets: Vec::new(),
        };
        let mut postings: Vec<Vec<(u32, f32)>> = Vec::new();
        for (key, embedding) in documents {
            let doc = u32::try_from(index.keys.len()).expect("more than u32::MAX documents");
            index.keys.push(key.clone());
            for token in embedding.iter() {
                let term = *index
                    .dictionary
                    .entry(token.index.clone())
                    .or_insert_with(|| {
                        postings.push(Vec::new());
                        TermId(u32::try_from(postings.len() - 1).expect("term ID exceeds u32"))
                    });
                postings[term.0 as usize].push((doc, token.value));
            }
        }
        for posting in postings {
            let df = posting.len() as f32;
            let idf = (1.0 + (index.len() as f32 - df + 0.5) / (df + 0.5)).ln();
            let start = index.blocks.len();
            let mut remaining = posting.as_slice();
            while let Some(&(first, _)) = remaining.first() {
                let doc_block = first / BLOCK_SIZE as u32;
                let count =
                    remaining.partition_point(|&(doc, _)| doc / BLOCK_SIZE as u32 == doc_block);
                let (chunk, rest) = remaining.split_at(count);
                remaining = rest;
                let dense = count >= DENSE_THRESHOLD;
                let weights_start = index.weights.len();
                let offsets = index.offsets.len();
                let mut max_weight = 0.0_f32;
                if dense {
                    index.weights.resize(weights_start + BLOCK_SIZE, 0.0);
                }
                for &(doc, weight) in chunk {
                    let offset = (doc % BLOCK_SIZE as u32) as u8;
                    if dense {
                        index.weights[weights_start + offset as usize] = weight;
                    } else {
                        index.offsets.push(offset);
                        index.weights.push(weight);
                    }
                    max_weight = max_weight.max(weight);
                }
                index.blocks.push(Block {
                    doc_block,
                    weights: weights_start..index.weights.len(),
                    offsets,
                    max_weight,
                    dense,
                });
            }
            index.terms.push(Term {
                blocks: start..index.blocks.len(),
                idf,
            });
        }
        index
    }

    pub(crate) fn score_embedding(&self, embedding: &Embedding<D>, query: &Query<D>) -> f32 {
        let mut score = 0.0;
        for token in query.terms() {
            if let Some(&term) = self.dictionary.get(&token.index) {
                if let Some(weight) = embedding.iter().find(|e| e.index == token.index) {
                    score += weight.value * (self.terms[term.0 as usize].idf * token.value);
                }
            }
        }
        score
    }

    /// Resolves terms and computes IDF once for this snapshot.
    pub fn prepare(&self, query: &Query<D>) -> PreparedQuery {
        let mut prepared = PreparedQuery::default();
        self.prepare_into(query, &mut prepared);
        prepared
    }

    /// Reuses a prepared-query buffer, replacing its previous snapshot binding.
    pub fn prepare_into(&self, query: &Query<D>, prepared: &mut PreparedQuery) {
        prepared.snapshot = self.snapshot;
        prepared.terms.clear();
        for token in query.terms() {
            if let Some(&term) = self.dictionary.get(&token.index) {
                let factor = self.terms[term.0 as usize].idf * token.value;
                assert!(factor.is_finite(), "query coefficient overflow");
                prepared.terms.push(PreparedTerm {
                    blocks: self.terms[term.0 as usize].blocks.clone(),
                    factor,
                });
            }
        }
    }
}
