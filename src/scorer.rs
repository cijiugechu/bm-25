use crate::{
    embedder::{DefaultEmbeddingSpace, Embedding},
    FrozenIndex, Query, SearchScratch,
};
use std::{collections::HashMap, hash::Hash, sync::OnceLock};

/// A document scored by BM25.
#[derive(PartialEq, Debug)]
pub struct ScoredDocument<K> {
    /// External document ID.
    pub id: K,
    /// Relevance score.
    pub score: f32,
}

/// Mutable index builder with a lazily rebuilt immutable search snapshot.
/// Updates invalidate the cached snapshot. Batch writes before calling `snapshot`.
/// Existing document weights/avgdl stay fixed on updates; rebuild from text to refit.
pub struct Scorer<K, D = DefaultEmbeddingSpace> {
    ids: HashMap<K, usize>,
    documents: Vec<Option<(K, Embedding<D>)>>,
    free: Vec<usize>,
    frozen: OnceLock<FrozenIndex<K, D>>,
}

impl<K, D> Default for Scorer<K, D> {
    fn default() -> Self {
        Self {
            ids: HashMap::new(),
            documents: Vec::new(),
            free: Vec::new(),
            frozen: OnceLock::new(),
        }
    }
}

impl<K: Eq + Hash + Clone, D: Eq + Hash + Clone> Scorer<K, D> {
    /// Creates an empty index builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a canonical document vector. Invalid weights panic before mutation.
    pub fn upsert(&mut self, id: &K, embedding: Embedding<D>) {
        self.frozen.take();
        let slot = if let Some(&slot) = self.ids.get(id) {
            slot
        } else {
            let slot = self.free.pop().unwrap_or_else(|| {
                self.documents.push(None);
                self.documents.len() - 1
            });
            self.ids.insert(id.clone(), slot);
            slot
        };
        self.documents[slot] = Some((id.clone(), embedding));
    }

    /// Removes a document. The next snapshot compacts vacant slots.
    pub fn remove(&mut self, id: &K) {
        if let Some(slot) = self.ids.remove(id) {
            self.frozen.take();
            self.documents[slot] = None;
            self.free.push(slot);
        }
    }

    /// Builds once after each write batch and returns the cached immutable snapshot.
    pub fn snapshot(&self) -> &FrozenIndex<K, D> {
        self.frozen.get_or_init(|| {
            FrozenIndex::build(
                self.documents
                    .iter()
                    .filter_map(|d| d.as_ref().map(|(k, e)| (k, e))),
            )
        })
    }

    /// Freezes this builder, releasing mutable document vectors and external-ID lookup storage.
    pub fn into_frozen(self) -> FrozenIndex<K, D> {
        if let Some(index) = self.frozen.into_inner() {
            index
        } else {
            FrozenIndex::build(
                self.documents
                    .iter()
                    .filter_map(|d| d.as_ref().map(|(k, e)| (k, e))),
            )
        }
    }

    /// Scores one document. Prefer `score_query` for explicit query boosts.
    pub fn score(&self, id: &K, query: &Embedding<D>) -> Option<f32> {
        self.score_query(id, &Query::from(query))
    }

    /// Scores one document using query frequencies/boosts, in query term order.
    pub fn score_query(&self, id: &K, query: &Query<D>) -> Option<f32> {
        let slot = *self.ids.get(id)?;
        let embedding = &self.documents[slot].as_ref()?.1;
        let index = self.snapshot();
        // This convenience operation deliberately leaves the batch search kernel untouched.
        Some(index.score_embedding(embedding, query))
    }

    /// Returns every positive match; embedding values are ignored in this compatibility adapter.
    pub fn matches(&self, query: &Embedding<D>) -> Vec<ScoredDocument<K>> {
        self.matches_query(&Query::from(query), usize::MAX)
    }

    /// Returns up to `limit` positive matches, without sorting all candidates.
    pub fn matches_query(&self, query: &Query<D>, limit: usize) -> Vec<ScoredDocument<K>> {
        let index = self.snapshot();
        let prepared = index.prepare(query);
        let mut hits = Vec::new();
        index.search_into(&prepared, limit, &mut SearchScratch::default(), &mut hits);
        hits.into_iter()
            .map(|hit| ScoredDocument {
                id: index.document_id(hit.doc_id).unwrap().clone(),
                score: hit.score,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::TokenEmbedding;

    use super::*;

    fn scorer_with_embeddings(embeddings: &[Embedding]) -> Scorer<usize> {
        let mut scorer = Scorer::<usize>::new();

        for (i, document_embedding) in embeddings.iter().enumerate() {
            scorer.upsert(&i, document_embedding.clone());
        }

        scorer
    }

    #[test]
    fn it_scores_missing_document_as_none() {
        let scorer = Scorer::<usize>::new();
        let query_embedding = Embedding::new([TokenEmbedding::new(1, 1.0)]);
        let score = scorer.score(&12345, &query_embedding);
        let matches = scorer.matches(&query_embedding);
        assert_eq!(score, None);
        assert!(matches.is_empty());
    }

    #[test]
    fn it_scores_mutually_exclusive_indices_as_zero() {
        let document_embeddings = vec![Embedding::new(vec![TokenEmbedding::new(1, 1.0)])];
        let scorer = scorer_with_embeddings(&document_embeddings);

        let query_embedding = Embedding::new(vec![TokenEmbedding::new(0, 1.0)]);
        let score = scorer.score(&0, &query_embedding);

        assert_eq!(score, Some(0.0));
    }

    #[test]
    fn it_scores_rare_indices_higher_than_common_ones() {
        // BM25 should score rare token matches higher than common token matches.
        let document_embeddings = vec![
            Embedding::new(vec![TokenEmbedding::new(0, 1.0)]),
            Embedding::new(vec![TokenEmbedding::new(0, 1.0)]),
            Embedding::new(vec![TokenEmbedding::new(1, 1.0)]),
        ];
        let scorer = scorer_with_embeddings(&document_embeddings);

        let score_1 = scorer.score(&0, &Embedding::new(vec![TokenEmbedding::new(0, 1.0)]));
        let score_2 = scorer.score(&2, &Embedding::new(vec![TokenEmbedding::new(1, 1.0)]));

        assert!(score_1.unwrap() < score_2.unwrap());
    }

    #[test]
    fn it_scores_longer_embeddings_lower_than_shorter_ones() {
        let document_embeddings = vec![
            // Longer embeddings will have a lower value for unique tokens.
            Embedding::new(vec![
                TokenEmbedding::new(0, 0.9),
                TokenEmbedding::new(1, 0.9),
            ]),
            Embedding::new(vec![TokenEmbedding::new(0, 1.0)]),
        ];
        let scorer = scorer_with_embeddings(&document_embeddings);

        let score_1 = scorer.score(&0, &Embedding::new(vec![TokenEmbedding::new(0, 1.0)]));
        let score_2 = scorer.score(&1, &Embedding::new(vec![TokenEmbedding::new(0, 1.0)]));

        assert!(score_1.unwrap() < score_2.unwrap());
    }

    #[test]
    fn it_only_matches_embeddings_with_non_zero_score() {
        let document_embeddings = vec![
            Embedding::new(vec![TokenEmbedding::new(0, 1.0)]),
            Embedding::new(vec![TokenEmbedding::new(1, 1.0)]),
        ];
        let scorer = scorer_with_embeddings(&document_embeddings);

        let query_embedding = Embedding::new(vec![TokenEmbedding::new(0, 1.0)]);
        let matches = scorer.matches(&query_embedding);

        assert_eq!(
            matches,
            vec![ScoredDocument {
                id: 0,
                score: std::f32::consts::LN_2
            }]
        );
    }

    #[test]
    fn it_does_not_score_frequent_terms_negatively() {
        // In versions 2.2.1 and earlier, the IDF considered the total occurrences of a token where
        // it should have considered the total number of documents containing the token. In
        // instances where the occurrences exceeded the number of documents, the IDF (and therefore
        // the score) would be negative.
        // See this bug report for more information: https://github.com/Michael-JB/bm25/pull/20
        let document_embeddings = vec![Embedding::new(vec![
            TokenEmbedding::new(0, 1.5),
            TokenEmbedding::new(0, 1.5),
        ])];
        let scorer = scorer_with_embeddings(&document_embeddings);
        let query_embedding = Embedding::new(vec![TokenEmbedding::new(0, 1.0)]);

        let matches = scorer.matches(&query_embedding);

        assert!(matches[0].score >= 0.0);
    }

    #[test]
    fn it_sorts_matches_by_score() {
        let document_embeddings = vec![
            Embedding::new(vec![
                TokenEmbedding::new(0, 0.9),
                TokenEmbedding::new(1, 0.9),
            ]),
            Embedding::new(vec![TokenEmbedding::new(0, 1.0)]),
        ];
        let scorer = scorer_with_embeddings(&document_embeddings);

        let query_embedding = Embedding::new(vec![TokenEmbedding::new(0, 1.0)]);
        let matches = scorer.matches(&query_embedding);

        assert_eq!(
            matches,
            vec![
                ScoredDocument {
                    id: 1,
                    score: 0.1823216
                },
                ScoredDocument {
                    id: 0,
                    score: 0.16408943
                }
            ]
        );
    }
}
