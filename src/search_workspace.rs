use crate::{
    embedder::DefaultEmbeddingSpace, Embedder, FrozenIndex, Hit, PreparedQuery, Query,
    SearchResult, SearchScratch, TokenEmbedder, Tokenizer, TokenizerScratch,
};
use std::{collections::HashMap, hash::Hash};

/// Borrowed search result; neither the external ID nor the document text is copied.
#[derive(Debug, PartialEq)]
pub struct SearchResultRef<'a, K> {
    /// External document identifier.
    pub id: &'a K,
    /// Original document text.
    pub contents: &'a str,
    /// Relevance score.
    pub score: f32,
}

/// Reusable tokenization, preparation and ranking buffers for text searches.
pub struct SearchWorkspace<D = DefaultEmbeddingSpace> {
    tokenizer: TokenizerScratch,
    query: Query<D>,
    prepared: PreparedQuery,
    /// Ranking workspace; use it to choose scoring mode and inspect pruning counters.
    pub ranking: SearchScratch,
    hits: Vec<Hit>,
}

impl<D> Default for SearchWorkspace<D> {
    fn default() -> Self {
        Self {
            tokenizer: TokenizerScratch::default(),
            query: Query::default(),
            prepared: PreparedQuery::default(),
            ranking: SearchScratch::default(),
            hits: Vec::new(),
        }
    }
}

pub(crate) fn search_into<'a, K, D, T>(
    embedder: &Embedder<D, T>,
    index: &'a FrozenIndex<K, D::EmbeddingSpace>,
    documents: &'a HashMap<K, String>,
    query: &str,
    limit: usize,
    workspace: &mut SearchWorkspace<D::EmbeddingSpace>,
    out: &mut Vec<SearchResultRef<'a, K>>,
) where
    K: Hash + Eq + Clone,
    D: TokenEmbedder,
    D::EmbeddingSpace: Hash + Eq + Clone,
    T: Tokenizer,
{
    out.clear();
    embedder.query_into(query, &mut workspace.tokenizer, &mut workspace.query);
    index.prepare_into(&workspace.query, &mut workspace.prepared);
    index.search_into(
        &workspace.prepared,
        limit,
        &mut workspace.ranking,
        &mut workspace.hits,
    );
    for hit in &workspace.hits {
        let id = index.document_id(hit.doc_id).unwrap();
        out.push(SearchResultRef {
            id,
            contents: &documents[id],
            score: hit.score,
        });
    }
}

pub(crate) fn owned<K: Clone>(results: Vec<SearchResultRef<'_, K>>) -> Vec<SearchResult<K>> {
    results
        .into_iter()
        .map(|r| SearchResult {
            document: crate::Document::new(r.id.clone(), r.contents),
            score: r.score,
        })
        .collect()
}
