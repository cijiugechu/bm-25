use crate::{
    embedder::DefaultTokenEmbedder, search_workspace, DefaultTokenizer, Embedder, FrozenIndex,
    SearchResult, SearchResultRef, SearchWorkspace, TokenEmbedder, Tokenizer,
};
use std::{collections::HashMap, hash::Hash};

/// Read-only search engine. Publish a new instance (for example via Arc) to apply updates.
/// `SearchEngine::freeze` releases the mutable builder's forward vectors.
pub struct FrozenSearchEngine<K, D: TokenEmbedder = DefaultTokenEmbedder, T = DefaultTokenizer> {
    pub(crate) embedder: Embedder<D, T>,
    pub(crate) index: FrozenIndex<K, D::EmbeddingSpace>,
    pub(crate) documents: HashMap<K, String>,
}

impl<K, D, T> FrozenSearchEngine<K, D, T>
where
    K: Eq + Hash + Clone,
    D: TokenEmbedder,
    D::EmbeddingSpace: Eq + Hash + Clone,
    T: Tokenizer,
{
    /// Returns the immutable scorer for repeated prepared queries.
    pub fn index(&self) -> &FrozenIndex<K, D::EmbeddingSpace> {
        &self.index
    }

    /// Returns the embedder for query preparation or external vector export.
    pub fn embedder(&self) -> &Embedder<D, T> {
        &self.embedder
    }

    /// Searches using caller-owned buffers and borrows the matching documents.
    pub fn search_into<'a>(
        &'a self,
        query: &str,
        limit: usize,
        workspace: &mut SearchWorkspace<D::EmbeddingSpace>,
        out: &mut Vec<SearchResultRef<'a, K>>,
    ) {
        search_workspace::search_into(
            &self.embedder,
            &self.index,
            &self.documents,
            query,
            limit,
            workspace,
            out,
        );
    }

    /// Convenience search returning owned documents.
    pub fn search(&self, query: &str, limit: impl Into<Option<usize>>) -> Vec<SearchResult<K>> {
        let mut results = Vec::new();
        self.search_into(
            query,
            limit.into().unwrap_or(usize::MAX),
            &mut SearchWorkspace::default(),
            &mut results,
        );
        search_workspace::owned(results)
    }
}
