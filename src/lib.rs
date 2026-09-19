#![warn(missing_docs)]
#![cfg_attr(feature = "language_detection", doc = include_str!("../README.md"))]
// Suppress missing docs warning for this file
//!

#[cfg(test)]
mod test_data_loader;

mod embedder;
mod frozen_search;
mod index;
mod kernels;
mod query;
mod ranking;
mod scorer;
mod search;
mod search_workspace;
mod tokenizer;
mod traversal;

#[cfg(feature = "default_tokenizer")]
mod default_tokenizer;

#[cfg(feature = "default_tokenizer")]
pub use default_tokenizer::{Language, LanguageMode};

pub use embedder::{
    DefaultTokenizer, Embedder, EmbedderBuilder, Embedding, TokenEmbedder, TokenEmbedding,
};
pub use index::{DocId, FrozenIndex, Hit, PreparedQuery, ScoringMode, SearchScratch, SearchStats};
pub use query::Query;
pub use scorer::{ScoredDocument, Scorer};
pub use search::{Document, SearchEngine, SearchEngineBuilder, SearchResult};
pub use tokenizer::{Tokenizer, TokenizerScratch};
pub use traversal::TraversalMode;

pub use frozen_search::FrozenSearchEngine;
pub use search_workspace::{SearchResultRef, SearchWorkspace};
