use crate::tokenizer::Tokenizer;
use fxhash::{hash, hash32, hash64};
#[cfg(feature = "parallelism")]
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fmt::{self, Debug, Display},
    hash::Hash,
    marker::PhantomData,
    ops::Deref,
};

pub type DefaultTokenEmbedder = u32;
pub type DefaultEmbeddingSpace = u32;

/// The default tokenizer is available via the `default_tokenizer` feature. It should fit most
/// use-cases. It splits on whitespace and punctuation, removes stop words and stems the
/// remaining words. It can also detect languages via the `language_detection` feature. This crate
/// uses `DefaultTokenizer` as the default concrete type for things that are generic
/// over a `Tokenizer`.
#[cfg(feature = "default_tokenizer")]
pub type DefaultTokenizer = crate::default_tokenizer::DefaultTokenizer;

/// A dummy type to represent the absence of a default tokenizer. If a compile error led you here,
/// you either need to enable the `default_tokenizer` feature, or specify your custom tokenizer as
/// a type parameter to whatever you're trying to construct.
#[cfg(not(feature = "default_tokenizer"))]
pub struct NoDefaultTokenizer {}
/// The default tokenizer is available via the `default_tokenizer` feature. It should fit most
/// use-cases. It splits on whitespace and punctuation, removes stop words and stems the
/// remaining words. It can also detect languages via the `language_detection` feature. This crate
/// uses `DefaultTokenizer` as the default concrete type for things that are generic
/// over a `Tokenizer`.
#[cfg(not(feature = "default_tokenizer"))]
pub type DefaultTokenizer = NoDefaultTokenizer;

/// Represents a token embedded in a D-dimensional space.
#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub struct TokenEmbedding<D = DefaultEmbeddingSpace> {
    /// The index of the token in the embedding space.
    pub index: D,
    /// The value of the token in the embedding space.
    pub value: f32,
}

impl<D> TokenEmbedding<D> {
    /// Creates a token entry. `Embedding::new` and `Query::new` validate its weight.
    pub fn new(index: D, value: f32) -> Self {
        Self { index, value }
    }
}

impl Display for TokenEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Represents a document embedded in a D-dimensional space.
#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub struct Embedding<D = DefaultEmbeddingSpace>(pub(crate) Vec<TokenEmbedding<D>>);

impl<D> Deref for Embedding<D> {
    type Target = Vec<TokenEmbedding<D>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D: Eq + Hash + Clone> Embedding<D> {
    /// Constructs a canonical document vector, keeping the first weight for each index.
    /// Panics for negative or nonfinite weights. Zero weights are omitted.
    pub fn new(tokens: impl IntoIterator<Item = TokenEmbedding<D>>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut terms = Vec::new();
        for token in tokens {
            assert!(
                token.value.is_finite() && token.value >= 0.0,
                "document weights must be finite and nonnegative"
            );
            if seen.insert(token.index.clone()) && token.value != 0.0 {
                terms.push(token);
            }
        }
        Self(terms)
    }
}

pub(crate) struct DocumentTerms<D> {
    pub(crate) terms: Vec<(D, u32)>,
    pub(crate) len: usize,
}

impl<D> Embedding<D> {
    /// Returns an iterator over the indices of the embedding.
    pub fn indices(&self) -> impl Iterator<Item = &D> {
        self.iter().map(|TokenEmbedding { index, .. }| index)
    }

    /// Returns an iterator over the values of the embedding.
    pub fn values(&self) -> impl Iterator<Item = &f32> {
        self.iter().map(|TokenEmbedding { value, .. }| value)
    }
}

impl<D: Debug> Display for Embedding<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A trait for embedding. Implement this to customise the embedding space and function.
pub trait TokenEmbedder: Sync {
    /// The output type of the embedder, i.e., the embedding space.
    type EmbeddingSpace: Eq + Hash + Clone + Send + Sync;
    /// Embeds a token into the embedding space.
    fn embed(token: &str) -> Self::EmbeddingSpace;
}

impl TokenEmbedder for u32 {
    type EmbeddingSpace = Self;
    fn embed(token: &str) -> u32 {
        hash32(token)
    }
}

impl TokenEmbedder for u64 {
    type EmbeddingSpace = Self;
    fn embed(token: &str) -> u64 {
        hash64(token)
    }
}

impl TokenEmbedder for usize {
    type EmbeddingSpace = Self;
    fn embed(token: &str) -> usize {
        hash(token)
    }
}

/// Creates sparse embeddings from text. D is the type of the token embedder and T is the type of
/// the tokenizer.
#[derive(Debug)]
pub struct Embedder<D = DefaultTokenEmbedder, T = DefaultTokenizer> {
    tokenizer: T,
    k1: f32,
    b: f32,
    avgdl: f32,
    token_embedder_type: PhantomData<D>,
}

impl<D, T> Embedder<D, T> {
    const FALLBACK_AVGDL: f32 = 256.0;

    /// Returns the average document length used by the embedder.
    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    /// Embeds the given text, producing one entry per distinct token index.
    pub fn embed(&self, text: &str) -> Embedding<D::EmbeddingSpace>
    where
        D: TokenEmbedder,
        D::EmbeddingSpace: Eq + Hash + Clone,
        T: Tokenizer,
    {
        let terms = self.count_terms(text, &mut crate::TokenizerScratch::default());
        self.embed_terms(terms)
    }

    /// Tokenizes a query without computing document BM25 weights.
    pub fn query(&self, text: &str) -> crate::Query<D::EmbeddingSpace>
    where
        D: TokenEmbedder,
        D::EmbeddingSpace: Eq,
        T: Tokenizer,
    {
        let mut query = crate::Query::default();
        self.query_into(text, &mut crate::TokenizerScratch::default(), &mut query);
        query
    }

    /// Reuses normalized-text and query buffers; duplicate words increase the query boost.
    pub fn query_into(
        &self,
        text: &str,
        scratch: &mut crate::TokenizerScratch,
        query: &mut crate::Query<D::EmbeddingSpace>,
    ) where
        D: TokenEmbedder,
        D::EmbeddingSpace: Eq,
        T: Tokenizer,
    {
        query.clear();
        self.tokenizer
            .for_each_token(text, scratch, |s| query.push(D::embed(s), 1.0));
    }

    pub(crate) fn count_terms(
        &self,
        text: &str,
        scratch: &mut crate::TokenizerScratch,
    ) -> DocumentTerms<D::EmbeddingSpace>
    where
        D: TokenEmbedder,
        D::EmbeddingSpace: Eq + Hash + Clone,
        T: Tokenizer,
    {
        let mut positions = HashMap::new();
        let mut terms: Vec<(D::EmbeddingSpace, u32)> = Vec::new();
        let mut len = 0;
        self.tokenizer.for_each_token(text, scratch, |s| {
            let index = D::embed(s);
            let position = *positions.entry(index.clone()).or_insert_with(|| {
                terms.push((index, 0));
                terms.len() - 1
            });
            terms[position].1 = terms[position]
                .1
                .checked_add(1)
                .expect("token frequency exceeds u32");
            len += 1;
        });
        DocumentTerms { terms, len }
    }

    pub(crate) fn set_avgdl(&mut self, avgdl: f32) {
        self.avgdl = avgdl;
    }

    pub(crate) fn embed_terms(
        &self,
        terms: DocumentTerms<D::EmbeddingSpace>,
    ) -> Embedding<D::EmbeddingSpace>
    where
        D: TokenEmbedder,
    {
        let avgdl = if self.avgdl <= 0.0 {
            Self::FALLBACK_AVGDL
        } else {
            self.avgdl
        };
        let norm = if self.k1 == 0.0 {
            0.0
        } else if self.b == 0.0 {
            self.k1
        } else {
            self.k1 * (1.0 - self.b + self.b * (terms.len as f32 / avgdl))
        };
        Embedding(
            terms
                .terms
                .into_iter()
                .map(|(index, tf)| {
                    let tf = tf as f32;
                    let value = tf * (self.k1 + 1.0) / (tf + norm);
                    assert!(
                        value.is_finite() && value > 0.0,
                        "BM25 weight overflow/underflow; use less extreme parameters"
                    );
                    TokenEmbedding { index, value }
                })
                .collect(),
        )
    }
}

/// A consuming builder for Embedder.
pub struct EmbedderBuilder<D = DefaultTokenEmbedder, T = DefaultTokenizer> {
    k1: f32,
    b: f32,
    avgdl: f32,
    tokenizer: T,
    token_embedder_type: PhantomData<D>,
}

impl<D, T> EmbedderBuilder<D, T> {
    /// Constructs a new EmbedderBuilder with the given average document length. Use this if you
    /// know the average document length in advance. If you don't, but you have your full corpus
    /// ahead of time, use `with_fit_to_corpus` or `with_tokenizer_and_fit_to_corpus` instead.
    ///
    /// If you have neither the full corpus nor a sample of it, you can configure the embedder to
    /// disregard document length by setting `b` to 0.0. In this case, it doesn't matter what
    /// value you pass to `with_avgdl`.
    ///
    /// The average document length is the average number of tokens in a document from your corpus;
    /// if you need access to this value, you can construct an Embedder and call `avgdl` on it.
    pub fn with_avgdl(avgdl: f32) -> EmbedderBuilder<D, T>
    where
        T: Default,
    {
        EmbedderBuilder {
            k1: 1.2,
            b: 0.75,
            avgdl,
            tokenizer: T::default(),
            token_embedder_type: PhantomData,
        }
    }

    /// Constructs a new EmbedderBuilder with its average document length fit to the given corpus.
    /// Use this if you have the full corpus (or a sample of it) available in advance. The embedder
    /// will assume the given tokenizer. Use the `parallelism` feature to speed the fitting process
    /// up for large corpora.
    pub fn with_tokenizer_and_fit_to_corpus(tokenizer: T, corpus: &[&str]) -> EmbedderBuilder<D, T>
    where
        T: Tokenizer + Sync,
    {
        let avgdl = if corpus.is_empty() {
            Embedder::<D>::FALLBACK_AVGDL
        } else {
            #[cfg(not(feature = "parallelism"))]
            let corpus_iter = corpus.iter();
            #[cfg(feature = "parallelism")]
            let corpus_iter = corpus.par_iter();
            let total_len: u64 = corpus_iter
                .map(|doc| {
                    let mut count = 0;
                    tokenizer.for_each_token(doc, &mut crate::TokenizerScratch::default(), |_| {
                        count += 1
                    });
                    count
                })
                .sum();
            (total_len as f64 / corpus.len() as f64) as f32
        };

        EmbedderBuilder {
            k1: 1.2,
            b: 0.75,
            avgdl,
            tokenizer,
            token_embedder_type: PhantomData,
        }
    }

    /// Sets the k1 parameter for the embedder. The default value is 1.2.
    pub fn k1(self, k1: f32) -> EmbedderBuilder<D, T> {
        EmbedderBuilder { k1, ..self }
    }

    /// Sets the b parameter for the embedder. The default value is 0.75.
    pub fn b(self, b: f32) -> EmbedderBuilder<D, T> {
        EmbedderBuilder { b, ..self }
    }

    /// Overrides the average document length for the embedder.
    pub fn avgdl(self, avgdl: f32) -> EmbedderBuilder<D, T> {
        EmbedderBuilder { avgdl, ..self }
    }

    /// Sets the tokenizer for the embedder.
    pub fn tokenizer(self, tokenizer: T) -> EmbedderBuilder<D, T> {
        EmbedderBuilder { tokenizer, ..self }
    }

    pub(crate) fn with_tokenizer(tokenizer: T) -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            avgdl: 256.0,
            tokenizer,
            token_embedder_type: PhantomData,
        }
    }

    /// Builds the Embedder. Panics for nonfinite parameters, negative k1, or b outside [0, 1].
    pub fn build(self) -> Embedder<D, T> {
        assert!(
            self.k1.is_finite() && self.k1 >= 0.0 && self.k1 < f32::MAX,
            "invalid k1"
        );
        assert!(
            self.b.is_finite() && (0.0..=1.0).contains(&self.b),
            "invalid b"
        );
        assert!(self.avgdl.is_finite(), "invalid avgdl");
        Embedder {
            tokenizer: self.tokenizer,
            k1: self.k1,
            b: self.b,
            avgdl: self.avgdl,
            token_embedder_type: PhantomData,
        }
    }
}

#[cfg(feature = "default_tokenizer")]
impl<D> EmbedderBuilder<D, DefaultTokenizer> {
    /// Constructs a new EmbedderBuilder with its average document length fit to the given corpus.
    /// Use this if you have the full corpus (or a sample of it) available in advance. This
    /// function uses the default tokenizer configured with the input language mode. The embedder
    /// will assume this tokenizer. Use the `parallelism` feature to speed the fitting process up
    /// for large corpora.
    pub fn with_fit_to_corpus(
        language_mode: impl Into<crate::LanguageMode>,
        corpus: &[&str],
    ) -> EmbedderBuilder<D, DefaultTokenizer> {
        let tokenizer = DefaultTokenizer::new(language_mode);
        EmbedderBuilder::with_tokenizer_and_fit_to_corpus(tokenizer, corpus)
    }

    /// Sets the language mode for the embedder tokenizer.
    pub fn language_mode(
        self,
        language_mode: impl Into<crate::LanguageMode>,
    ) -> EmbedderBuilder<D, DefaultTokenizer> {
        let tokenizer = DefaultTokenizer::new(language_mode);
        EmbedderBuilder { tokenizer, ..self }
    }
}

#[cfg(all(test, feature = "default_tokenizer"))]
#[allow(missing_docs)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::{
        test_data_loader::tests::{read_recipes, Recipe},
        Language, LanguageMode,
    };

    use super::*;

    fn embed_recipes(recipe_file: &str, language_mode: LanguageMode) -> Vec<Embedding> {
        let recipes = read_recipes(recipe_file);
        let embedder: Embedder = EmbedderBuilder::with_fit_to_corpus(
            language_mode,
            &recipes
                .iter()
                .map(|Recipe { recipe, .. }| recipe.as_str())
                .collect::<Vec<_>>(),
        )
        .build();

        recipes
            .iter()
            .map(|Recipe { recipe, .. }| recipe.as_str())
            .map(|recipe| embedder.embed(recipe))
            .collect::<Vec<_>>()
    }

    #[test]
    fn it_weights_unique_words_equally() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(3.0).build();
        let embedding = embedder.embed("banana apple orange");

        assert!(embedding.len() == 3);
        assert!(embedding.windows(2).all(|e| e[0].value == e[1].value));
    }

    #[test]
    fn it_weights_repeated_words_unequally() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(3.0)
            .tokenizer(DefaultTokenizer::new(Language::English))
            .build();
        let embedding = embedder.embed("space station station");

        assert!(
            *embedding
                == vec![
                    TokenEmbedding::new(866767497, 1.0),
                    TokenEmbedding::new(666609503, 1.375)
                ]
        );
    }

    #[test]
    fn it_constrains_avgdl() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(0.0)
            .language_mode(Language::English)
            .build();

        let embedding = embedder.embed("space station");

        assert!(!embedding.is_empty());
        assert!(embedding.iter().all(|e| e.value > 0.0));
    }

    #[test]
    fn it_handles_empty_corpus() {
        let embedder = EmbedderBuilder::<u32>::with_fit_to_corpus(Language::English, &[]).build();

        let embedding = embedder.embed("space station");

        assert!(!embedding.is_empty());
    }

    #[test]
    fn it_handles_empty_input() {
        let embedder = EmbedderBuilder::<u32>::with_avgdl(1.0).build();

        let embedding = embedder.embed("");

        assert!(embedding.is_empty());
    }

    #[test]
    fn it_allows_customisation_of_embedder() {
        #[derive(Eq, PartialEq, Hash, Clone, Debug)]
        struct MyType(u32);

        impl TokenEmbedder for MyType {
            type EmbeddingSpace = Self;
            fn embed(_: &str) -> Self {
                MyType(42)
            }
        }

        let embedder = EmbedderBuilder::<MyType>::with_avgdl(2.0).build();

        let embedding = embedder.embed("space station");

        assert_eq!(
            embedding.indices().cloned().collect::<Vec<_>>(),
            vec![MyType(42)]
        );
    }

    #[test]
    fn it_matches_snapshot_en() {
        let embeddings = embed_recipes("recipes_en.csv", LanguageMode::Fixed(Language::English));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(embeddings);
        });
    }

    #[test]
    fn it_matches_snapshot_de() {
        let embeddings = embed_recipes("recipes_de.csv", LanguageMode::Fixed(Language::German));

        insta::with_settings!({snapshot_path => "../snapshots"}, {
            assert_debug_snapshot!(embeddings);
        });
    }

    #[test]
    fn it_allows_customisation_of_tokenizer() {
        #[derive(Default)]
        struct MyTokenizer {}

        impl Tokenizer for MyTokenizer {
            fn tokenize<'a>(&'a self, input_text: &'a str) -> impl Iterator<Item = String> + 'a {
                input_text
                    .split('T')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            }
        }

        let embedder = EmbedderBuilder::<u32, MyTokenizer>::with_avgdl(1.0).build();

        let embedding = embedder.embed("CupTofTtea");

        assert_eq!(
            embedding.indices().cloned().collect::<Vec<_>>(),
            vec![3568447556, 3221979461, 415655421]
        );
    }
}
