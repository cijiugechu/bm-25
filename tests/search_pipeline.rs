use bm_25::{
    Document, EmbedderBuilder, SearchEngineBuilder, SearchWorkspace, Tokenizer, TokenizerScratch,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
struct CountingTokenizer(Arc<AtomicUsize>);
impl Tokenizer for CountingTokenizer {
    fn tokenize<'a>(&'a self, text: &'a str) -> impl Iterator<Item = String> + 'a {
        self.0.fetch_add(1, Ordering::Relaxed);
        text.split_whitespace().map(str::to_owned)
    }
    fn for_each_token(&self, text: &str, _: &mut TokenizerScratch, mut visit: impl FnMut(&str)) {
        self.0.fetch_add(1, Ordering::Relaxed);
        for token in text.split_whitespace() {
            visit(token);
        }
    }
}

#[test]
fn builder_tokenizes_once_and_frozen_borrowed_search_reuses_buffers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = SearchEngineBuilder::<u32, u32, _>::with_tokenizer_and_documents(
        CountingTokenizer(calls.clone()),
        [
            Document::new(0, "a a b"),
            Document::new(1, "b c"),
            Document::new(2, ""),
        ],
    )
    .build_frozen();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(engine.embedder().avgdl(), 5.0 / 3.0);
    let single = engine.search("a", 10);
    let double = engine.search("a a", 10);
    assert_eq!(double[0].score, single[0].score * 2.0);
    let mut workspace = SearchWorkspace::default();
    let mut out = Vec::new();
    engine.search_into("a a", 10, &mut workspace, &mut out);
    assert_eq!(*out[0].id, 0);
    assert_eq!(out[0].contents, "a a b");
    assert_eq!(out[0].score, double[0].score);
    engine.search_into("unknown", 10, &mut workspace, &mut out);
    assert!(out.is_empty());
    engine.search_into("b", 0, &mut workspace, &mut out);
    assert!(out.is_empty());
}

#[test]
fn canonical_embedding_preserves_original_length_and_frequency() {
    let embedder = EmbedderBuilder::<u32, _>::with_avgdl(3.0)
        .tokenizer(CountingTokenizer::default())
        .build();
    let embedding = embedder.embed("a a b");
    assert_eq!(embedding.len(), 2);
    assert_eq!(embedding[0].value, 2.0 * 2.2 / (2.0 + 1.2));
    assert_eq!(embedding[1].value, 1.0);
    assert_eq!(embedder.query("a a b").terms()[0].value, 2.0);
}

#[test]
fn disabled_length_normalization_ignores_extreme_avgdl() {
    let embedder = EmbedderBuilder::<u32, _>::with_avgdl(f32::from_bits(1))
        .tokenizer(CountingTokenizer::default())
        .b(0.0)
        .build();
    assert!(embedder.embed("a b c").iter().all(|t| t.value == 1.0));
    let embedder = EmbedderBuilder::<u32, _>::with_avgdl(f32::from_bits(1))
        .tokenizer(CountingTokenizer::default())
        .k1(0.0)
        .build();
    assert!(embedder.embed("a a b").iter().all(|t| t.value == 1.0));
}

#[test]
fn large_parallel_build_matches_serial_embedding_statistics() {
    let calls = Arc::new(AtomicUsize::new(0));
    let docs: Vec<_> = (0..1050_u32)
        .map(|i| Document::new(i, if i % 2 == 0 { "a a b" } else { "b c" }))
        .collect();
    let engine = SearchEngineBuilder::<u32, u32, _>::with_tokenizer_and_documents(
        CountingTokenizer(calls.clone()),
        docs,
    )
    .build_frozen();
    assert_eq!(calls.load(Ordering::Relaxed), 1050);
    assert_eq!(engine.embedder().avgdl(), 2.5);
    let hits = engine.search("a", 7);
    assert_eq!(
        hits.iter().map(|h| h.document.id).collect::<Vec<_>>(),
        [0, 2, 4, 6, 8, 10, 12]
    );
}
