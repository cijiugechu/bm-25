use bm_25::{Embedding, Query, Scorer, SearchScratch, TokenEmbedding as Token, TraversalMode};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

fn search<const OVERLAP: bool>(bencher: Bencher, terms: u32, mode: TraversalMode) {
    let mut scorer = Scorer::new();
    for doc in 0..131_072_u32 {
        let embedding = if doc % 128 != 0 {
            Embedding::new([])
        } else if OVERLAP {
            Embedding::new((0..terms).map(|term| Token::new(term, 1.0)))
        } else {
            Embedding::new([Token::new((doc / 128) % terms, 1.0)])
        };
        scorer.upsert(&doc, embedding);
    }
    let index = scorer.into_frozen();
    let query = index.prepare(&Query::new((0..terms).map(|term| Token::new(term, 1.0))));
    let mut scratch = SearchScratch::default();
    scratch.set_traversal_mode(mode);
    let mut out = Vec::new();
    index.search_into(&query, 20, &mut scratch, &mut out);
    assert_eq!(out.len(), 20);
    assert_eq!(scratch.stats().evaluated_blocks, 1024);
    assert_eq!(scratch.stats().skipped_blocks, 0);
    bencher.bench_local(|| {
        index.search_into(&query, 20, &mut scratch, &mut out);
        divan::black_box(&out);
    });
}

#[divan::bench(args = [2, 32, 128, 1024], consts = [false, true], sample_count = 30)]
fn auto<const OVERLAP: bool>(bencher: Bencher, terms: u32) {
    search::<OVERLAP>(bencher, terms, TraversalMode::Auto);
}

#[divan::bench(args = [2, 32, 128, 1024], consts = [false, true], sample_count = 30)]
fn scan<const OVERLAP: bool>(bencher: Bencher, terms: u32) {
    search::<OVERLAP>(bencher, terms, TraversalMode::Scan);
}

#[divan::bench(args = [2, 32, 128, 1024], consts = [false, true], sample_count = 30)]
fn heap<const OVERLAP: bool>(bencher: Bencher, terms: u32) {
    search::<OVERLAP>(bencher, terms, TraversalMode::Heap);
}

#[divan::bench(args = [2, 32, 128, 1024], sample_count = 30)]
fn query_build(bencher: Bencher, terms: u32) {
    bencher.bench(|| Query::new((0..terms).map(|term| Token::new(divan::black_box(term), 1.0))));
}

#[divan::bench(args = [2, 32, 128, 1024], sample_count = 30)]
fn query_reuse(bencher: Bencher, terms: u32) {
    let mut query = Query::new((0..terms).map(|term| Token::new(term, 1.0)));
    bencher.bench_local(|| {
        query.clear();
        query.extend((0..terms).map(|term| Token::new(divan::black_box(term), 1.0)));
        divan::black_box(&query);
    });
}
