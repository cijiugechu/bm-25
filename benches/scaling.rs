use bm_25::{Embedding, Query, Scorer, SearchScratch, TokenEmbedding as Token};
use divan::{AllocProfiler, Bencher};
use std::collections::{HashMap, HashSet};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

// Reproduces the original hash-set candidates / repeated IDF / linear vector lookup /
// full-sort path. Both implementations receive identical canonical vectors, so this
// comparison intentionally excludes the extra gain from removing duplicate entries.
struct Legacy {
    docs: HashMap<u32, Embedding>,
    inverted: HashMap<u32, HashSet<u32>>,
}
impl Legacy {
    fn new(docs: &[Embedding]) -> Self {
        let mut legacy = Self {
            docs: HashMap::new(),
            inverted: HashMap::new(),
        };
        for (id, doc) in docs.iter().enumerate() {
            for token in doc.indices() {
                legacy.inverted.entry(*token).or_default().insert(id as u32);
            }
            legacy.docs.insert(id as u32, doc.clone());
        }
        legacy
    }
    fn search(&self, query: &[u32], limit: usize) -> Vec<(u32, f32)> {
        let candidates: HashSet<_> = query
            .iter()
            .filter_map(|q| self.inverted.get(q))
            .flat_map(|ids| ids.iter())
            .collect();
        let mut scores: Vec<_> = candidates
            .into_iter()
            .map(|id| {
                let doc = &self.docs[id];
                let mut score = 0.0;
                for token in query {
                    let df = self.inverted.get(token).map_or(0, |ids| ids.len()) as f32;
                    let idf = (1.0 + (self.docs.len() as f32 - df + 0.5) / (df + 0.5)).ln();
                    score += idf
                        * doc
                            .iter()
                            .find(|t| t.index == *token)
                            .map_or(0.0, |t| t.value);
                }
                (*id, score)
            })
            .collect();
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scores.truncate(limit);
        scores
    }
}

fn corpus(n: usize, density: usize) -> Vec<Embedding> {
    (0..n)
        .map(|id| {
            let mut terms: Vec<_> = (0..40)
                .map(|j| Token::new(100 + ((id * 37 + j) % 2048) as u32, 1.0))
                .collect();
            if id % 128 < density {
                terms.push(Token::new(1, 0.5 + (id % 17) as f32 / 10.0));
                terms.push(Token::new(2, 0.25 + (id % 13) as f32 / 10.0));
            }
            Embedding::new(terms)
        })
        .collect()
}

#[divan::bench(args = [1_000, 10_000, 100_000], consts = [1, 16, 64, 128], sample_count = 30)]
fn legacy<const DENSITY: usize>(bencher: Bencher, n: usize) {
    let docs = corpus(n, DENSITY);
    let legacy = Legacy::new(&docs);
    drop(docs);
    bencher.bench(|| legacy.search(&[1, 2], 20));
}

#[divan::bench(args = [1_000, 10_000], sample_count = 20)]
fn update_and_refreeze(bencher: Bencher, n: usize) {
    let docs = corpus(n, 16);
    bencher
        .with_inputs(|| {
            let mut scorer = Scorer::new();
            for (id, doc) in docs.iter().enumerate() {
                scorer.upsert(&(id as u32), doc.clone());
            }
            scorer.snapshot();
            scorer
        })
        .bench_values(|mut scorer| {
            scorer.remove(&0);
            scorer.upsert(&1, Embedding::new([Token::new(1, 2.0)]));
            divan::black_box(scorer.into_frozen());
        });
}

#[divan::bench(args = [1_000, 10_000, 100_000], consts = [1, 16, 64, 128], sample_count = 30)]
fn prepared<const DENSITY: usize>(bencher: Bencher, n: usize) {
    let docs = corpus(n, DENSITY);
    let mut scorer = Scorer::new();
    for (id, doc) in docs.into_iter().enumerate() {
        scorer.upsert(&(id as u32), doc);
    }
    let index = scorer.into_frozen();
    let query = index.prepare(&Query::new([Token::new(1, 1.0), Token::new(2, 1.0)]));
    let mut scratch = SearchScratch::default();
    let mut out = Vec::new();
    index.search_into(&query, 20, &mut scratch, &mut out);
    bencher.bench_local(|| {
        index.search_into(&query, 20, &mut scratch, &mut out);
        divan::black_box(&out);
    });
}
