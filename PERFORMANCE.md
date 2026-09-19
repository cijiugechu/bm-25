# Performance implementation and validation

The read path resolves external terms once, then reads flat posting arenas. Each
block covers 128 consecutive document IDs. Below 64 entries it stores one-byte local
positions and f32 weights; otherwise it stores 128 contiguous weights. Empty dense
lanes are zero. A block tag selects the kernel outside the numeric loop. Sparse
results use a local touched bitmap instead of scanning all document slots.

Strict top-k maintains a bounded heap and skips a document block only when its
conservative score bound is below the current threshold. Full-result searches use
a final sort. Scores are accumulated in query term order; ties prefer smaller local
IDs. This is aligned-block pruning, not the full WAND pivot algorithm. The density
threshold is an initial speed/space choice, not a universal optimum.

No source uses unsafe code, explicit CPU intrinsics, quantized weights, or global
fast-math switches. `Relaxed` is opt-in algebraic arithmetic and disables pruning.

Posting traversal emits a reusable batch of matching block/factor descriptors in
query order. Bounds and scoring consume only these descriptors. The scan path
advances cursors and caches the next minimum block in one pass. The heap path
maintains only active streams, ordered by `(document block, query ordinal)`, so it
preserves strict addition order without sorting each batch. For M matching term
blocks and Q terms, heap scheduling costs O((Q + M) log Q), rather than scanning Q
cursors for each visited document block. Both paths retain O(Q) scratch capacity.

`TraversalMode::Auto` uses scanning for up to 16 prepared terms. For larger queries,
it estimates overlap as total term blocks divided by the span from first to last
document block, and uses a heap if this is below Q/4. Gaps can bias this estimate;
it is a heuristic, not a universal crossover. `set_traversal_mode` lets applications
compare/override it, and `stats().traversal` reports the chosen strategy. Dense
scoring still uses the same contiguous SIMD kernel. No arithmetic relaxation was
needed for the traversal improvement.

Query construction uses linear deduplication below 128 distinct terms, then a
lazy hash index. This bounds the initial linear work and makes larger construction
expected linear in the number of input terms (subject to hashing costs/collisions).
`clear`/`push`/`extend` reuse both allocations. Large queries retain cloned keys and
hash-table storage; prepared queries now cache posting ranges directly. These
trade additional query/workspace memory for fewer repeated lookups and scans.

## Long-query follow-up

The following sequential comparison uses preserved executables of the immediately
preceding 4.0.0 implementation and the updated implementation, on the same M4 host.
Each cell is the median of nine groups of warmed iterations; construction and
preparation are outside search timing. There is no allocation profiler in these
diagnostics. The host was not isolated, so these are observations, not latency
guarantees. Earlier absolute times varied substantially; compare the paired run.

`experiments/query_fanout.rs` fixes 131,072 documents, 1,024 nonzero postings and
1,024 matching document blocks. Increasing Q partitions those same postings among
more terms. Scores tie within each case, and assertions verify zero pruned blocks.

| Query terms | Previous top-20 | Updated top-20 |
|---:|---:|---:|
| 2 | 31.094 µs | 32.236 µs |
| 8 | 60.300 µs | 38.741 µs |
| 32 | 196.922 µs | 41.482 µs |
| 128 | 841.802 µs | 48.969 µs |
| 512 | 2.640 ms | 77.385 µs |
| 1,024 | 4.738 ms | 84.729 µs |

At 1,024 terms, Query construction including allocation/destruction went from
355.391 to 71.078 µs. This is separate from the prepared search improvement.
The two-term diagnostic was about 4% slower in this run; short queries are not
claimed to improve universally.

`experiments/query_overlap.rs` instead has 8,192 documents and 64 matching blocks,
each containing all Q terms in a single document. Auto selects scanning here.
Top-20 times for Q=2/128/1024 were 2.579/75.365/840.188 µs before and
2.562/69.469/756.385 µs after. This guards against assuming a heap is always faster.
Both diagnostics use unusually sparse documents to expose scheduling overhead;
they do not represent general text workloads or isolate SIMD's contribution.

Reproduce the current diagnostics:

```sh
cargo build --release --all-features
rustc --edition=2021 -C opt-level=3 experiments/query_fanout.rs --extern bm_25=target/release/libbm_25.rlib -L dependency=target/release/deps -o /tmp/bm25-query-fanout
/tmp/bm25-query-fanout
rustc --edition=2021 -C opt-level=3 experiments/query_overlap.rs --extern bm_25=target/release/libbm_25.rlib -L dependency=target/release/deps -o /tmp/bm25-query-overlap
/tmp/bm25-query-overlap
cargo bench --bench traversal --all-features
```

The permanent traversal benchmark compares Auto/Scan/Heap, disjoint/fully
overlapping postings (`OVERLAP=false/true`), and 2/32/128/1024 terms. It also measures
Query construction and reuse, with Divan's allocation profiler. Mixed-density
reference tests check exact scores and IDs for all strategies and multiple limits,
including full output. The heuristic can still need tuning for clustered or mixed
overlap; use the explicit modes to measure those workloads.

In the recorded permanent benchmark, 1,024-term disjoint searches took 78.22 µs
with Auto, 78.95 µs with Heap and 1.466 ms with Scan. With all terms present in
all 1,024 matching blocks, the same modes took 29.14 / 80.75 / 30.29 ms. These
larger fully overlapping cases differ from the 64-block diagnostic above. All
warmed search cases and Query-reuse cases reported no timed allocations; a reused
1,024-term Query took 32.39 µs to rebuild. Raw outputs are saved locally in
`target/performance/traversal.txt` and `query-{fanout,overlap}-{before,after}.csv`.

## Recorded measurements

Host: Apple M4, aarch64 macOS, rustc 1.98.1 / LLVM 22.1.8, optimized Cargo bench
profile, all features, Divan allocation profiler enabled. Times below are medians.
The machine was not isolated: substantial outliers and run-to-run variation were
observed. These are workload-specific observations, not promised speedups or tail
latencies. No x86 performance measurements have been made.

The final small-corpus comparison ran the original Git HEAD (3.0.0, extracted to a
temporary directory) and then the working 4.0.0 source sequentially. Both searched
the same 50 English recipes for `bacon sandwich`, limit 20, 100 samples.

| Operation | Original 3.0.0 | Current 4.0.0 |
|---|---:|---:|
| Owned search, fixed English | 3.095 µs | 1.975 µs |
| Owned search, language detection | 17.680 µs | 2.225 µs |
| Known-avgdl index creation, fixed English | 6.796 ms | 4.083 ms |
| Known-avgdl index creation, detection | 9.852 ms | 7.768 ms |
| Reused workspace, borrowed results, fixed English | unavailable | 1.291 µs |

The new index-creation bench includes immutable snapshot construction. Full frozen
builder creation, including fitting, took 3.775 ms in this run (no paired old full
builder measurement). Owned fixed-English search allocated 16 times versus 25;
detection allocated a median of 16 versus 332. The warmed borrowed-result benchmark
reported no allocations; other inputs, especially stemming/Unicode, may allocate.

`benches/scaling.rs` independently compares the original candidate-set / repeated
IDF / document-vector scan / full-sort algorithm with the new prepared-query path.
Both receive identical canonical vectors; this excludes duplicate-removal gains.
Each document has 40 background terms and, according to density, two matching
terms. The query has two terms, top-k=20. Corpus/index construction and preparation
are outside the timed query loop. There are 30 samples per case.

| Documents | Matching documents per 128 | Legacy median | Prepared median |
|---:|---:|---:|---:|
| 10,000 | 1 | 15.580 µs | 3.499 µs |
| 10,000 | 16 | 449.700 µs | 18.240 µs |
| 10,000 | 64 | 2.605 ms | 44.080 µs |
| 10,000 | 128 | 8.160 ms | 62.140 µs |
| 100,000 | 1 | 254.900 µs | 16.080 µs |
| 100,000 | 128 | 138.900 ms | 601.800 µs |

The prepared path reported zero timed allocations after warming the workspace.
The comparison includes algorithmic, layout, preparation, and allocation differences;
it does not isolate SIMD's contribution. Synthetic high-match cases favor top-k
over full sorting and are not representative of every production corpus.

Updates are a real tradeoff: deleting one document, replacing another, and consuming
the builder into a fresh snapshot measured 2.344 ms for 1,000 documents and 37.070 ms
for 10,000 in the synthetic workload. This includes rebuilding and dropping staging
storage, but excludes creating the initial builder. Batch updates or publish frozen
snapshots; do not infer cheap alternating writes/reads from the query benchmarks.

Raw local outputs are in the ignored `target/performance/` directory:
`baseline-paired.txt`, `current-paired.txt`, and `scaling-final.txt`.

## Reproduction

```sh
cargo bench --all-features --bench search
cargo bench --all-features --bench scaling
cargo rustc --release --all-features --lib -- --emit=asm,llvm-ir -C target-cpu=native
```

Inspect `target/release/deps/bm_25-*.s`. The actual product functions in
`src/kernels.rs` generated `fmul.4s` / `fadd.4s` for strict accumulation and
`fmla.4s` for relaxed accumulation on this host. `native` is only a local inspection
option; it is not committed as a distribution target. Safe scalar Rust remains
portable and other targets make their own vectorization choices.

## Correctness coverage

- Exhaustive reference scores compared bit-for-bit with strict block top-k across
  randomized density, query boosts, limits, and 0/1/63/64/127/128/129/513 documents.
- Block skipping, ties, tails, empty/zero/duplicate terms, invalid weights,
  overflow/subnormal scores, stale prepared queries, updates, and concurrent readers.
- Original search/tokenizer snapshots unchanged; embedding snapshots checked to
  differ only by removal of duplicate indices, retaining every remaining weight.
- Single-pass fitting, parallel builder ordering/statistics, borrowed results,
  repeated query frequencies, and disabled length-normalization edge cases.
- Relaxed scores compared with strict on finite representative data. This is not
  an error bound for arbitrary algebraic floating-point expressions.

```sh
cargo fmt -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
cargo test --release --all-features --test index
cargo test --no-default-features
```

All eight combinations of the three explicit Cargo features were also checked
with default features disabled. `language_detection` enables its tokenizer
dependency through the existing feature declaration.
