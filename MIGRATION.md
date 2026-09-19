# Migrating from 3.x to 4.0

Version 4 requires Rust 1.98. The crate version has been bumped for the following
source and behavior changes; this does not publish a release.

## Documents and queries

Construct document embeddings with `Embedding::new(entries)` instead of the tuple
constructor. The vector is read-only and contains one entry per distinct index,
in first-occurrence order. Duplicate entries retain the first weight; zero weights
are omitted, and negative/nonfinite weights panic. Mutable dereferencing is removed.
The embedder computes term frequency and original document length before deduplication.

Use `embedder.query(text)` for queries. It returns `Query`, which combines repeated
terms by adding their nonnegative boosts. `Query::new` also accepts explicit boosts.
No document-length normalization is performed for a query. Multiplying by a combined
boost can round differently from repeatedly adding identical contributions in 3.x.

`Query::new` and `Query::from(&embedding)` require `D: Eq + Hash + Clone`.
This matches the existing token embedding-space contract. Queries use a small
linear lookup initially and switch to a hash table for large distinct-term sets;
the table does not determine term order. Public `push(index, boost)` and
`extend(entries)` support incremental construction. Reuse `Query::clear()` and
`Embedder::query_into` to retain term and hash-table capacity across searches.

`Scorer::score` and `matches` remain adapters for embedding indices, ignoring their
values. Since embeddings are now unique, these adapters cannot recover original query
frequencies. Use `score_query`, `matches_query`, or a prepared query instead.

`TokenEmbedder` now requires `Sync`; its embedding space requires
`Eq + Hash + Clone + Send + Sync`. This supports parallel single-pass fitting.
The default external token hashes are unchanged: internal compact IDs do not
remove collisions in a custom/default external hash space.

BM25 builders reject nonfinite parameters, negative k1, and b outside `[0, 1]`.
Finite nonpositive avgdl still selects the existing fallback. Extreme parameters
that produce zero or nonfinite document weights panic during embedding.

## Index and update lifecycle

`Scorer` stages document vectors and caches an immutable `FrozenIndex`. Each write
invalidates the cache; `snapshot()` or the next search rebuilds it. Batch writes to
amortize this cost. `SearchEngineBuilder::build` eagerly creates its initial snapshot.
This favors batch updates/read-heavy use; an alternating write/search workload pays
for a full rebuild on each query.

`Scorer::into_frozen`, `SearchEngine::freeze`, and `build_frozen` discard mutable
forward vectors. Publish a new immutable instance for changes (an application may
use `Arc` to keep the previous instance alive while readers finish). There is no
background compactor or implicit asynchronous update visibility.

Snapshot statistics N and df are consistent with its document weights. Mutable
upserts retain the configured avgdl and BM25 parameters, just as before; rebuild
from text to recompute avgdl. Corpus builders tokenize once per input document,
including when the tokenizer is changed before `build`. An explicit `.avgdl(...)`
override disables fitting. With `parallelism`, corpora of at least 1024 documents
are tokenized in parallel, retaining document order and per-worker scratch buffers.

`PreparedQuery` is tied to exactly one snapshot. Reprepare after updates; stale
queries panic, including when the requested limit is zero. `DocId` is also local
to its originating snapshot and must not be used to resolve documents in another.
External IDs remain generic. Equal scores sort by increasing snapshot-local ID,
which follows live builder-slot order; deletion/reinsertion can change that order.

## Buffers and result ownership

Use `SearchWorkspace` with text `search_into` and `SearchScratch` with prepared
queries. Keep one workspace/output buffer per concurrent search; the index itself
is immutable and shared without a query mutex. Buffers grow as needed, then retain
their capacity. Text normalization/stemming can still allocate for some inputs.

`SearchResultRef` borrows the external ID and text. The convenient `search` method
still returns owned documents. Prepared-query search returns compact `Hit` values.
Resolve their IDs using the same `FrozenIndex::document_id` that produced them.

Existing custom tokenizers continue to work. Override `for_each_token` for the
allocation-saving path: borrowed token slices are consumed within a callback, so
normalization/stemming scratch never escapes its lifetime. `TokenizerScratch` is
reusable normalized-text storage for the default tokenizer.

## Posting traversal

`SearchScratch` defaults to `TraversalMode::Auto`: short or highly overlapping
queries use a scan, while long sparse queries use an active-cursor heap. Both
produce matching block batches in query order before scoring. This preserves
strict scores, conservative bounds and document-ID tie-breaking.

Use `scratch.set_traversal_mode(TraversalMode::Scan)` or `Heap` to benchmark or
override the heuristic for your workload. For text searches, set it on
`workspace.ranking`. Inspect `scratch.stats().traversal` for the selected strategy
and `matched_term_blocks` for visited term blocks, including pruned ones. Empty or
zero-limit searches report `Auto` because no strategy was selected. Scheduling
buffers grow with the number of query terms and are retained for reuse.

## Numerical policy

Strict mode keeps multiply and add separate and accumulates in unique query-term
order. Nonnegative monotone operations in the same order give conservative block
upper bounds; strict inequality when pruning preserves score ties. This is block
upper-bound pruning over aligned document ranges, not a WAND pivot implementation.

Relaxed mode uses Rust algebraic arithmetic with unspecified precision and disables
pruning. Treat it as an opt-in performance/quality experiment, not an exact-rank
guarantee. Positive scores can overflow to infinity; strict mode handles those ties.
Normal input validation rejects negative weights and NaN, which would invalidate
the bounds. The default does not use quantized weights or approximate reciprocals.
