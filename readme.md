# WispDB
A small vector database that runs in the browser, uses WebGPU for scoring, and doesn’t wipe itself on refresh.
---

## Why this exists
Most “vector search” demos are either:
- fast but inaccurate,
- accurate but slow,
- or they quietly freeze your UI and call it a feature.

WispDB is built to be:
- **correct** when it matters (exact search),
- **fast** when the dataset grows (IVF-Flat),
- **usable** in a real UI (Worker-first),
- **repeatable** (deterministic results),
- **measurable** (benchmarks + quality gates),
- **persistent** (IndexedDB snapshot + journal).

---

## What you can do right now

- Insert vectors with stable IDs
- Search with dot / cosine / L2
- Filter results using metadata
- Persist and reload without losing data
- Run everything off the main thread
- Get a real speedup with IVF-Flat
- Print benchmark numbers you can track

---

## Repository layout
apps/
playground/        demo app + browser benchmarks + e2e tests
packages/
core/              storage, metadata, indexes, persistence, worker protocol
gpu/               WebGPU runtime + kernels
bench/             CPU benchmark harness (CI-friendly)
scripts/
bench/             quality gate checker
---

## Install
pnpm install
---

## Run the demo
pnpm dev --filter playground
Open the local URL Vite prints.
---

## Basic usage
### Brute-force exact search
* Scores every vector (in chunks)
* Uses GPU when available
* Deterministic top-k

### IVF-Flat search
* Trains k-means centroids on CPU
* Assigns vectors into lists
* Probes a few lists per query
* GPU reranks the candidates exactly
---

## Metadata + filtering
Metadata is stored per internal ID.
Filtering happens before scoring:
1. CPU selects candidate IDs that match `where`
2. GPU scores and selects top-k
Example:
await db.search(q, {
  k: 10,
  where: {
    premium: true,
    year: { gte: 2022 },
    lang: { in: ["en", "fr"] }
  }
});

Hot fields can be configured as:
* booleans
* numeric ranges
* small enums

Everything else stays in a simple JSON row store.
---

## Persistence model
Persistence is IndexedDB-first.
* snapshots are versioned
* snapshots are written as “writing” then finalized as “complete”
* journal is append-only
* load = latest complete snapshot + journal replay
If a snapshot is interrupted mid-write, it stays “writing” and is ignored on next load.
---

## Worker-first runtime
WispDB can run inside a Worker.
* main thread sends requests
* worker does compute + persistence
* vectors and queries use Transferable ArrayBuffers to avoid copies
This is the default direction: no UI freezing.
---

## Benchmarks and quality gates
Benchmarks are not screenshots and vibes.
They print numbers.

### What gets measured
* ingest latency p50 / p95
* search latency p50 / p95
* recall@k vs brute force
* memory footprint (best effort)
* cold start and warm start

### Run the benchmarks
Smoke + correctness:
pnpm test

E2E (browser, worker, persistence):
pnpm e2e

CPU bench (CI-friendly):
pnpm --filter @wispdb/bench bench:cpu

Browser bench (pre-deploy):
pnpm --filter playground bench:browser

### Quality gates
A gate file defines acceptable ranges for key metrics.
Example fields:
* max allowed p95 search latency
* minimum recall@10 at a chosen nprobe
* minimum IVF speedup vs brute force
* max warm start time
* max frame-time p95 during scroll + search
A check script reads benchmark output and fails the build if a gate is violated.
---

## Notes
* WebGPU is not guaranteed on every device. WispDB falls back to CPU when needed.
* SharedArrayBuffer zero-copy requires `crossOriginIsolated`:
  * `Cross-Origin-Opener-Policy: same-origin`
  * `Cross-Origin-Embedder-Policy: require-corp`
---

## Authors

**Mahesh Chandra Teja Garnepudi**  
**Sagarika Srivastava**  

Built at **Kairais Tech**

---

## License

Private project for now. Licensing details will be published later.
