# WispDB API Guide
This is the “how do I actually use this thing” doc. No marketing poetry, no vague diagrams. Just the API shapes, what they do, and examples that won’t betray you at runtime.
---
## Concepts (so names make sense)
### Vector
A `Float32Array` of fixed length `dim` (the embedding dimension).
### External ID vs Internal ID
- **externalId**: your stable string ID (`"user_123"`, `"doc:abc"`)
- **internalId**: WispDB’s stable `int32` index used internally for storage and speed
You will see `internalId` in search results and debug output. It’s useful for filters, metadata, and perf.
### Metrics
- `"dot"`: dot product
- `"cosine"`: dot on normalized vectors (either normalized at query-time or pre-normalized at ingest)
- `"l2"`: distance-based ranking (often implemented as negative L2² so higher is better)
---
## Packages
- `wispdb`  
  Storage engine, metadata, brute-force search, IVF-Flat index, persistence wrapper, worker-friendly shapes.
- `@wispdb/gpu`  
  WebGPU runtime + kernels used internally by core (most users don’t touch this directly).
---
## Quickstart (main thread)
### Brute-force exact search (GPU if available)
import { BruteForceIndex } from "wispdb";
const dim = 128;
const db = new BruteForceIndex({
  dim,
  metric: "dot",
  preferGPU: true,
  batchRows: 8192, // scan chunk size
});
await db.open();
db.upsert("hello", new Float32Array(dim).fill(0.01), { tag: "demo" });
const hits = await db.search(new Float32Array(dim).fill(0.01), {
  k: 10,
});
console.log(hits);
---
## Core API
### `BruteForceIndex`
Exact search baseline. Scores everything (in batches) and returns deterministic TopK.
#### Constructor
new BruteForceIndex({
  dim: number,
  metric: "dot" | "cosine" | "l2",
  preferGPU?: boolean,
  batchRows?: number,
  kMax?: number
})
**Notes**
* `batchRows` controls memory safety and smoothness during scanning.
* `kMax` caps the maximum k supported by the TopK path.
#### Methods
##### `open(): Promise<void>`
Initializes runtime. On WebGPU-capable browsers this prepares device/pipelines.
##### `upsert(id: string, vec: Float32Array, meta?: any): number`
Insert or update a vector. Returns `internalId`.
##### `delete(id: string): boolean`
Marks an ID as deleted (tombstone). Returns whether it existed.
##### `search(query: Float32Array, opts: SearchOptions): Promise<SearchHit[]>`
**Options**
type SearchOptions = {
  k: number;
  where?: WhereClause;
  scoreThreshold?: number | null;
};
**Result**
type SearchHit = {
  id: string;
  internalId: number;
  score: number;
};
---
### `PersistentBruteForceDB`
Brute-force DB + persistence in one wrapper.
* `upsert/delete` append to a journal first
* snapshots are versioned and crash-tolerant
* reload restores from latest complete snapshot + journal replay
#### Constructor
new PersistentBruteForceDB(dbName: string, {
  dim,
  metric,
  preferGPU?: boolean,
  batchRows?: number,
  kMax?: number
})
#### Methods
##### `open(): Promise<void>`
Opens IndexedDB and restores state, then initializes runtime.
##### `upsert(id: string, vec: Float32Array, meta?: any): Promise<number>`
Journal append + in-memory upsert. Returns `internalId`.
##### `delete(id: string): Promise<boolean>`
Journal append + tombstone.
##### `search(query: Float32Array, opts: SearchOptions): Promise<SearchHit[]>`
Same shape as `BruteForceIndex.search`.
##### `snapshotNow(): Promise<string>`
Forces a snapshot (compaction-friendly). Returns `snapshotId`.
**Typical usage**
* call `snapshotNow()` after bulk ingest
* call it occasionally after churn (updates/deletes)
---
## Metadata + filtering API
Filtering is intentionally simple and deterministic.
### `where` clause (common shapes)
const hits = await db.search(q, {
  k: 10,
  where: {
    premium: true,                 // boolean
    year: { gte: 2022, lte: 2026 }, // numeric range
    lang: { in: ["en", "fr"] },     // small enum set
  }
});
**How it runs**
1. CPU prefilter candidate IDs from metadata
2. GPU scores candidates and selects top-k
**Tip**
* Keep “hot” fields (booleans, ranges, small enums) in column form internally.
* Dump everything else into the JSON row store.
---
## IVF-Flat API (first real speedup)
IVF-Flat trades a small amount of recall for much lower scan cost.
### `IVFFlatIndex`
This is the “index layer” that sits on top of stored vectors.
In your current setup it’s usually built alongside a DB that owns the storage and metadata.
#### Training + build
ivf.trainAndBuild({
  nlist: 256,       // number of centroids/lists
  iters: 20,
  seed: 1234,
  sampleSize: 20000
});
#### Search
const hits = await ivf.search(query, {
  k: 10,
  nprobe: 8,          // how many lists to scan
  where: { premium: true }
});
**How to tune**
* Increase `nprobe` → higher recall, slower search
* Decrease `nprobe` → faster search, lower recall
* `nprobe = nlist` should match brute force exactly (good test)
---
## Worker-first API (recommended)
This is the default “don’t freeze UI” path:
* Worker owns DB + IndexedDB
* main thread does UI only
* vectors and queries are sent as Transferable `ArrayBuffer` payloads
### `WispWorkerClient` (main thread)
import { WispWorkerClient } from "./worker/client";
const client = new WispWorkerClient();
await client.open("demo", 128, "dot", true);
// upsert in batches
await client.upsertBatch(
  ["id_0", "id_1"],
  new Float32Array(2 * 128),
  128,
  [{ tag: "a" }, { tag: "b" }]
);
const res = await client.search(new Float32Array(128), 10, { tag: { in: ["a"] } });
console.log(res.hits, res.tookMs);
await client.snapshot();
### RPC methods (client)
* `ping(): Promise<{ worker: true; crossOriginIsolated: boolean; userAgent: string }>`
* `open(dbName: string, dim: number, metric: Metric, preferGPU?: boolean)`
* `upsertBatch(ids: string[], vectors: Float32Array, dim: number, metas?: any[])`
* `search(query: Float32Array, k: number, where?: any, scoreThreshold?: number | null)`
* `snapshot()`
**Important**
`upsertBatch()` and `search()` should transfer the underlying `ArrayBuffer` so you don’t copy megabytes around.
---
## SharedArrayBuffer fast path (optional)
If `crossOriginIsolated === true`, you can use `SharedArrayBuffer` for zero-copy streaming.
### Vite dev headers
// vite.config.ts
export default {
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
};
Then you can upgrade the worker protocol to use shared buffers (ring buffer pattern).
This is optional for deployment, but great for high-frequency query streaming.
---
## Determinism rules (what WispDB guarantees)
* Same data + same query + same metric + same filter → same top-k IDs
* Tie-breaking is stable (internalId order used as the final tie-breaker)
* IVF is approximate by design, but should remain deterministic for a fixed seed and fixed state
---
## Performance tips (what actually matters)
* **Batch your ingests.** Use `upsertBatch` in the worker.
* Call `snapshotNow()` after large ingests or heavy churn.
* For cosine: pre-normalize vectors at ingest if you want speed.
* Use IVF when `N` grows. Start with `nlist=256`, `nprobe=8` at 50k/128d and tune from there.
* Never do big data generation loops on the main thread if your DoD is “no stutter”.
---
## Common errors
### Query dimension mismatch
You gave a query vector with the wrong length.
Fix: ensure `query.length === dim`.
### No WebGPU
Browser doesn’t support WebGPU or it’s disabled. Core should fall back to CPU.
Fix: test with Chrome/Edge, enable flags if needed, or accept CPU.
### Persistence doesn’t restore
Usually caused by:
* different `dim` / chunkRows between runs
* clearing site data
* switching dbName
Fix: keep your DB config stable or use a new name.
---
## Minimal “DB interface” you can rely on
If you want your app code to stay clean, treat both brute-force and IVF-backed DBs as:
type VecDB = {
  open(): Promise<void>;
  upsert(id: string, vec: Float32Array, meta?: any): any;
  delete(id: string): any;
  search(query: Float32Array, opts: { k: number; where?: any; scoreThreshold?: number | null }): Promise<any[]>;
};
WispDB’s internals can evolve without forcing your UI to rewrite itself every milestone.
---
