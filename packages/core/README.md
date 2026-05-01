# WispDB

WispDB is a small TypeScript vector database for JavaScript apps. It gives you an easy `WispDB.open()` API for in-memory semantic search, with support for cosine, dot-product, and L2 distance.

It is designed to run in Node.js and browser-based applications, and the repository also includes lower-level APIs for WebGPU search, IndexedDB persistence, workers, IVF-Flat indexes, and benchmarks.

## Install

WispDB is published on npm:

https://www.npmjs.com/package/wispdb

Install it with your package manager:

```bash
npm install wispdb
```

```bash
pnpm add wispdb
```

```bash
yarn add wispdb
```

WispDB ships as an ESM package with TypeScript declarations.

## Quick Start

```ts
import { WispDB } from "wispdb";

const db = await WispDB.open({
  dimensions: 3,
  metric: "cosine",
});

await db.upsert("one", [1, 0, 0], { name: "First" });
await db.upsert("two", [0, 1, 0], { name: "Second" });

const results = await db.search([1, 0, 0], { k: 1 });

console.log(results);
```

Output:

```ts
[
  {
    id: "one",
    score: 1,
    metadata: { name: "First" },
  },
]
```

## Core API

### Open a Database

```ts
const db = await WispDB.open({
  dimensions: 384,
  metric: "cosine",
});
```

Options:

| Option | Type | Description |
| --- | --- | --- |
| `dimensions` | `number` | Required vector dimension. Every inserted vector and query must match this length. |
| `metric` | `"cosine" \| "dot" \| "l2"` | Similarity metric. Defaults to `"cosine"`. |

### Insert or Update Vectors

```ts
await db.upsert("doc-1", embedding, {
  title: "Vector search overview",
  source: "docs",
});
```

`upsert(id, vector, metadata)` inserts a new vector or replaces the vector for an existing ID.

Supported vector inputs:

```ts
await db.upsert("a", [1, 0, 0]);
await db.upsert("b", new Float32Array([0, 1, 0]));
```

### Search

```ts
const matches = await db.search(queryEmbedding, {
  k: 5,
});
```

Each result contains:

```ts
type SearchResult = {
  id: string;
  score: number;
  metadata: Record<string, unknown> | undefined;
};
```

You can also apply a minimum score:

```ts
const matches = await db.search(queryEmbedding, {
  k: 10,
  scoreThreshold: 0.75,
});
```

### Size

```ts
const count = db.size();
```

## Example: Searching Text Embeddings

WispDB does not generate embeddings for you. Create embeddings with your preferred model, then store and search them:

```ts
import { WispDB } from "wispdb";

const db = await WispDB.open({
  dimensions: 1536,
  metric: "cosine",
});

await db.upsert("intro", introEmbedding, {
  title: "Introduction",
  url: "/docs/intro",
});

await db.upsert("api", apiEmbedding, {
  title: "API Reference",
  url: "/docs/api",
});

const results = await db.search(questionEmbedding, { k: 3 });
```

## Metrics

WispDB supports three distance or similarity modes:

| Metric | Best for | Notes |
| --- | --- | --- |
| `cosine` | Most embedding models | Vectors are normalized before storage/search. Higher score is better. |
| `dot` | Pre-normalized embeddings or custom scoring | Higher score is better. |
| `l2` | Euclidean distance use cases | Internally ranked so better matches appear first. |

## Advanced APIs

The package also exports lower-level building blocks used by the playground and benchmark suite:

```ts
import {
  BruteForceIndex,
  IVFFlatIndex,
  PersistentBruteForceDB,
} from "wispdb";
```

Use these when you need direct control over indexing, persistence, or browser worker workflows.

### BruteForceIndex

Exact search over every live vector. This is useful for correctness checks, smaller datasets, and benchmark baselines.

### IVFFlatIndex

Approximate search using IVF-Flat. It trains centroids, probes selected lists, and reranks candidates.

### PersistentBruteForceDB

Browser-oriented persistent storage using IndexedDB snapshots and journal replay.

## Browser Notes

The simple `WispDB` API works without WebGPU. Lower-level GPU features require browser WebGPU support.

For worker and zero-copy browser flows, configure cross-origin isolation headers:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The included Vite playground already sets these headers.

## Development

Clone the repository and install dependencies:

```bash
pnpm install
```

Build all packages:

```bash
pnpm build
```

Run type checks:

```bash
pnpm typecheck
```

Run lint:

```bash
pnpm lint
```

Run unit tests:

```bash
pnpm test
```

Run the browser playground tests:

```bash
pnpm e2e
```

Run benchmarks:

```bash
pnpm bench:cpu
pnpm bench:browser
pnpm bench:gate
```

## Repository Layout

```text
apps/
  playground/        Vite playground, browser benchmarks, Playwright tests
packages/
  core/              Public wispdb package and core index/storage logic
  gpu/               WebGPU runtime and kernels
  bench/             CPU benchmark harness
  utils/             Shared utilities
scripts/
  bench/             Benchmark gate checker
```

## Package Build

The npm package is built with `tsup` from `packages/core`:

```bash
pnpm --filter wispdb build
```

Create a local tarball:

```bash
pnpm --filter wispdb pack
```

## License

ISC
