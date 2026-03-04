import { describe, it, expect } from "vitest";
import { StorageEngineV1 } from "../storage/StorageEngineV1";
import { MetadataStoreV1 } from "../metadata/MetadataStoreV1";
import { IVFFlatIndex } from "./IVFFlatIndex";

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bruteTopKDot(store: StorageEngineV1, query: Float32Array, k: number) {
  const dim = store.dim();
  const maxId = store.maxInternalIdExclusive();
  const scores: Array<{ id: number; score: number }> = [];

  for (let id = 0; id < maxId; id++) {
    if (!store.hasInternalId(id)) continue;
    const v = store.getVectorViewByInternalId(id);
    if (!v) continue;
    let s = 0;
    for (let j = 0; j < dim; j++) s += v[j] * query[j];
    scores.push({ id, score: s });
  }

  scores.sort((a, b) => (b.score - a.score) || (a.id - b.id));
  return scores.slice(0, k).map((x) => x.id);
}

describe("IVFFlatIndex recall", () => {
  it("D: recall@k is non-decreasing as nprobe increases", async () => {
    const dim = 64;
    const N = 10_000;
    const Q = 20;
    const k = 10;
    const nlist = 64;
    const nprobes = [1, 2, 4, 8, 16, nlist];

    const rng = mulberry32(555);
    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 1024);
    ivf.trainAndBuild({ nlist, iters: 12, seed: 1234, sampleSize: 20000 });

    const queries: Float32Array[] = [];
    for (let i = 0; i < Q; i++) {
      const q = new Float32Array(dim);
      for (let j = 0; j < dim; j++) q[j] = (rng() * 2 - 1) * 0.5;
      queries.push(q);
    }

    const bf = queries.map((q) => bruteTopKDot(store, q, k));

    const recalls: number[] = [];
    for (const nprobe of nprobes) {
      let hit = 0;
      for (let i = 0; i < Q; i++) {
        const res = await ivf.search(queries[i], { k, nprobe });
        const ids = res.map((x) => x.internalId);
        const set = new Set(bf[i]);
        for (const id of ids) if (set.has(id)) hit++;
      }
      recalls.push(hit / (Q * k));
    }

    for (let i = 1; i < recalls.length; i++) {
      expect(recalls[i] + 1e-6).toBeGreaterThanOrEqual(recalls[i - 1]);
    }
  });
});
