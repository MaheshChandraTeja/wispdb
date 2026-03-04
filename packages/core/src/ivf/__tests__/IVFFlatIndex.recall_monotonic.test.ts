import { describe, it, expect } from "vitest";
import { StorageEngineV1 } from "../../storage/StorageEngineV1";
import { MetadataStoreV1 } from "../../metadata/MetadataStoreV1";
import { IVFFlatIndex } from "../IVFFlatIndex";

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

function recallAtK(exact: number[], approx: number[]) {
  const set = new Set(exact);
  let hit = 0;
  for (const id of approx) if (set.has(id)) hit++;
  return hit / exact.length;
}

describe("IVF recall monotonicity", () => {
  it("recall@k is non-decreasing as nprobe increases", async () => {
    const dim = 64;
    const N = 10000;
    const k = 10;
    const nlist = 64;
    const probes = [1, 2, 4, 8, 16, nlist];

    const rng = mulberry32(2024);
    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 1024);
    ivf.trainAndBuild({ nlist, iters: 10, seed: 1234, sampleSize: N });

    const queries: Float32Array[] = [];
    for (let i = 0; i < 10; i++) {
      const q = new Float32Array(dim);
      for (let j = 0; j < dim; j++) q[j] = (rng() * 2 - 1) * 0.5;
      queries.push(q);
    }

    let lastRecall = 0;
    for (const p of probes) {
      let rsum = 0;
      for (const q of queries) {
        const exact = bruteTopKDot(store, q, k);
        const approx = await ivf.search(q, { k, nprobe: p });
        rsum += recallAtK(exact, approx.map((x) => x.internalId));
      }
      const recall = rsum / queries.length;
      expect(recall + 1e-6).toBeGreaterThanOrEqual(lastRecall);
      lastRecall = recall;
    }
  });
});
