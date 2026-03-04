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

describe("IVF degeneracy: nprobe == nlist", () => {
  it("matches brute force top-k exactly (dot)", async () => {
    const dim = 64;
    const N = 4096;
    const k = 10;
    const nlist = 64;
    const Q = 10;

    const rng = mulberry32(12345);
    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 1024);
    ivf.trainAndBuild({ nlist, iters: 12, seed: 1234, sampleSize: N });

    for (let qi = 0; qi < Q; qi++) {
      const q = new Float32Array(dim);
      for (let j = 0; j < dim; j++) q[j] = (rng() * 2 - 1) * 0.5;

      const bf = bruteTopKDot(store, q, k);
      const ivfRes = await ivf.search(q, { k, nprobe: nlist });
      const ivfIds = ivfRes.map((x) => x.internalId);
      expect(ivfIds).toEqual(bf);
    }
  });
});
