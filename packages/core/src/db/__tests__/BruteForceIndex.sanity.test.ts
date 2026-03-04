import { describe, it, expect } from "vitest";
import { BruteForceIndex } from "../BruteForceIndex";

function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 4294967296) * 2 - 1;
  };
}

function normalize(v: Float32Array) {
  let ss = 0;
  for (let i = 0; i < v.length; i++) ss += v[i] * v[i];
  const inv = 1 / Math.sqrt(ss || 1e-12);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

function bruteTopK(store: any, query: Float32Array, k: number, metric: "dot" | "cosine" | "l2") {
  const dim = store.dim();
  const maxId = store.maxInternalIdExclusive();
  const q = metric === "cosine" ? normalize(query) : query;

  const scores: Array<{ id: number; score: number }> = [];
  for (let id = 0; id < maxId; id++) {
    if (!store.hasInternalId(id)) continue;
    const v = store.getVectorViewByInternalId(id);
    if (!v) continue;

    let s = 0;
    if (metric === "dot") {
      for (let j = 0; j < dim; j++) s += v[j] * q[j];
    } else if (metric === "l2") {
      let acc = 0;
      for (let j = 0; j < dim; j++) {
        const d = v[j] - q[j];
        acc += d * d;
      }
      s = -acc;
    } else {
      for (let j = 0; j < dim; j++) s += v[j] * q[j];
    }
    scores.push({ id, score: s });
  }

  scores.sort((a, b) => (b.score - a.score) || (a.id - b.id));
  return scores.slice(0, k).map((x) => x.id);
}

describe("BruteForceIndex sanity (CPU deterministic)", () => {
  const metrics: Array<"dot" | "cosine" | "l2"> = ["dot", "cosine", "l2"];

  for (const metric of metrics) {
    it(`matches brute force for ${metric} and is deterministic with ties`, async () => {
      const dim = 32;
      const k = 10;
      const db = new BruteForceIndex({ dim, metric, preferGPU: false });
      const r = rng(123);

      // Insert random vectors + a tie block
      for (let i = 0; i < 2000; i++) {
        const v = new Float32Array(dim);
        for (let j = 0; j < dim; j++) v[j] = r() * 0.5;
        db.upsert(`id_${i}`, v);
      }

      // Add vectors that tie on dot/cosine (same vector)
      const tieVec = new Float32Array(dim);
      for (let j = 0; j < dim; j++) tieVec[j] = 0.01;
      db.upsert("tie_a", tieVec);
      db.upsert("tie_b", tieVec);

      const q = new Float32Array(dim);
      for (let j = 0; j < dim; j++) q[j] = r() * 0.5;

      const a = await db.search(q, k);
      const b = await db.search(q, k);
      expect(a.map((x) => x.internalId)).toEqual(b.map((x) => x.internalId));
      expect(a.map((x) => x.score)).toEqual(b.map((x) => x.score));

      // Compare to brute force baseline
      const store = (db as any).store;
      const bf = bruteTopK(store, q, k, metric);
      expect(a.map((x) => x.internalId)).toEqual(bf);

      // Optional GPU vs CPU if WebGPU available
      if ((globalThis as any).navigator?.gpu) {
        const dbGpu = new BruteForceIndex({ dim, metric, preferGPU: true });
        for (let i = 0; i < 2000; i++) {
          const v = new Float32Array(dim);
          for (let j = 0; j < dim; j++) v[j] = r() * 0.5;
          dbGpu.upsert(`id_${i}`, v);
        }
        dbGpu.upsert("tie_a", tieVec);
        dbGpu.upsert("tie_b", tieVec);
        await dbGpu.open();
        const gpuRes = await dbGpu.search(q, k);
        expect(gpuRes.map((x) => x.internalId)).toEqual(bf);
      }
    });
  }
});
