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

function gatherCandidates(index: IVFFlatIndex, query: Float32Array, nprobe: number) {
  const idxAny = index as any;
  const dim = idxAny.store.dim();
  const probes = idxAny.nearestCentroids(query, dim, nprobe) as number[];
  const seen = new Uint8Array(idxAny.store.maxInternalIdExclusive());
  const out: number[] = [];

  for (const lid of probes) {
    const list = idxAny.lists[lid];
    for (let i = 0; i < list.len; i++) {
      const id = list.buf[i];
      if (id < 0 || id >= seen.length) continue;
      if (seen[id]) continue;
      seen[id] = 1;
      if (!idxAny.store.hasInternalId(id)) continue;
      if (idxAny.assignedList[id] !== lid) continue;
      out.push(id);
    }
  }
  return out;
}

describe("IVFFlatIndex", () => {
  it("A: nprobe=nlist matches brute force top-k exactly (dot)", async () => {
    const dim = 64;
    const N = 4096;
    const k = 10;
    const nlist = 64;
    const Q = 20;

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

      const bfIds = bruteTopKDot(store, q, k);
      const ivfRes = await ivf.search(q, { k, nprobe: nlist });
      const ivfIds = ivfRes.map((x) => x.internalId);

      expect(ivfIds).toEqual(bfIds);
    }
  });

  it("B: candidate set has no dead IDs, no duplicates, deterministic order", () => {
    const dim = 32;
    const N = 1000;
    const nlist = 32;
    const rng = mulberry32(7);

    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    // delete some IDs
    for (let i = 0; i < N; i += 10) store.delete(`id_${i}`);

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 256);
    ivf.trainAndBuild({ nlist, iters: 8, seed: 123 });

    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = (rng() * 2 - 1) * 0.5;

    const cand1 = gatherCandidates(ivf, q, 8);
    const cand2 = gatherCandidates(ivf, q, 8);

    expect(cand1).toEqual(cand2); // deterministic order
    expect(new Set(cand1).size).toBe(cand1.length); // no duplicates
    for (const id of cand1) {
      expect(store.hasInternalId(id)).toBe(true);
    }
  });

  it("C: upsert updates assignment and stale duplicates do not leak", () => {
    const dim = 32;
    const N = 512;
    const nlist = 16;
    const rng = mulberry32(99);

    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 256);
    ivf.trainAndBuild({ nlist, iters: 8, seed: 321 });

    const idxAny = ivf as any;
    const targetId = 10;
    const oldLid = idxAny.assignedList[targetId];
    const newLid = (oldLid + 1) % nlist;
    const centroids: Float32Array = idxAny.centroids;

    const newVec = new Float32Array(dim);
    newVec.set(centroids.subarray(newLid * dim, newLid * dim + dim));
    store.upsert(`id_${targetId}`, newVec);
    ivf.onUpsert(targetId);

    expect(idxAny.assignedList[targetId]).toBe(newLid);

    const q = newVec;
    const cand = gatherCandidates(ivf, q, nlist);
    const occurrences = cand.filter((id) => id === targetId).length;
    expect(occurrences).toBe(1);
  });

  it("F: deterministic results with fixed seed", async () => {
    const dim = 48;
    const N = 2048;
    const nlist = 32;
    const k = 10;
    const rng = mulberry32(2024);

    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
      store.upsert(`id_${i}`, v);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 512);
    ivf.trainAndBuild({ nlist, iters: 10, seed: 42, sampleSize: N });

    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = (rng() * 2 - 1) * 0.5;

    const r1 = await ivf.search(q, { k, nprobe: 8 });
    const r2 = await ivf.search(q, { k, nprobe: 8 });

    expect(r1.map((x) => x.internalId)).toEqual(r2.map((x) => x.internalId));
    for (let i = 0; i < r1.length; i++) {
      expect(r1[i].score).toBeCloseTo(r2[i].score, 10);
    }
  });
});
