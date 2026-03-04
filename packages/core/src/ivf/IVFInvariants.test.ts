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

function randVec(rng: () => number, dim: number) {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = (rng() * 2 - 1) * 0.5;
  return v;
}

describe("IVF invariants after churn", () => {
  it("keeps assignedList + posInList consistent and no dead IDs in lists", () => {
    const dim = 32;
    const env = (globalThis as any).process?.env ?? {};
    const N = Number(env.WISP_IVF_INV_N ?? 10000);
    const OPS = Number(env.WISP_IVF_INV_OPS ?? 30000);
    const nlist = 32;

    const rng = mulberry32(999);
    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);
    const liveIds: string[] = [];
    let nextExternal = 0;

    for (let i = 0; i < N; i++) {
      const id = `id_${nextExternal++}`;
      store.upsert(id, randVec(rng, dim));
      liveIds.push(id);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 512);
    ivf.trainAndBuild({ nlist, iters: 8, seed: 7, sampleSize: N });

    for (let op = 0; op < OPS; op++) {
      const r = rng();
      if (r < 0.6 && liveIds.length > 0) {
        const idx = (rng() * liveIds.length) | 0;
        const id = liveIds[idx];
        const internal = store.upsert(id, randVec(rng, dim));
        ivf.onUpsert(internal);
      } else if (r < 0.8 && liveIds.length > 0) {
        const idx = (rng() * liveIds.length) | 0;
        const id = liveIds[idx];
        const internal = store.getInternalId(id);
        const ok = store.delete(id);
        if (ok && internal !== null) ivf.onDelete(internal);
        liveIds[idx] = liveIds[liveIds.length - 1];
        liveIds.pop();
      } else {
        const id = `id_${nextExternal++}`;
        const internal = store.upsert(id, randVec(rng, dim));
        ivf.onUpsert(internal);
        liveIds.push(id);
      }
    }

    const idxAny = ivf as any;
    const assigned: Int32Array = idxAny.assignedList;
    const pos: Int32Array = idxAny.posInList;
    const lists = idxAny.lists;

    const maxId = store.maxInternalIdExclusive();
    for (let id = 0; id < maxId; id++) {
      const live = store.hasInternalId(id);
      if (live) {
        const lid = assigned[id];
        expect(lid).toBeGreaterThanOrEqual(0);
        expect(lid).toBeLessThan(lists.length);
        const p = pos[id];
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThan(lists[lid].len);
        expect(lists[lid].buf[p]).toBe(id);
      } else {
        if (id < assigned.length) expect(assigned[id]).toBe(-1);
      }
    }

    const seen = new Set<number>();
    for (let lid = 0; lid < lists.length; lid++) {
      const list = lists[lid];
      for (let i = 0; i < list.len; i++) {
        const id = list.buf[i];
        expect(store.hasInternalId(id)).toBe(true);
        expect(assigned[id]).toBe(lid);
        expect(pos[id]).toBe(i);
        expect(seen.has(id)).toBe(false);
        seen.add(id);
      }
    }
  }, 60000);
});
