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

function percentile(xs: number[], p: number) {
  if (xs.length === 0) return 0;
  const a = xs.slice().sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.floor(p * a.length)));
  return a[idx];
}

async function measureSearches(
  queries: Float32Array[],
  fn: (q: Float32Array) => Promise<any>,
) {
  const times: number[] = [];
  for (const q of queries) {
    const t0 = performance.now();
    await fn(q);
    times.push(performance.now() - t0);
  }
  return {
    median: percentile(times, 0.5),
    p95: percentile(times, 0.95),
  };
}

describe("IVF churn soak (CPU)", () => {
  it("keeps search latency bounded and lists/liveSet consistent under churn", async () => {
    const dim = 64;
    const env = (globalThis as any).process?.env ?? {};
    const N = Number(env.WISP_CHURN_N ?? (env.CI ? 10000 : 50000));
    const OPS = Number(env.WISP_CHURN_OPS ?? (env.CI ? 40000 : 200000));
    const SEARCH_EVERY = Number(env.WISP_CHURN_SEARCH_EVERY ?? 1000);
    const Q = 10;
    const k = 10;
    const nlist = 64;
    const nprobe = 8;
    const p95Factor = Number(env.WISP_CHURN_P95_FACTOR ?? 2.0);

    const rng = mulberry32(1234);
    const store = new StorageEngineV1(dim);
    const meta = new MetadataStoreV1({}, 4096);

    const liveIds: string[] = [];
    let nextExternal = 0;

    for (let i = 0; i < N; i++) {
      const id = `id_${nextExternal++}`;
      store.upsert(id, randVec(rng, dim));
      liveIds.push(id);
    }

    const ivf = new IVFFlatIndex(store, meta, null, "dot", 1024);
    ivf.trainAndBuild({ nlist, iters: 10, seed: 42, sampleSize: N });

    const queries: Float32Array[] = [];
    for (let i = 0; i < Q; i++) queries.push(randVec(rng, dim));

    await ivf.search(queries[0], { k, nprobe });
    const baseline = await measureSearches(queries, (q) => ivf.search(q, { k, nprobe }));
    let maxP95 = baseline.p95;

    const maintain = () => {
      const s = store.stats().vectors;
      if (s.deletedCount > s.liveCount * 0.5) {
        ivf.rebuildLists();
      }
      store.compactTrailingDead();
    };

    for (let op = 1; op <= OPS; op++) {
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

      if (op % SEARCH_EVERY === 0) {
        const stats = await measureSearches(queries, (q) => ivf.search(q, { k, nprobe }));
        if (stats.p95 > maxP95) maxP95 = stats.p95;
        maintain();
      }
    }

    const endStats = store.stats().vectors;
    const liveSetSize = (store as any).store["liveSet"].size();
    expect(liveSetSize).toBe(endStats.liveCount);

    const sizes = ivf.getListSizes();
    let sum = 0;
    for (let i = 0; i < sizes.length; i++) sum += sizes[i];
    expect(sum).toBe(endStats.liveCount);

    expect(maxP95).toBeLessThanOrEqual(baseline.p95 * p95Factor);
  }, 120000);
});
