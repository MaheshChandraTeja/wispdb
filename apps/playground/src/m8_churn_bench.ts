import { PersistentBruteForceDB, IVFFlatIndex, type MaintenanceRequest, type MaintenanceResponse } from "@wispdb/core";

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

async function measureSearches(queries: Float32Array[], fn: (q: Float32Array) => Promise<any>) {
  const times: number[] = [];
  for (const q of queries) {
    const t0 = performance.now();
    await fn(q);
    times.push(performance.now() - t0);
  }
  return { median: percentile(times, 0.5), p95: percentile(times, 0.95) };
}

export async function runM8ChurnBench(log: (s: string) => void, opts?: Partial<{
  dim: number;
  N: number;
  ops: number;
  nlist: number;
  nprobe: number;
  k: number;
}>) {
  const dim = opts?.dim ?? 128;
  const N = opts?.N ?? 50000;
  const OPS = opts?.ops ?? 200000;
  const nlist = opts?.nlist ?? 64;
  const nprobe = opts?.nprobe ?? 8;
  const k = opts?.k ?? 10;

  const name = `m8_bench_${Date.now()}`;
  const db = new PersistentBruteForceDB(name, { dim, metric: "dot", preferGPU: true });
  (window as any).__wispdb = db;

  log(`Opening DB ${name}...`);
  await db.open();

  const rng = mulberry32(1234);
  const liveIds: string[] = [];
  let nextExternal = 0;

  log(`Ingesting ${N} vectors...`);
  for (let i = 0; i < N; i++) {
    const id = `id_${nextExternal++}`;
    await db.upsert(id, randVec(rng, dim));
    liveIds.push(id);
    if (i % 1000 === 0) {
      log(`... ${i}/${N}`);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  log("Ingest done.");

  const idxAny = (db as any).idx;
  const store = idxAny.store;
  const meta = idxAny.meta;
  const gpu = idxAny.gpu ?? null;
  const ivf = new IVFFlatIndex(store, meta, gpu, "dot", 1024);
  ivf.trainAndBuild({ nlist, iters: 10, seed: 42, sampleSize: N });

  const queries: Float32Array[] = [];
  for (let i = 0; i < 10; i++) queries.push(randVec(rng, dim));

  await db.search(queries[0], { k });
  await ivf.search(queries[0], { k, nprobe });

  log("Baseline searches...");
  const bfBase = await measureSearches(queries, (q) => db.search(q, { k }));
  const ivfBase = await measureSearches(queries, (q) => ivf.search(q, { k, nprobe }));
  log(`BF baseline: median ${bfBase.median.toFixed(2)} ms, p95 ${bfBase.p95.toFixed(2)} ms`);
  log(`IVF baseline: median ${ivfBase.median.toFixed(2)} ms, p95 ${ivfBase.p95.toFixed(2)} ms`);

  const worker = new Worker(
    new URL("@wispdb/core/maintenance/maintenance.worker.ts", import.meta.url),
    { type: "module" },
  );

  let reqId = 0;
  const send = (msg: MaintenanceRequest) =>
    new Promise<MaintenanceResponse>((resolve) => {
      const id = ++reqId;
      (msg as any).requestId = id;
      const handler = (e: MessageEvent<MaintenanceResponse>) => {
        if (e.data?.requestId === id) {
          worker.removeEventListener("message", handler);
          resolve(e.data);
        }
      };
      worker.addEventListener("message", handler);
      worker.postMessage(msg);
    });

  log(`Churn ops=${OPS} with maintenance worker...`);
  for (let op = 1; op <= OPS; op++) {
    const r = rng();
    if (r < 0.6 && liveIds.length > 0) {
      const idx = (rng() * liveIds.length) | 0;
      const id = liveIds[idx];
      const internal = await db.upsert(id, randVec(rng, dim));
      ivf.onUpsert(internal);
    } else if (r < 0.8 && liveIds.length > 0) {
      const idx = (rng() * liveIds.length) | 0;
      const id = liveIds[idx];
      const internal = store.getInternalId(id);
      const ok = await db.delete(id);
      if (ok && internal !== null) ivf.onDelete(internal);
      liveIds[idx] = liveIds[liveIds.length - 1];
      liveIds.pop();
    } else {
      const id = `id_${nextExternal++}`;
      const internal = await db.upsert(id, randVec(rng, dim));
      ivf.onUpsert(internal);
      liveIds.push(id);
    }

    if (op % 10000 === 0) {
      log(`... ops ${op}/${OPS}`);
      await send({
        type: "RUN_MAINTENANCE",
        dbName: `wispdb_${name}`,
        dim,
        policy: {
          trimTail: true,
          snapshotIfDeadRatio: 0.35,
          snapshotMinIntervalMs: 60_000,
          ivfRebuildIfDeadRatio: 0.5,
        },
      });
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  log("Post-churn searches...");
  const bfAfter = await measureSearches(queries, (q) => db.search(q, { k }));
  const ivfAfter = await measureSearches(queries, (q) => ivf.search(q, { k, nprobe }));
  log(`BF after: median ${bfAfter.median.toFixed(2)} ms, p95 ${bfAfter.p95.toFixed(2)} ms`);
  log(`IVF after: median ${ivfAfter.median.toFixed(2)} ms, p95 ${ivfAfter.p95.toFixed(2)} ms`);

  worker.terminate();
  log("Done.");
}
