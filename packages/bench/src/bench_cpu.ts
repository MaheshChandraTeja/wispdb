// @ts-ignore
import { BruteForceIndex } from "../../core/src/db/BruteForceIndex";
import { IVFFlatIndex } from "../../core/src/ivf/IVFFlatIndex";
import { StorageEngineV1 } from "../../core/src/storage/StorageEngineV1";
import { MetadataStoreV1 } from "../../core/src/metadata/MetadataStoreV1";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 4294967296) * 2 - 1;
  };
}

function percentile(sorted: number[], p: number) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

function statsFrom(values: number[]) {
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
  };
}

function nowMs() {
  if (typeof performance !== "undefined") return performance.now();
  const [s, ns] = process.hrtime();
  return s * 1000 + ns / 1e6;
}

async function runTrial(cfg: any) {
  const { N, dim, metric, k, nlist, nprobeList, queries, batchSize, seed, scale } = cfg;

  const t0 = nowMs();
  const db = new BruteForceIndex({ dim, metric, preferGPU: false, batchRows: 8192 });
  const cold_start_ms = nowMs() - t0;

  const ids: string[] = new Array(N);
  const vecs = new Float32Array(N * dim);
  const r = rng(seed);
  for (let i = 0; i < N; i++) {
    ids[i] = `id_${i}`;
    const off = i * dim;
    for (let j = 0; j < dim; j++) vecs[off + j] = r() * scale;
  }

  const ingestTimes: number[] = [];
  for (let start = 0; start < N; start += batchSize) {
    const n = Math.min(batchSize, N - start);
    const t1 = nowMs();
    for (let i = 0; i < n; i++) {
      const idx = start + i;
      const v = vecs.subarray(idx * dim, (idx + 1) * dim);
      db.upsert(ids[idx], v);
    }
    const t2 = nowMs();
    ingestTimes.push((t2 - t1) / n);
  }

  const qRand = rng(seed + 999);
  const qs: Float32Array[] = [];
  for (let i = 0; i < queries; i++) {
    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = qRand() * scale;
    qs.push(q);
  }

  for (let i = 0; i < Math.min(5, qs.length); i++) {
    await db.search(qs[i], k);
  }

  const searchTimes: number[] = [];
  const exactResults: Array<number[]> = [];
  for (const q of qs) {
    const t1 = nowMs();
    const res = await db.search(q, k);
    const t2 = nowMs();
    searchTimes.push(t2 - t1);
    exactResults.push(res.map((x: any) => x.internalId));
  }

  const store = (db as any).store as StorageEngineV1;
  const meta = (db as any).meta as MetadataStoreV1;
  const ivf = new IVFFlatIndex(store, meta, null, metric, 1024);
  ivf.trainAndBuild({ nlist, iters: 10, seed: 1234, sampleSize: Math.min(N, 20000) });

  const ivfLatencyTimes: number[] = [];
  const ivfRecallByNprobe: Record<string, number> = {};

  for (const nprobe of nprobeList) {
    let recallSum = 0;
    for (let i = 0; i < qs.length; i++) {
      const q = qs[i];
      const t1 = nowMs();
      const res = await ivf.search(q, { k, nprobe });
      const t2 = nowMs();
      if (nprobe === nprobeList[Math.min(2, nprobeList.length - 1)]) ivfLatencyTimes.push(t2 - t1);

      const approxIds = res.map((x) => x.internalId);
      const exactIds = exactResults[i];
      const set = new Set(exactIds);
      let hit = 0;
      for (const id of approxIds) if (set.has(id)) hit++;
      recallSum += hit / exactIds.length;
    }
    ivfRecallByNprobe[String(nprobe)] = recallSum / qs.length;
  }

  const ingestStats = statsFrom(ingestTimes);
  const searchStats = statsFrom(searchTimes);
  const ivfStats = statsFrom(ivfLatencyTimes);
  const ivf_recall_at_10 = ivfRecallByNprobe[String(nprobeList[nprobeList.length - 1])] ?? 0;
  const ivf_speedup = ivfStats.p95 > 0 ? (searchStats.p95 / ivfStats.p95) : 0;

  const mem = process.memoryUsage();

  return {
    ingest: ingestStats,
    search: searchStats,
    ivf: ivfStats,
    ivf_recall_at_10,
    ivf_recall_by_nprobe: ivfRecallByNprobe,
    ivf_speedup,
    cold_start_ms,
    memory: { rss: mem.rss, heapUsed: mem.heapUsed },
  };
}

function medianOf(values: number[]) {
  const a = values.slice().sort((x, y) => x - y);
  return percentile(a, 0.5);
}

async function main() {
  const env = process.env;
  const cfg = {
    N: Number(env.WISP_BENCH_N ?? 20000),
    dim: Number(env.WISP_BENCH_DIM ?? 64),
    metric: (env.WISP_BENCH_METRIC as any) ?? "dot",
    k: Number(env.WISP_BENCH_K ?? 10),
    nlist: Number(env.WISP_BENCH_NLIST ?? 64),
    nprobeList: (env.WISP_BENCH_NPROBES ? env.WISP_BENCH_NPROBES.split(",").map(Number) : [1, 2, 4, 8, 16]),
    queries: Number(env.WISP_BENCH_Q ?? 20),
    batchSize: Number(env.WISP_BENCH_BATCH ?? 1000),
    seed: Number(env.WISP_BENCH_SEED ?? 1234),
    scale: Number(env.WISP_BENCH_SCALE ?? 0.5),
    trials: Number(env.WISP_BENCH_TRIALS ?? 3),
  };

  const trials: any[] = [];
  for (let i = 0; i < cfg.trials; i++) {
    // shift seed per trial for small variability while deterministic
    trials.push(await runTrial({ ...cfg, seed: cfg.seed + i * 1000 }));
  }

  const ingest_p50 = medianOf(trials.map((t) => t.ingest.p50));
  const ingest_p95 = medianOf(trials.map((t) => t.ingest.p95));
  const search_p50 = medianOf(trials.map((t) => t.search.p50));
  const search_p95 = medianOf(trials.map((t) => t.search.p95));
  const ivf_p50 = medianOf(trials.map((t) => t.ivf.p50));
  const ivf_p95 = medianOf(trials.map((t) => t.ivf.p95));
  const ivf_recall_at_10 = medianOf(trials.map((t) => t.ivf_recall_at_10));
  const ivf_speedup = medianOf(trials.map((t) => t.ivf_speedup));
  const cold_start_ms = medianOf(trials.map((t) => t.cold_start_ms));

  const memory = {
    rss: medianOf(trials.map((t) => t.memory.rss)),
    heapUsed: medianOf(trials.map((t) => t.memory.heapUsed)),
  };

  const result = {
    config: { ...cfg, trials: cfg.trials },
    ingest_ms_p50: ingest_p50,
    ingest_ms_p95: ingest_p95,
    search_ms_p50: search_p50,
    search_ms_p95: search_p95,
    ivf_search_ms_p50: ivf_p50,
    ivf_search_ms_p95: ivf_p95,
    ivf_recall_at_10,
    ivf_speedup,
    ivf_recall_by_nprobe: trials[0]?.ivf_recall_by_nprobe ?? {},
    cold_start_ms,
    warm_start_ms: null,
    warm_first_search_ms: null,
    memory,
  };

  const jsonFlag = process.argv.includes("--json");
  if (jsonFlag) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
