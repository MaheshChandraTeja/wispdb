type BenchOpts = {
  N?: number;
  dim?: number;
  k?: number;
  nlist?: number;
  nprobeList?: number[];
  queries?: number;
  batchSize?: number;
  seed?: number;
  metric?: "dot" | "cosine" | "l2";
  preferGPU?: boolean;
  scale?: number;
  frameSearches?: number;
};

type BenchResult = {
  config: Record<string, any>;
  ingest_ms_p50: number;
  ingest_ms_p95: number;
  search_ms_p50: number;
  search_ms_p95: number;
  ivf_search_ms_p50: number;
  ivf_search_ms_p95: number;
  ivf_recall_at_10: number;
  ivf_recall_by_nprobe: Record<string, number>;
  ivf_speedup: number;
  cold_start_ms: number;
  warm_start_ms: number | null;
  warm_first_search_ms: number | null;
  frame_p95_ms: number;
  memory: Record<string, number>;
  stats?: any;
};

type WorkerCall = (req: any, transfer?: Transferable[]) => Promise<any>;

type WorkerClient = {
  call: WorkerCall;
  terminate: () => void;
};

function makeWorkerClient(): WorkerClient {
  const worker = new Worker(new URL("../worker/wispdb.worker.ts", import.meta.url), { type: "module" });
  let nextId = 1;
  const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

  worker.onmessage = (ev) => {
    const res = ev.data as { id: number; ok: boolean; v?: any; err?: string };
    const p = pending.get(res.id);
    if (!p) return;
    pending.delete(res.id);
    if (res.ok) p.resolve(res.v);
    else p.reject(new Error(res.err));
  };

  const call = (req: any, transfer?: Transferable[]) => {
    const id = nextId++;
    const msg = { id, req };
    return new Promise<any>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker.postMessage(msg, transfer ?? []);
    });
  };

  return {
    call,
    terminate: () => worker.terminate(),
  };
}

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

async function measureFramesWhile(scrollBox: HTMLElement, promise: Promise<unknown>) {
  return new Promise<{ p95: number }>((resolve) => {
    let running = true;
    promise.finally(() => { running = false; });

    const deltas: number[] = [];
    let last = performance.now();

    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      if (running) deltas.push(dt);

      const maxScroll = scrollBox.scrollHeight - scrollBox.clientHeight;
      if (maxScroll > 0) {
        let next = scrollBox.scrollTop + 4;
        if (next > maxScroll) next = 0;
        scrollBox.scrollTop = next;
      }

      if (running) requestAnimationFrame(tick);
      else {
        const sorted = deltas.slice().sort((a, b) => a - b);
        resolve({ p95: percentile(sorted, 0.95) });
      }
    };

    requestAnimationFrame(tick);
  });
}

async function runBench(opts: BenchOpts = {}): Promise<BenchResult> {
  const N = opts.N ?? 10000;
  const dim = opts.dim ?? 64;
  const k = opts.k ?? 10;
  const metric = opts.metric ?? "dot";
  const nlist = opts.nlist ?? 64;
  const rawProbes = opts.nprobeList ?? [1, 2, 4, 8, 16];
  const nprobeList = rawProbes.filter((n) => n > 0 && n <= nlist);
  if (nprobeList.length === 0) nprobeList.push(Math.max(1, Math.min(1, nlist)));
  const queries = opts.queries ?? 20;
  const batchSize = opts.batchSize ?? 1000;
  const seed = opts.seed ?? 1234;
  const scale = opts.scale ?? 0.5;
  const preferGPU = opts.preferGPU ?? false;
  const frameSearches = opts.frameSearches ?? 200;

  const config = { N, dim, k, metric, nlist, nprobeList, queries, batchSize, seed, scale, preferGPU };

  const client = makeWorkerClient();
  const ping = await client.call({ t: "ping" });

  const coldStartT0 = performance.now();
  const dbName = `bench_${Date.now()}`;
  await client.call({ t: "open", dbName, dim, metric, preferGPU });
  const cold_start_ms = performance.now() - coldStartT0;

  const ingestTimes: number[] = [];
  for (let start = 0; start < N; start += batchSize) {
    const n = Math.min(batchSize, N - start);
    const t0 = performance.now();
    await client.call({
      t: "generateAndIngest",
      seed: seed + start,
      n,
      dim,
      scale,
      idPrefix: "id_",
      idOffset: start,
    });
    const t1 = performance.now();
    ingestTimes.push((t1 - t0) / n);
  }

  const ingestStats = statsFrom(ingestTimes);

  const qRand = rng(seed + 999);
  const qs: Float32Array[] = [];
  for (let i = 0; i < queries; i++) {
    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = qRand() * scale;
    qs.push(q);
  }

  // warmup
  for (let i = 0; i < Math.min(5, qs.length); i++) {
    const buf = qs[i].buffer.slice(0);
    await client.call({ t: "search", query: buf, k }, [buf]);
  }

  const searchTimes: number[] = [];
  const exactResults: Array<number[]> = [];
  for (const q of qs) {
    const buf = q.buffer.slice(0);
    const res = await client.call({ t: "search", query: buf, k }, [buf]);
    searchTimes.push(res.tookMs);
    exactResults.push(res.hits.map((h: any) => h.internalId));
  }

  const searchStats = statsFrom(searchTimes);

  await client.call({ t: "ivfBuild", nlist, iters: 10, seed: 1234, sampleSize: Math.min(N, 20000) });

  const ivfLatencyTimes: number[] = [];
  const ivfRecallByNprobe: Record<string, number> = {};

  for (const nprobe of nprobeList) {
    let recallSum = 0;
    for (let i = 0; i < qs.length; i++) {
      const q = qs[i];
      const buf = q.buffer.slice(0);
      const res = await client.call({ t: "ivfSearch", query: buf, k, nprobe }, [buf]);
      if (nprobe === nprobeList[Math.min(2, nprobeList.length - 1)]) {
        ivfLatencyTimes.push(res.tookMs);
      }
      const approxIds = res.hits.map((h: any) => h.internalId);
      const exactIds = exactResults[i];
      const set = new Set(exactIds);
      let hit = 0;
      for (const id of approxIds) if (set.has(id)) hit++;
      recallSum += hit / exactIds.length;
    }
    ivfRecallByNprobe[String(nprobe)] = recallSum / qs.length;
  }

  const ivfLatencyStats = statsFrom(ivfLatencyTimes);
  const ivf_recall_at_10 = ivfRecallByNprobe[String(nprobeList[nprobeList.length - 1])] ?? 0;
  const ivf_speedup = ivfLatencyStats.p95 > 0 ? (searchStats.p95 / ivfLatencyStats.p95) : 0;

  await client.call({ t: "snapshot" });
  client.terminate();

  // warm start
  const warmClient = makeWorkerClient();
  const warmT0 = performance.now();
  await warmClient.call({ t: "open", dbName, dim, metric, preferGPU });
  const warm_start_ms = performance.now() - warmT0;
  const warmQuery = qs[0];
  const warmBuf = warmQuery.buffer.slice(0);
  const warmRes = await warmClient.call({ t: "search", query: warmBuf, k }, [warmBuf]);
  const warm_first_search_ms = warmRes.tookMs as number;

  // frame time test
  let scrollBox = document.querySelector<HTMLElement>("#scrollbox");
  if (!scrollBox) {
    scrollBox = document.createElement("div");
    scrollBox.style.height = "240px";
    scrollBox.style.overflow = "auto";
    scrollBox.style.border = "1px solid #ccc";
    document.body.appendChild(scrollBox);
  }
  if (scrollBox.childElementCount === 0) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 5000; i++) {
      const row = document.createElement("div");
      row.textContent = `row ${i}`;
      row.style.padding = "2px 6px";
      frag.appendChild(row);
    }
    scrollBox.appendChild(frag);
  }

  const searchLoop = (async () => {
    for (let i = 0; i < frameSearches; i++) {
      const qb = qs[i % qs.length];
      const buf = qb.buffer.slice(0);
      await warmClient.call({ t: "search", query: buf, k }, [buf]);
      if (i % 10 === 0) await new Promise((r) => setTimeout(r, 0));
    }
  })();

  const frameStats = await measureFramesWhile(scrollBox, searchLoop);

  const mem: Record<string, number> = {};
  const perfMem = (performance as any).memory;
  if (perfMem?.usedJSHeapSize) mem.usedJSHeapSize = perfMem.usedJSHeapSize;
  if (perfMem?.totalJSHeapSize) mem.totalJSHeapSize = perfMem.totalJSHeapSize;
  if (perfMem?.jsHeapSizeLimit) mem.jsHeapSizeLimit = perfMem.jsHeapSizeLimit;

  const stats = await warmClient.call({ t: "stats" });
  warmClient.terminate();

  return {
    config: { ...config, ping },
    ingest_ms_p50: ingestStats.p50,
    ingest_ms_p95: ingestStats.p95,
    search_ms_p50: searchStats.p50,
    search_ms_p95: searchStats.p95,
    ivf_search_ms_p50: ivfLatencyStats.p50,
    ivf_search_ms_p95: ivfLatencyStats.p95,
    ivf_recall_at_10,
    ivf_recall_by_nprobe: ivfRecallByNprobe,
    ivf_speedup,
    cold_start_ms,
    warm_start_ms,
    warm_first_search_ms,
    frame_p95_ms: frameStats.p95,
    memory: mem,
    stats: stats?.stats,
  };
}

declare global {
  interface Window {
    __WISP_BENCH_RUN__?: (opts?: BenchOpts) => Promise<BenchResult>;
  }
}

window.__WISP_BENCH_RUN__ = runBench;
