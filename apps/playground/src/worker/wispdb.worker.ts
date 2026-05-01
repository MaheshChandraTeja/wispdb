/// <reference lib="webworker" />

import type { Msg, Res } from "./protocol";
import type { Metric } from "./protocol";

// Use your real exports:
import { PersistentBruteForceDB, IVFFlatIndex } from "wispdb";

let db: PersistentBruteForceDB | null = null;
let dim = 0;
let metric: Metric = "dot";
let ivf: IVFFlatIndex | null = null;

function replyOk(id: number, v: any) {
  const m: Res = { t: "res", id, ok: true, v };
  (self as any).postMessage(m);
}
function replyErr(id: number, err: any) {
  const m: Res = { t: "res", id, ok: false, err: String(err?.message ?? err) };
  (self as any).postMessage(m);
}

async function handle(msg: Msg) {
  const { id, req } = msg;

  try {
    if (req.t === "ping") {
      return replyOk(id, {
        worker: true,
        crossOriginIsolated: (self as any).crossOriginIsolated === true,
        userAgent: (self as any).navigator?.userAgent ?? "unknown",
      });
    }

    if (req.t === "open") {
      dim = req.dim;
      metric = req.metric;
      ivf = null;

      db = new PersistentBruteForceDB(req.dbName, {
        dim,
        metric,
        preferGPU: req.preferGPU ?? true,
        // keep your existing defaults
        batchRows: 8192,
        kMax: 256,
      });

      await db.open();

      return replyOk(id, { opened: true, dim, metric });
    }

    if (!db) throw new Error("DB not opened. Call open() first.");

    if (req.t === "upsertBatch") {
      if (req.dim !== dim) throw new Error(`dim mismatch: got ${req.dim}, expected ${dim}`);
      const vecs = new Float32Array(req.vectors);
      const n = req.ids.length;
      if (vecs.length !== n * dim) throw new Error(`vectors length mismatch: got ${vecs.length}, expected ${n * dim}`);

      for (let i = 0; i < n; i++) {
        const v = vecs.subarray(i * dim, (i + 1) * dim);
        const meta = req.metas ? req.metas[i] : undefined;
        // journal append happens inside wrapper upsert()
        await db.upsert(req.ids[i], v, meta);
      }

      return replyOk(id, { upserted: n });
    }

    if (req.t === "generateAndIngest") {
      if (req.dim !== dim) throw new Error(`dim mismatch: got ${req.dim}, expected ${dim}`);
      const n = req.n | 0;
      const scale = req.scale ?? 0.5;
      const prefix = req.idPrefix ?? "id_";
      const offset = req.idOffset ?? 0;

      let seed = req.seed >>> 0;
      const rnd = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return (seed / 4294967296) * 2 - 1;
      };

      for (let i = 0; i < n; i++) {
        const v = new Float32Array(dim);
        for (let j = 0; j < dim; j++) v[j] = rnd() * scale;
        await db.upsert(`${prefix}${offset + i}`, v);
        if (i % 1000 === 0 && i) await new Promise((r) => setTimeout(r, 0));
      }

      return replyOk(id, { upserted: n });
    }

    if (req.t === "ivfBuild") {
      if (!db) throw new Error("DB not opened. Call open() first.");
      // @ts-ignore
      const idx = (db as any).idx;
      const store = idx.store;
      const meta = idx.meta;
      const gpu = idx.gpu ?? null;

      ivf = new IVFFlatIndex(store, meta, gpu, metric, req.batchRows ?? 1024);
      ivf.trainAndBuild({
        nlist: req.nlist,
        iters: req.iters,
        seed: req.seed,
        sampleSize: req.sampleSize,
      });

      const sizes = ivf.getListSizes();
      return replyOk(id, { nlist: req.nlist, listSizes: Array.from(sizes) });
    }

    if (req.t === "ivfSearch") {
      if (!ivf) throw new Error("IVF not built. Call ivfBuild() first.");
      const q = new Float32Array(req.query);
      if (q.length !== dim) throw new Error(`query dim mismatch: got ${q.length}, expected ${dim}`);

      const t0 = performance.now();
      const hits = await ivf.search(q, { k: req.k, nprobe: req.nprobe });
      const t1 = performance.now();
      return replyOk(id, { hits, tookMs: t1 - t0 });
    }

    if (req.t === "delete") {
      const ok = await db.delete(req.id);
      return replyOk(id, { ok });
    }

    if (req.t === "search") {
      const q = new Float32Array(req.query);
      if (q.length !== dim) throw new Error(`query dim mismatch: got ${q.length}, expected ${dim}`);

      const t0 = performance.now();
      const hits = await db.search(q, {
        k: req.k,
        where: req.where,
        scoreThreshold: req.scoreThreshold ?? null,
      });
      const t1 = performance.now();

      return replyOk(id, { hits, tookMs: t1 - t0 });
    }

    if (req.t === "snapshot") {
      const snapId = await db.snapshotNow();
      return replyOk(id, { snapshotId: snapId });
    }

    if (req.t === "stats") {
      // @ts-ignore
      const s = db["idx"]?.["store"]?.stats?.() ?? null;
      return replyOk(id, { stats: s });
    }

    throw new Error(`Unknown request: ${(req as any).t}`);
  } catch (e) {
    replyErr(id, e);
  }
}

self.onmessage = (ev: MessageEvent<Msg>) => {
  // Serialize by awaiting handle, keeps state safe.
  // (If you want true concurrency later: add an internal queue.)
  handle(ev.data);
};
