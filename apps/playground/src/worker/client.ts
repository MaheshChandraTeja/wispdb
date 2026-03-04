import type { Msg, Req, Res, Metric } from "./protocol";

export class WispWorkerClient {
  private w: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

  constructor() {
    this.w = new Worker(new URL("./wispdb.worker.ts", import.meta.url), { type: "module" });

    this.w.onmessage = (ev: MessageEvent<Res>) => {
      const res = ev.data;
      const p = this.pending.get(res.id);
      if (!p) return;
      this.pending.delete(res.id);
      if (res.ok) p.resolve(res.v);
      else p.reject(new Error(res.err));
    };
  }

  private call(req: Req, transfer?: Transferable[]) {
    const id = this.nextId++;
    const msg: Msg = { id, req };
    const p = new Promise<any>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.w.postMessage(msg, transfer ?? []);
    return p;
  }

  ping() {
    return this.call({ t: "ping" });
  }

  open(dbName: string, dim: number, metric: Metric, preferGPU = true) {
    return this.call({ t: "open", dbName, dim, metric, preferGPU });
  }

  upsertBatch(ids: string[], vectors: Float32Array, dim: number, metas?: any[]) {
    // Transfer the underlying buffer: zero-copy move (main loses access after postMessage)
    const buf = vectors.buffer.slice(vectors.byteOffset, vectors.byteOffset + vectors.byteLength);
    return this.call({ t: "upsertBatch", ids, vectors: buf, dim, metas }, [buf]);
  }

  generateAndIngest(seed: number, n: number, dim: number, scale = 0.5, idPrefix = "id_") {
    return this.call({ t: "generateAndIngest", seed, n, dim, scale, idPrefix });
  }

  generateAndIngestBatch(seed: number, n: number, dim: number, idOffset: number, scale = 0.5, idPrefix = "id_") {
    return this.call({ t: "generateAndIngest", seed, n, dim, scale, idPrefix, idOffset });
  }

  ivfBuild(nlist: number, iters?: number, seed?: number, sampleSize?: number, batchRows?: number) {
    return this.call({ t: "ivfBuild", nlist, iters, seed, sampleSize, batchRows });
  }

  ivfSearch(query: Float32Array, k: number, nprobe: number) {
    const buf = query.buffer.slice(query.byteOffset, query.byteOffset + query.byteLength);
    return this.call({ t: "ivfSearch", query: buf, k, nprobe }, [buf]);
  }

  search(query: Float32Array, k: number, where?: any, scoreThreshold?: number | null) {
    const buf = query.buffer.slice(query.byteOffset, query.byteOffset + query.byteLength);
    return this.call({ t: "search", query: buf, k, where, scoreThreshold: scoreThreshold ?? null }, [buf]);
  }

  snapshot() {
    return this.call({ t: "snapshot" });
  }
}
