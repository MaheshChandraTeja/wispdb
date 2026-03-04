import { StorageEngineV1 } from "../storage/StorageEngineV1";
import { DeviceManager, ChunkScannerGPU } from "@wispdb/gpu";
import { MetadataStoreV1 } from "../metadata/MetadataStoreV1";
import type { MetadataSchema, WhereClause } from "../metadata/types";


export type Metric = "dot" | "cosine" | "l2";

export interface SearchHit {
  id: string;
  score: number;
  internalId: number;
}

export interface BruteForceOptions {
  dim: number;
  metric?: Metric;
  batchRows?: number;
  kMax?: number;               // GPU TopK limit (<=256)
  scoreThreshold?: number;     // filter results after topk
  preferGPU?: boolean;         // fallback to CPU if no WebGPU
  labelPrefix?: string;
  metadataSchema?: MetadataSchema;
}

function isWorse(aId: number, aScore: number, bId: number, bScore: number): boolean {
  // worse = smaller score; tie -> bigger id is worse (stable)
  if (aScore < bScore) return true;
  if (aScore > bScore) return false;
  return aId > bId;
}

export class BruteForceIndex {
  private store: StorageEngineV1;
  private meta: MetadataStoreV1;
  private metric: Metric;
  private batchRows: number;
  private kMax: number;
  private scoreThreshold: number | null;
  private cosinePreNormalized: boolean;

  private dm: DeviceManager | null = null;
  private gpu: ChunkScannerGPU | null = null;
  private preferGPU: boolean;
  private labelPrefix: string;

  constructor(opts: BruteForceOptions) {
    this.metric = opts.metric ?? "dot";
    this.batchRows = opts.batchRows ?? 16384;
    this.kMax = Math.min(opts.kMax ?? 256, 256);
    this.scoreThreshold = opts.scoreThreshold ?? null;
    this.preferGPU = opts.preferGPU ?? true;
    this.labelPrefix = opts.labelPrefix ?? "wispdb";
    this.cosinePreNormalized = this.metric === "cosine";

    this.store = new StorageEngineV1(opts.dim);
    this.meta = new MetadataStoreV1(opts.metadataSchema ?? {}, 4096);
  }

  async open() {
    // GPU optional, but preferred
    if (this.preferGPU && typeof navigator !== "undefined" && (navigator as any).gpu) {
      this.dm = new DeviceManager({ labelPrefix: this.labelPrefix });
      this.gpu = new ChunkScannerGPU(this.dm, this.labelPrefix);
      await this.gpu.open();
    }
  }

  upsert(id: string, vec: Float32Array, metadata?: any) {
    const v = this.cosinePreNormalized ? this.normalizeForCosine(vec) : vec;
    const internal = this.store.upsert(id, v);
    if (metadata !== undefined) this.meta.upsert(internal, metadata);
    else if (this.meta.get(internal) === null) this.meta.upsert(internal, {});
    return internal;
  }

  delete(id: string) {
    const internal = this.store.getInternalId(id);
    const ok = this.store.delete(id);
    if (ok && internal != null) this.meta.delete(internal);
    return ok;
  }

  async search(
    query: Float32Array,
    arg1: number | { k: number; where?: WhereClause; scoreThreshold?: number },
  ): Promise<SearchHit[]> {
    const opts = typeof arg1 === "number" ? { k: arg1 } : arg1;
    const kk = Math.min(opts.k, this.kMax);
    const threshold = opts.scoreThreshold ?? this.scoreThreshold ?? null;
    const where = opts.where;

    if (kk <= 0) return [];

    if (this.gpu) {
      return this.searchGPUFiltered(query, kk, threshold, where);
    }
    return this.searchCPUFiltered(query, kk, threshold, where);
  }

  // -------- GPU search (chunked scan) --------

  private async searchGPU(query: Float32Array, k: number, threshold: number | null): Promise<SearchHit[]> {
    const dim = this.store.dim();
    if (query.length !== dim) throw new Error(`Query dim mismatch: expected ${dim}, got ${query.length}`);

    const prepared = await this.gpu!.prepareQuery(query, dim, this.metric);

    // min-heap of size k over (score, internalId)
    const heapScore = new Float32Array(k);
    const heapId = new Int32Array(k);
    let size = 0;

    const siftUp = (i: number) => {
      while (i > 0) {
        const p = (i - 1) >> 1;
        const ci = heapId[i], cs = heapScore[i];
        const pi = heapId[p], ps = heapScore[p];
        if (isWorse(ci, cs, pi, ps)) {
          heapId[i] = pi; heapScore[i] = ps;
          heapId[p] = ci; heapScore[p] = cs;
          i = p;
        } else break;
      }
    };

    const siftDown = (i: number) => {
      while (true) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < size && isWorse(heapId[l], heapScore[l], heapId[m], heapScore[m])) m = l;
        if (r < size && isWorse(heapId[r], heapScore[r], heapId[m], heapScore[m])) m = r;
        if (m === i) break;
        const ti = heapId[i], ts = heapScore[i];
        heapId[i] = heapId[m]; heapScore[i] = heapScore[m];
        heapId[m] = ti; heapScore[m] = ts;
        i = m;
      }
    };

    for (const batch of this.store.iterateLiveBatches(this.batchRows)) {
      const pairs = await this.gpu!.topKBatch(
        batch.vectors,
        dim,
        prepared.queryUsed,
        this.metric,
        k,
        this.cosinePreNormalized,
      );

      for (const p of pairs) {
        if (p.idx < 0 || p.idx >= batch.ids.length) continue; // HARD GUARD
        const internalId = batch.ids[p.idx];
        const score = p.score;

        if (!Number.isFinite(score)) continue;
        if (threshold != null && score < threshold) continue;

        if (size < k) {
          heapId[size] = internalId;
          heapScore[size] = score;
          siftUp(size);
          size++;
        } else {
          const wi = heapId[0], ws = heapScore[0];
          const better = score > ws || (score === ws && internalId < wi);
          if (better) {
            heapId[0] = internalId;
            heapScore[0] = score;
            siftDown(0);
          }
        }
      }
    }

    prepared.release();

    // Heap -> sorted results
    const tmp: Array<{ internalId: number; score: number }> = [];
    for (let i = 0; i < size; i++) tmp.push({ internalId: heapId[i], score: heapScore[i] });
    tmp.sort((a, b) => (b.score - a.score) || (a.internalId - b.internalId));

    const out: SearchHit[] = [];
    for (const r of tmp) {
      const ext = this.store.getExternalIdByInternal(r.internalId);
      if (!ext) continue; // should not happen unless raced deletes
      out.push({ id: ext, score: r.score, internalId: r.internalId });
    }
    return out;
  }

  private async searchGPUFiltered(
    query: Float32Array,
    k: number,
    threshold: number | null,
    where?: WhereClause,
  ): Promise<SearchHit[]> {
    const dim = this.store.dim();
    if (query.length !== dim) throw new Error(`Query dim mismatch: expected ${dim}, got ${query.length}`);

    const prepared = await this.gpu!.prepareQuery(query, dim, this.metric);

    const maxId = this.store.maxInternalIdExclusive();
    const candidates = where
      ? this.meta.prefilter(where, maxId, (id) => this.store.hasInternalId(id))
      : null;

    // heap (same as your fixed one)
    const heapScore = new Float32Array(k);
    const heapId = new Int32Array(k);
    let size = 0;

    const isWorse = (aId: number, aScore: number, bId: number, bScore: number): boolean => {
      if (aScore < bScore) return true;
      if (aScore > bScore) return false;
      return aId > bId;
    };

    const siftUp = (i: number) => {
      while (i > 0) {
        const p = (i - 1) >> 1;
        const ci = heapId[i], cs = heapScore[i];
        const pi = heapId[p], ps = heapScore[p];
        if (isWorse(ci, cs, pi, ps)) {
          heapId[i] = pi; heapScore[i] = ps;
          heapId[p] = ci; heapScore[p] = cs;
          i = p;
        } else break;
      }
    };

    const siftDown = (i: number) => {
      while (true) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < size && isWorse(heapId[l], heapScore[l], heapId[m], heapScore[m])) m = l;
        if (r < size && isWorse(heapId[r], heapScore[r], heapId[m], heapScore[m])) m = r;
        if (m === i) break;
        const ti = heapId[i], ts = heapScore[i];
        heapId[i] = heapId[m]; heapScore[i] = heapScore[m];
        heapId[m] = ti; heapScore[m] = ts;
        i = m;
      }
    };

    const iter = candidates
      ? this.iterateCandidateBatches(candidates, dim, this.batchRows)
      : this.store.iterateLiveBatches(this.batchRows);

    for (const batch of iter as any) {
      const pairs = await this.gpu!.topKBatch(
        batch.vectors,
        dim,
        prepared.queryUsed,
        this.metric,
        k,
        this.cosinePreNormalized,
      );

      for (const p of pairs) {
        if (p.idx < 0 || p.idx >= batch.ids.length) continue;
        const internalId = batch.ids[p.idx];
        const score = p.score;

        if (!Number.isFinite(score)) continue;
        if (threshold != null && score < threshold) continue;

        if (size < k) {
          heapId[size] = internalId;
          heapScore[size] = score;
          siftUp(size);
          size++;
        } else {
          const wi = heapId[0], ws = heapScore[0];
          const better = score > ws || (score === ws && internalId < wi);
          if (better) {
            heapId[0] = internalId;
            heapScore[0] = score;
            siftDown(0);
          }
        }
      }
    }

    prepared.release();

    const tmp: Array<{ internalId: number; score: number }> = [];
    for (let i = 0; i < size; i++) tmp.push({ internalId: heapId[i], score: heapScore[i] });
    tmp.sort((a, b) => (b.score - a.score) || (a.internalId - b.internalId));

    const out: any[] = [];
    const seen = new Set<number>();
    for (const r of tmp) {
      if (seen.has(r.internalId)) continue;
      seen.add(r.internalId);
      const ext = this.store.getExternalIdByInternal(r.internalId);
      if (!ext) continue;
      out.push({ id: ext, score: r.score, internalId: r.internalId });
    }
    return out;
  }

  // -------- CPU fallback (exact) --------

  private searchCPU(query: Float32Array, k: number, threshold: number | null): SearchHit[] {
    const dim = this.store.dim();
    if (query.length !== dim) throw new Error(`Query dim mismatch: expected ${dim}, got ${query.length}`);

    // normalize query for cosine
    let q = query;
    if (this.metric === "cosine") {
      let ss = 0;
      for (let j = 0; j < dim; j++) ss += query[j] * query[j];
      const inv = 1 / Math.sqrt(ss || 1e-12);
      const qn = new Float32Array(dim);
      for (let j = 0; j < dim; j++) qn[j] = query[j] * inv;
      q = qn;
    }

    const heapScore = new Float32Array(k);
    const heapId = new Int32Array(k);
    let size = 0;

    const siftUp = (i: number) => {
      while (i > 0) {
        const p = (i - 1) >> 1;
        const ci = heapId[i], cs = heapScore[i];
        const pi = heapId[p], ps = heapScore[p];
        if (isWorse(ci, cs, pi, ps)) {
          heapId[i] = pi; heapScore[i] = ps;
          heapId[p] = ci; heapScore[p] = cs;
          i = p;
        } else break;
      }
    };
    const siftDown = (i: number) => {
      while (true) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < size && isWorse(heapId[l], heapScore[l], heapId[m], heapScore[m])) m = l;
        if (r < size && isWorse(heapId[r], heapScore[r], heapId[m], heapScore[m])) m = r;
        if (m === i) break;
        const ti = heapId[i], ts = heapScore[i];
        heapId[i] = heapId[m]; heapScore[i] = heapScore[m];
        heapId[m] = ti; heapScore[m] = ts;
        i = m;
      }
    };

    for (const batch of this.store.iterateLiveBatches(this.batchRows)) {
      const n = batch.ids.length;

      for (let i = 0; i < n; i++) {
        const internalId = batch.ids[i];
        const off = i * dim;

        let score = 0;

        if (this.metric === "dot") {
          for (let j = 0; j < dim; j++) score += batch.vectors[off + j] * q[j];
        } else if (this.metric === "l2") {
          let s = 0;
          for (let j = 0; j < dim; j++) {
            const d = batch.vectors[off + j] - q[j];
            s += d * d;
          }
          score = -s;
        } else {
          if (this.cosinePreNormalized) {
            let s = 0;
            for (let j = 0; j < dim; j++) s += batch.vectors[off + j] * q[j];
            score = s;
          } else {
            // cosine: normalize vector then dot
            let ss = 0;
            for (let j = 0; j < dim; j++) {
              const x = batch.vectors[off + j];
              ss += x * x;
            }
            const inv = 1 / Math.sqrt(ss || 1e-12);
            let s = 0;
            for (let j = 0; j < dim; j++) s += (batch.vectors[off + j] * inv) * q[j];
            score = s;
          }
        }

        if (threshold != null && score < threshold) continue;

        if (size < k) {
          heapId[size] = internalId;
          heapScore[size] = score;
          siftUp(size);
          size++;
        } else {
          const wi = heapId[0], ws = heapScore[0];
          const better = score > ws || (score === ws && internalId < wi);
          if (better) {
            heapId[0] = internalId;
            heapScore[0] = score;
            siftDown(0);
          }
        }
      }
    }

    const tmp: Array<{ internalId: number; score: number }> = [];
    for (let i = 0; i < size; i++) tmp.push({ internalId: heapId[i], score: heapScore[i] });
    tmp.sort((a, b) => (b.score - a.score) || (a.internalId - b.internalId));

    const out: SearchHit[] = [];
    for (const r of tmp) {
      const ext = this.store.getExternalIdByInternal(r.internalId);
      if (!ext) continue;
      out.push({ id: ext, score: r.score, internalId: r.internalId });
    }
    return out;
  }

  private searchCPUFiltered(
    query: Float32Array,
    k: number,
    threshold: number | null,
    where?: WhereClause,
  ): SearchHit[] {
    const dim = this.store.dim();
    if (query.length !== dim) throw new Error(`Query dim mismatch: expected ${dim}, got ${query.length}`);

    let q = query;
    if (this.metric === "cosine") {
      let ss = 0;
      for (let j = 0; j < dim; j++) ss += query[j] * query[j];
      const inv = 1 / Math.sqrt(ss || 1e-12);
      const qn = new Float32Array(dim);
      for (let j = 0; j < dim; j++) qn[j] = query[j] * inv;
      q = qn;
    }

    const heapScore = new Float32Array(k);
    const heapId = new Int32Array(k);
    let size = 0;

    const siftUp = (i: number) => {
      while (i > 0) {
        const p = (i - 1) >> 1;
        const ci = heapId[i], cs = heapScore[i];
        const pi = heapId[p], ps = heapScore[p];
        if (isWorse(ci, cs, pi, ps)) {
          heapId[i] = pi; heapScore[i] = ps;
          heapId[p] = ci; heapScore[p] = cs;
          i = p;
        } else break;
      }
    };
    const siftDown = (i: number) => {
      while (true) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < size && isWorse(heapId[l], heapScore[l], heapId[m], heapScore[m])) m = l;
        if (r < size && isWorse(heapId[r], heapScore[r], heapId[m], heapScore[m])) m = r;
        if (m === i) break;
        const ti = heapId[i], ts = heapScore[i];
        heapId[i] = heapId[m]; heapScore[i] = heapScore[m];
        heapId[m] = ti; heapScore[m] = ts;
        i = m;
      }
    };

    const candidates = where
      ? this.meta.prefilter(
          where,
          this.store.maxInternalIdExclusive(),
          (id) => this.store.hasInternalId(id),
        )
      : null;

    if (candidates && candidates.length === 0) return [];

    const iter = candidates
      ? this.iterateCandidateBatches(candidates, dim, this.batchRows)
      : this.store.iterateLiveBatches(this.batchRows);

    for (const batch of iter as any) {
      const n = batch.ids.length;

      for (let i = 0; i < n; i++) {
        const internalId = batch.ids[i];
        const off = i * dim;

        let score = 0;
        if (this.metric === "dot") {
          for (let j = 0; j < dim; j++) score += batch.vectors[off + j] * q[j];
        } else if (this.metric === "l2") {
          let s = 0;
          for (let j = 0; j < dim; j++) {
            const d = batch.vectors[off + j] - q[j];
            s += d * d;
          }
          score = -s;
        } else {
          if (this.cosinePreNormalized) {
            let s = 0;
            for (let j = 0; j < dim; j++) s += batch.vectors[off + j] * q[j];
            score = s;
          } else {
            let ss = 0;
            for (let j = 0; j < dim; j++) {
              const x = batch.vectors[off + j];
              ss += x * x;
            }
            const inv = 1 / Math.sqrt(ss || 1e-12);
            let s = 0;
            for (let j = 0; j < dim; j++) s += (batch.vectors[off + j] * inv) * q[j];
            score = s;
          }
        }

        if (threshold != null && score < threshold) continue;

        if (size < k) {
          heapId[size] = internalId;
          heapScore[size] = score;
          siftUp(size);
          size++;
        } else {
          const wi = heapId[0], ws = heapScore[0];
          const better = score > ws || (score === ws && internalId < wi);
          if (better) {
            heapId[0] = internalId;
            heapScore[0] = score;
            siftDown(0);
          }
        }
      }
    }

    const tmp: Array<{ internalId: number; score: number }> = [];
    for (let i = 0; i < size; i++) tmp.push({ internalId: heapId[i], score: heapScore[i] });
    tmp.sort((a, b) => (b.score - a.score) || (a.internalId - b.internalId));

    const out: SearchHit[] = [];
    for (const r of tmp) {
      const ext = this.store.getExternalIdByInternal(r.internalId);
      if (!ext) continue;
      out.push({ id: ext, score: r.score, internalId: r.internalId });
    }
    return out;
  }

  private normalizeForCosine(vec: Float32Array): Float32Array {
    const dim = this.store.dim();
    if (vec.length !== dim) {
      throw new Error(`Vector dim mismatch: expected ${dim}, got ${vec.length}`);
    }
    let ss = 0;
    for (let j = 0; j < dim; j++) ss += vec[j] * vec[j];
    const inv = 1 / Math.sqrt(ss || 1e-12);
    const out = new Float32Array(dim);
    for (let j = 0; j < dim; j++) out[j] = vec[j] * inv;
    return out;
  }

  compactInMemory() {
    const res = this.store.compactTrailingDead();
    this.meta.compactToMaxId(res.newNextId);
    return res.newNextId;
  }

  private *iterateCandidateBatches(candidateIds: Int32Array, dim: number, batchRows: number) {
    const idsTmp = new Int32Array(Math.min(batchRows, candidateIds.length));
    const vecTmp = new Float32Array(idsTmp.length * dim);

    let count = 0;

    for (let i = 0; i < candidateIds.length; i++) {
      const id = candidateIds[i];
      const view = this.store.getVectorViewByInternalId(id);
      if (!view) continue;

      idsTmp[count] = id;
      vecTmp.set(view, count * dim);
      count++;

      if (count === idsTmp.length) {
        yield { ids: idsTmp.slice(0, count), vectors: vecTmp.slice(0, count * dim) };
        count = 0;
      }
    }

    if (count > 0) {
      yield { ids: idsTmp.slice(0, count), vectors: vecTmp.slice(0, count * dim) };
    }
  }
}
