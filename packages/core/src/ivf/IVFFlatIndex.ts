import type { WhereClause } from "../metadata/types";
import type { MetadataStoreV1 } from "../metadata/MetadataStoreV1";
import type { StorageEngineV1 } from "../storage/StorageEngineV1";
import type { Metric } from "../db/BruteForceIndex";
import type { ChunkScannerGPU } from "@wispdb/gpu";
import { trainKMeans } from "./kmeans";
import { Int32Bag } from "./Int32Bag";

function dot(a: Float32Array, aOff: number, b: Float32Array, bOff: number, dim: number) {
  let s = 0;
  for (let j = 0; j < dim; j++) s += a[aOff + j] * b[bOff + j];
  return s;
}
function l2sq(a: Float32Array, aOff: number, b: Float32Array, bOff: number, dim: number) {
  let s = 0;
  for (let j = 0; j < dim; j++) {
    const d = a[aOff + j] - b[bOff + j];
    s += d * d;
  }
  return s;
}

export interface IVFTrainOptions {
  nlist: number;
  iters?: number;
  seed?: number;
  sampleSize?: number;
}

export interface IVFSearchOptions {
  k: number;
  nprobe: number;
  where?: WhereClause;
  scoreThreshold?: number | null;
  maxCandidates?: number; // safety cap
}

export interface SearchHit {
  id: string;
  score: number;
  internalId: number;
}

export class IVFFlatIndex {
  private centroids: Float32Array | null = null;
  private nlist = 0;

  private lists: Int32Bag[] = [];
  private assignedList: Int32Array = new Int32Array(0);
  private posInList: Int32Array = new Int32Array(0);

  constructor(
    private store: StorageEngineV1,
    private meta: MetadataStoreV1,
    private gpu: ChunkScannerGPU | null,
    private metric: Metric,
    private batchRows: number,
    private kMax = 256,
  ) {}

  // ---- training & build ----

  trainAndBuild(opts: IVFTrainOptions) {
    const dim = this.store.dim();
    const maxId = this.store.maxInternalIdExclusive();

    // sample live vectors deterministically (ascending internalId)
    const sampleSize = Math.min(opts.sampleSize ?? 20000, maxId);
    const sample = new Float32Array(sampleSize * dim);

    let count = 0;
    for (let id = 0; id < maxId && count < sampleSize; id++) {
      if (!this.store.hasInternalId(id)) continue;
      const v = this.store.getVectorViewByInternalId(id);
      if (!v) continue;
      sample.set(v, count * dim);
      count++;
    }
    if (count === 0) throw new Error("No vectors to train on.");
    const n = count;

    this.nlist = Math.min(opts.nlist, n);
    this.centroids = trainKMeans(sample, n, dim, {
      nlist: this.nlist,
      iters: opts.iters ?? 20,
      seed: opts.seed ?? 1234,
    });

    this.rebuildLists();
  }

  rebuildLists() {
    if (!this.centroids) throw new Error("No centroids. Train first.");
    const dim = this.store.dim();
    const maxId = this.store.maxInternalIdExclusive();

    this.lists = Array.from({ length: this.nlist }, () => new Int32Bag());
    this.assignedList = new Int32Array(maxId);
    this.assignedList.fill(-1);
    this.posInList = new Int32Array(maxId);
    this.posInList.fill(-1);

    for (let id = 0; id < maxId; id++) {
      if (!this.store.hasInternalId(id)) continue;
      const v = this.store.getVectorViewByInternalId(id);
      if (!v) continue;

      const lid = this.nearestCentroid(v, dim);
      const pos = this.lists[lid].push(id);
      this.assignedList[id] = lid;
      this.posInList[id] = pos;
    }
  }

  private ensureAssignCapacity(maxIdExclusive: number) {
    if (this.assignedList.length >= maxIdExclusive) return;
    const grown = new Int32Array(maxIdExclusive);
    grown.fill(-1);
    grown.set(this.assignedList, 0);
    this.assignedList = grown;

    const pos = new Int32Array(maxIdExclusive);
    pos.fill(-1);
    pos.set(this.posInList, 0);
    this.posInList = pos;
  }

  private removeFromList(lid: number, internalId: number) {
    if (lid < 0 || lid >= this.lists.length) return;
    const list = this.lists[lid];
    let pos = internalId < this.posInList.length ? this.posInList[internalId] : -1;

    if (pos < 0 || pos >= list.len || list.buf[pos] !== internalId) {
      pos = -1;
      for (let i = 0; i < list.len; i++) {
        if (list.buf[i] === internalId) {
          pos = i;
          break;
        }
      }
      if (pos === -1) {
        this.assignedList[internalId] = -1;
        this.posInList[internalId] = -1;
        return;
      }
    }

    const moved = list.swapRemoveAt(pos);
    if (moved !== internalId) this.posInList[moved] = pos;
    this.assignedList[internalId] = -1;
    this.posInList[internalId] = -1;
  }

  // optional incremental update for upsert
  onUpsert(internalId: number) {
    if (!this.centroids) return; // not trained yet
    const dim = this.store.dim();
    const v = this.store.getVectorViewByInternalId(internalId);
    if (!v) return;

    const maxId = this.store.maxInternalIdExclusive();
    this.ensureAssignCapacity(maxId);

    const oldLid = internalId < this.assignedList.length ? this.assignedList[internalId] : -1;
    const oldPos = internalId < this.posInList.length ? this.posInList[internalId] : -1;
    const lid = this.nearestCentroid(v, dim);

    if (oldLid === lid && oldLid !== -1 && oldPos !== -1) return;

    if (oldLid !== -1) this.removeFromList(oldLid, internalId);

    const pos = this.lists[lid].push(internalId);
    this.assignedList[internalId] = lid;
    this.posInList[internalId] = pos;
  }

  onDelete(internalId: number) {
    if (!this.centroids) return;
    if (internalId < 0 || internalId >= this.assignedList.length) return;
    const lid = this.assignedList[internalId];
    if (lid === -1) return;
    this.removeFromList(lid, internalId);
  }

  // ---- search ----

  async search(query: Float32Array, opts: IVFSearchOptions): Promise<SearchHit[]> {
    if (!this.centroids) throw new Error("IVF not trained. Call trainAndBuild().");
    const dim = this.store.dim();
    if (query.length !== dim) throw new Error(`Query dim mismatch: expected ${dim}, got ${query.length}`);

    const k = Math.min(opts.k, this.kMax);
    const nprobe = Math.min(opts.nprobe, this.nlist);
    const threshold = opts.scoreThreshold ?? null;

    const probes = this.nearestCentroids(query, dim, nprobe);

    // gather candidates deterministically: probe order then list order
    const maxCand = opts.maxCandidates ?? 200_000;
    const candidates: number[] = [];
    const seen = new Uint8Array(this.store.maxInternalIdExclusive()); // deterministic, fast

    for (const lid of probes) {
      const list = this.lists[lid];
      for (let i = 0; i < list.len; i++) {
        const id = list.buf[i];
        if (id < 0 || id >= seen.length) continue;
        if (seen[id]) continue;
        seen[id] = 1;

        if (!this.store.hasInternalId(id)) continue;
        if (this.assignedList[id] !== lid) continue; // skip stale duplicates
        if (opts.where && !this.meta["matches"]?.(opts.where, id)) {
          // If you didn't expose matches(), simplest: use prefilter bitset approach later.
          // For now: fall back to prefilter list and intersect (see below).
        }

        candidates.push(id);
        if (candidates.length >= maxCand) break;
      }
      if (candidates.length >= maxCand) break;
    }

    // Filtering v1 (deterministic):
    // If where exists, do metadata prefilter and intersect.
    let candIds = Int32Array.from(candidates);
    if (opts.where) {
      const allowed = this.meta.prefilter(opts.where, this.store.maxInternalIdExclusive(), (id) => this.store.hasInternalId(id));
      const allowSet = new Uint8Array(this.store.maxInternalIdExclusive());
      for (let i = 0; i < allowed.length; i++) allowSet[allowed[i]] = 1;

      const filtered: number[] = [];
      for (let i = 0; i < candIds.length; i++) {
        const id = candIds[i];
        if (allowSet[id]) filtered.push(id);
      }
      candIds = Int32Array.from(filtered);
    }

    if (this.gpu) {
      return this.gpuRerank(query, candIds, k, threshold);
    }
    return this.cpuRerank(query, candIds, k, threshold);
  }

  private nearestCentroid(v: Float32Array, dim: number): number {
    let best = 0;
    let bestD = Infinity;
    const c = this.centroids!;
    for (let i = 0; i < this.nlist; i++) {
      const d = l2sq(v, 0, c, i * dim, dim);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  private nearestCentroids(q: Float32Array, dim: number, nprobe: number): number[] {
    const c = this.centroids!;
    const bestIds = new Int32Array(nprobe);
    const bestD = new Float32Array(nprobe);
    bestD.fill(Infinity);
    bestIds.fill(-1);

    for (let i = 0; i < this.nlist; i++) {
      const d = l2sq(q, 0, c, i * dim, dim);
      // insert into small sorted buffer (nprobe is small)
      let j = nprobe - 1;
      if (d >= bestD[j]) continue;
      while (j > 0 && d < bestD[j - 1]) {
        bestD[j] = bestD[j - 1];
        bestIds[j] = bestIds[j - 1];
        j--;
      }
      bestD[j] = d;
      bestIds[j] = i;
    }

    const out: number[] = [];
    for (let i = 0; i < nprobe; i++) if (bestIds[i] >= 0) out.push(bestIds[i]);
    return out;
  }

  private *iterateCandidateBatches(candidateIds: Int32Array, dim: number, batchRows: number) {
    const idsTmp = new Int32Array(Math.min(batchRows, candidateIds.length));
    const vecTmp = new Float32Array(idsTmp.length * dim);

    let count = 0;
    for (let i = 0; i < candidateIds.length; i++) {
      const id = candidateIds[i];
      const v = this.store.getVectorViewByInternalId(id);
      if (!v) continue;

      idsTmp[count] = id;
      vecTmp.set(v, count * dim);
      count++;

      if (count === idsTmp.length) {
        yield { ids: idsTmp.slice(0, count), vectors: vecTmp.slice(0, count * dim) };
        count = 0;
      }
    }
    if (count > 0) yield { ids: idsTmp.slice(0, count), vectors: vecTmp.slice(0, count * dim) };
  }

  private async gpuRerank(query: Float32Array, candidateIds: Int32Array, k: number, threshold: number | null): Promise<SearchHit[]> {
    const dim = this.store.dim();
    const prepared = await this.gpu!.prepareQuery(query, dim, this.metric);

    // min-heap
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

    for (const batch of this.iterateCandidateBatches(candidateIds, dim, this.batchRows)) {
      const pairs = await this.gpu!.topKBatch(batch.vectors, dim, prepared.queryUsed, this.metric, k);
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

    const out: SearchHit[] = [];
    for (const r of tmp) {
      const ext = this.store.getExternalIdByInternal(r.internalId);
      if (!ext) continue;
      out.push({ id: ext, score: r.score, internalId: r.internalId });
    }
    return out;
  }

  private cpuRerank(query: Float32Array, candidateIds: Int32Array, k: number, threshold: number | null): SearchHit[] {
    const dim = this.store.dim();

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

    for (let i = 0; i < candidateIds.length; i++) {
      const id = candidateIds[i];
      const v = this.store.getVectorViewByInternalId(id);
      if (!v) continue;

      let score = 0;
      if (this.metric === "dot") score = dot(v, 0, query, 0, dim);
      else if (this.metric === "l2") score = -l2sq(v, 0, query, 0, dim);
      else {
        // cosine: normalize vector then dot (v1, not optimal)
        let ss = 0;
        for (let j = 0; j < dim; j++) ss += v[j] * v[j];
        const inv = 1 / Math.sqrt(ss || 1e-12);
        let s = 0;
        for (let j = 0; j < dim; j++) s += (v[j] * inv) * query[j];
        score = s;
      }

      if (threshold != null && score < threshold) continue;

      if (size < k) {
        heapId[size] = id;
        heapScore[size] = score;
        siftUp(size);
        size++;
      } else {
        const wi = heapId[0], ws = heapScore[0];
        const better = score > ws || (score === ws && id < wi);
        if (better) {
          heapId[0] = id;
          heapScore[0] = score;
          siftDown(0);
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

  getListSizes(): Int32Array {
    const out = new Int32Array(this.lists.length);
    for (let i = 0; i < this.lists.length; i++) out[i] = this.lists[i].len;
    return out;
  }
}
