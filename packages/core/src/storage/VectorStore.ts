import { LiveIdSet } from "./LiveIdSet";

export interface VectorStoreStats {
  dim: number;
  chunkRows: number;
  chunks: number;
  liveCount: number;
  deletedCount: number;
  capacity: number;
  nextId: number;
  freeListSize: number;
}

export interface LiveBatch {
  ids: Int32Array;
  vectors: Float32Array;
}

export interface VectorStoreSnapshot {
  dim: number;
  chunkRows: number;
  nextId: number;
  chunks: Array<{ vecs: ArrayBuffer; tomb: ArrayBuffer }>;
}

export class VectorStore {
  readonly dim: number;
  readonly chunkRows: number;

  private readonly CHUNK_SHIFT: number;
  private readonly CHUNK_MASK: number;

  private chunks: Float32Array[] = [];
  private tombstones: Uint8Array[] = [];

  private nextId = 0;
  private liveCount = 0;
  private deletedCount = 0;

  private freeList: number[] = [];
  private liveSet = new LiveIdSet();

  constructor(dim: number, chunkRows = 4096) {
    if (!Number.isInteger(dim) || dim <= 0) throw new Error("dim must be a positive integer");
    if (!Number.isInteger(chunkRows) || chunkRows <= 0) throw new Error("chunkRows must be positive integer");
    if ((chunkRows & (chunkRows - 1)) !== 0) {
      throw new Error("chunkRows must be power of 2 (e.g., 1024, 2048, 4096) for fast indexing");
    }

    this.dim = dim;
    this.chunkRows = chunkRows;

    this.CHUNK_SHIFT = Math.log2(chunkRows) | 0;
    this.CHUNK_MASK = chunkRows - 1;
  }

  /** Allocate an internal id (reuses tombstoned slots first). */
  allocId(): number {
    const reused = this.freeList.pop();
    if (reused !== undefined) {
      // reused slot becomes live again
      this.deletedCount--;
      this.liveCount++;
      this.setTombstone(reused, 0);
      this.liveSet.add(reused);
      return reused;
    }

    const id = this.nextId++;
    if (id > 0x7fffffff) throw new Error("Exceeded int32 id capacity");

    this.ensureCapacityForId(id);
    this.setTombstone(id, 0);
    this.liveCount++;
    this.liveSet.add(id);
    return id;
  }

  /** Mark internal id as deleted and make slot reusable later. */
  deleteId(id: number): void {
    if (!this.isValidId(id)) return;
    if (this.getTombstone(id) === 1) return;

    this.setTombstone(id, 1);
    this.liveCount--;
    this.deletedCount++;
    this.freeList.push(id);
    this.liveSet.remove(id);
  }

  upsertByInternalId(id: number, vec: Float32Array): void {
    if (vec.length !== this.dim) throw new Error(`Vector dim mismatch: expected ${this.dim}, got ${vec.length}`);
    if (!this.isValidId(id)) throw new Error(`Invalid internal id ${id}`);

    this.ensureCapacityForId(id);
    // If it was tombstoned, revive counts
    if (this.getTombstone(id) === 1) {
      this.setTombstone(id, 0);
      this.deletedCount--;
      this.liveCount++;
      // Also: remove from freelist if present. We keep freelist stack simple; reused path uses allocId().
      // If caller uses upsertByInternalId directly, it can “revive” without using allocId. That’s OK.
    }

    this.liveSet.add(id);

    const { chunk, offset } = this.addr(id);
    chunk.set(vec, offset);
  }

  /** Returns a COPY (safe). For a view without copy, use getViewByInternalId(). */
  getByInternalId(id: number): Float32Array | null {
    const view = this.getViewByInternalId(id);
    return view ? new Float32Array(view) : null;
  }

  /** Returns a view backed by internal storage. Do not mutate unless you know what you're doing. */
  getViewByInternalId(id: number): Float32Array | null {
    if (!this.isValidId(id)) return null;
    if (this.getTombstone(id) === 1) return null;

    const { chunk, offset } = this.addr(id);
    return chunk.subarray(offset, offset + this.dim);
  }

  hasInternalId(id: number): boolean {
    return this.isValidId(id) && this.getTombstone(id) === 0;
  }

  stats(): VectorStoreStats {
    return {
      dim: this.dim,
      chunkRows: this.chunkRows,
      chunks: this.chunks.length,
      liveCount: this.liveCount,
      deletedCount: this.deletedCount,
      capacity: this.chunks.length * this.chunkRows,
      nextId: this.nextId,
      freeListSize: this.freeList.length,
    };
  }

  snapshot(): VectorStoreSnapshot {
    const chunks = this.chunks.map((c, i) => ({
      vecs: c.buffer.slice(0),
      tomb: this.tombstones[i].buffer.slice(0),
    }));
    return {
      dim: this.dim,
      chunkRows: this.chunkRows,
      nextId: this.stats().nextId,
      chunks,
    };
  }

  compactTrailingDead(): { oldNextId: number; newNextId: number; trimmedChunks: number } {
    const oldNextId = this.nextId;

    let lastLive = -1;
    for (let id = oldNextId - 1; id >= 0; id--) {
      if (this.getTombstone(id) === 0) {
        lastLive = id;
        break;
      }
    }

    const newNextId = lastLive + 1;
    if (newNextId === oldNextId) {
      return { oldNextId, newNextId, trimmedChunks: 0 };
    }

    const oldChunkCount = this.chunks.length;
    const newChunkCount = newNextId <= 0 ? 0 : ((newNextId - 1) >> this.CHUNK_SHIFT) + 1;

    this.chunks = this.chunks.slice(0, newChunkCount);
    this.tombstones = this.tombstones.slice(0, newChunkCount);
    this.nextId = newNextId;

    // recompute counts + liveSet
    this.liveCount = 0;
    this.deletedCount = 0;
    this.liveSet = new LiveIdSet();
    this.liveSet.ensurePosCapacity(newNextId);
    for (let id = 0; id < newNextId; id++) {
      if (this.getTombstone(id) === 1) {
        this.deletedCount++;
      } else {
        this.liveCount++;
        this.liveSet.add(id);
      }
    }

    // rebuild freeList (descending so pop() returns smallest ids)
    this.freeList = [];
    for (let id = newNextId - 1; id >= 0; id--) {
      if (this.getTombstone(id) === 1) this.freeList.push(id);
    }

    return { oldNextId, newNextId, trimmedChunks: oldChunkCount - newChunkCount };
  }

  restoreFromSnapshot(s: VectorStoreSnapshot) {
    if (s.dim !== this.dim) throw new Error("dim mismatch in restore");
    if (s.chunkRows !== this.chunkRows) throw new Error("chunkRows mismatch in restore");

    this.chunks = s.chunks.map((x) => new Float32Array(x.vecs));
    this.tombstones = s.chunks.map((x) => new Uint8Array(x.tomb));

    // restore nextId + counts (recompute)
    // easiest deterministic way: set nextId and recount live/deleted
    this.nextId = s.nextId;

    // recompute counts
    this.liveCount = 0;
    this.deletedCount = 0;
    this.liveSet = new LiveIdSet();
    this.liveSet.ensurePosCapacity(s.nextId);
    for (let id = 0; id < s.nextId; id++) {
      if (this.getTombstone(id) === 1) {
        this.deletedCount++;
      } else {
        this.liveCount++;
        this.liveSet.add(id);
      }
    }

    // freeList rebuild (simple): any tombstoned ids are reusable
    this.freeList = [];
    for (let id = s.nextId - 1; id >= 0; id--) {
      if (this.getTombstone(id) === 1) this.freeList.push(id);
    }
  }

  *iterateLiveBatches(batchRows = 16384): Generator<LiveBatch> {
    if (!Number.isInteger(batchRows) || batchRows <= 0) throw new Error("batchRows must be > 0");

    for (const ids of this.liveSet.iterateBatches(batchRows)) {
      const vectors = new Float32Array(ids.length * this.dim);
      for (let i = 0; i < ids.length; i++) {
        const view = this.getViewByInternalId(ids[i]);
        if (!view) continue;
        vectors.set(view, i * this.dim);
      }
      yield { ids, vectors };
    }
  }

  // ---------- internals ----------

  private isValidId(id: number): boolean {
    return Number.isInteger(id) && id >= 0 && id < this.nextId;
  }

  private ensureCapacityForId(id: number): void {
    const needChunkIndex = id >> this.CHUNK_SHIFT;
    while (this.chunks.length <= needChunkIndex) {
      this.chunks.push(new Float32Array(this.chunkRows * this.dim));
      this.tombstones.push(new Uint8Array(this.chunkRows)); // default 0 (live), but we only mark used ids explicitly
      // Important: unused slots should behave as "deleted" logically; we handle that by checking id < nextId.
    }
  }

  private addr(id: number): { chunk: Float32Array; offset: number } {
    const chunkIndex = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    const offset = row * this.dim;
    return { chunk: this.chunks[chunkIndex], offset };
  }

  private getTombstone(id: number): 0 | 1 {
    const chunkIndex = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    return (this.tombstones[chunkIndex][row] as 0 | 1) ?? 1;
  }

  private setTombstone(id: number, v: 0 | 1): void {
    const chunkIndex = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    this.tombstones[chunkIndex][row] = v;
  }
}
