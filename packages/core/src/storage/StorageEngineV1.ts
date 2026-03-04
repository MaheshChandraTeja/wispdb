import { VectorStore } from "./VectorStore";
import { IdRegistry } from "./IdRegistry";

export interface StorageEngineStats {
  vectors: ReturnType<VectorStore["stats"]>;
  idCount: number;
}

export class StorageEngineV1 {
  private store: VectorStore;
  private ids = new IdRegistry();

  constructor(dim: number, chunkRows = 4096) {
    this.store = new VectorStore(dim, chunkRows);
  }

  dim(): number {
    return this.store.dim;
  }

  upsert(externalId: string, vec: Float32Array): number {
    const existing = this.ids.getInternal(externalId);
    if (existing !== null) {
      this.store.upsertByInternalId(existing, vec);
      return existing;
    }

    const internal = this.store.allocId();
    this.ids.bind(externalId, internal);
    this.store.upsertByInternalId(internal, vec);
    return internal;
  }

  get(externalId: string): Float32Array | null {
    const internal = this.ids.getInternal(externalId);
    if (internal === null) return null;
    return this.store.getByInternalId(internal);
  }

  getInternalId(externalId: string): number | null {
    return this.ids.getInternal(externalId);
  }

  getExternalIdByInternal(internalId: number): string | null {
    return this.ids.getExternal(internalId);
  }

  getVectorViewByInternalId(internalId: number): Float32Array | null {
    // @ts-ignore
    return this.store.getViewByInternalId(internalId);
  }

  hasInternalId(internalId: number): boolean {
    // @ts-ignore
    return this.store.hasInternalId(internalId);
  }

  maxInternalIdExclusive(): number {
    // nextId
    return this.stats().vectors.nextId;
  }

  snapshotIdMap(): Array<string | null> {
    return this.ids.exportIntToExt();
  }

  restoreIdMap(intToExt: Array<string | null>) {
    this.ids.restoreFromIntToExt(intToExt);
  }

  snapshotVectorStore() {
    return this.store.snapshot();
  }

  restoreVectorStore(s: any) {
    return this.store.restoreFromSnapshot(s);
  }

  compactTrailingDead(): { oldNextId: number; newNextId: number; trimmedChunks: number } {
    const res = this.store.compactTrailingDead();
    this.ids.compactToMaxId(res.newNextId);
    return res;
  }

  delete(externalId: string): boolean {
    const internal = this.ids.unbind(externalId);
    if (internal === null) return false;
    this.store.deleteId(internal);
    return true;
  }

  iterateLiveBatches(batchRows = 16384) {
    return this.store.iterateLiveBatches(batchRows);
  }

  stats(): StorageEngineStats {
    return {
      vectors: this.store.stats(),
      idCount: this.ids.size(),
    };
  }
}
