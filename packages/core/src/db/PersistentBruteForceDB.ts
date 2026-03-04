import { BruteForceIndex, type BruteForceOptions } from "./BruteForceIndex";
import { PersistenceV1 } from "../persist/PersistenceV1";

export class PersistentBruteForceDB {
  private idx: BruteForceIndex;
  private persist: PersistenceV1;

  constructor(private name: string, opts: BruteForceOptions) {
    this.idx = new BruteForceIndex(opts);
    this.persist = new PersistenceV1(`wispdb_${name}`);
  }

  async open() {
    await this.persist.open();

    // load snapshot + replay BEFORE GPU init (storage-only)
    // @ts-ignore access internals: store/meta live inside BruteForceIndex
    await this.persist.loadInto(this.idx["store"].dim(), this.idx["store"]["store"].chunkRows ?? 4096, this.idx["store"], this.idx["meta"]);

    await this.idx.open(); // GPU optional
  }

  get(id: string): Float32Array | null {
    // @ts-ignore
    return this.idx["store"].get(id);
  }

  async upsert(id: string, vec: Float32Array, meta?: any) {
    await this.persist.appendUpsert(id, vec, meta);
    return this.idx.upsert(id, vec, meta);
  }

  async delete(id: string) {
    await this.persist.appendDelete(id);
    return this.idx.delete(id);
  }

  async search(query: Float32Array, args: any) {
    return this.idx.search(query, args as any);
  }

  async snapshotNow() {
    // @ts-ignore
    const dim = this.idx["store"].dim();
    // @ts-ignore
    const chunkRows = this.idx["store"]["store"].chunkRows ?? 4096;
    // @ts-ignore
    return this.persist.saveSnapshot(dim, chunkRows, this.idx["store"], this.idx["meta"]);
  }

  compactInMemory() {
    // @ts-ignore
    return this.idx.compactInMemory();
  }

  async compactNow() {
    this.compactInMemory();
    return this.snapshotNow();
  }
}
