export * from "./storage/StorageEngineV1";
export * from "./storage/VectorStore";
export * from "./storage/IdRegistry";
export * from "./db/BruteForceIndex";
export * from "./metadata/types";
export * from "./metadata/MetadataStoreV1";
export * from "./db/PersistentBruteForceDB";
export * from "./persist/PersistenceV1";
export * from "./ivf/IVFFlatIndex";
export * from "./ivf/kmeans";
export * from "./maintenance/types";


export type Vector = Float32Array;

export interface SearchResult {
  id: string;
  score: number;
}

export class WispDB {
  private items = new Map<string, Vector>();

  upsert(id: string, v: Vector) {
    this.items.set(id, v);
  }

  size() {
    return this.items.size;
  }
}
