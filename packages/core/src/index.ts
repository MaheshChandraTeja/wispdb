import { BruteForceIndex, type Metric } from "./db/BruteForceIndex";

export * from "./storage/StorageEngineV1";
export * from "./storage/VectorStore";
export * from "./storage/IdRegistry";
export * from "./db/BruteForceIndex";
export * from "./metadata/types";
export * from "./metadata/MetadataStoreV1";
export * from "./db/PersistentBruteForceDB";
export * from "./persist/PersistenceV1";
export { IVFFlatIndex } from "./ivf/IVFFlatIndex";
export type { IVFSearchOptions, IVFTrainOptions, SearchHit as IVFSearchHit } from "./ivf/IVFFlatIndex";
export * from "./ivf/kmeans";
export * from "./maintenance/types";


export type VectorInput = Float32Array | readonly number[];
export type Metadata = Record<string, unknown>;

export interface WispDBOpenOptions {
  dimensions: number;
  metric?: Metric;
}

export interface WispDBSearchOptions {
  k: number;
  scoreThreshold?: number;
}

export interface SearchResult<TMetadata = Metadata> {
  id: string;
  score: number;
  metadata: TMetadata | undefined;
}

function toFloat32Vector(vec: VectorInput, dimensions: number, label: string): Float32Array {
  if (vec.length !== dimensions) {
    throw new Error(`${label} dimension mismatch: expected ${dimensions}, got ${vec.length}`);
  }
  return vec instanceof Float32Array ? vec : new Float32Array(vec);
}

export class WispDB<TMetadata = Metadata> {
  private readonly index: BruteForceIndex;
  private readonly metadata = new Map<string, TMetadata>();

  private constructor(
    private readonly dimensions: number,
    metric: Metric,
  ) {
    this.index = new BruteForceIndex({
      dim: dimensions,
      metric,
      preferGPU: false,
    });
  }

  static async open<TMetadata = Metadata>(opts: WispDBOpenOptions): Promise<WispDB<TMetadata>> {
    if (!Number.isInteger(opts.dimensions) || opts.dimensions <= 0) {
      throw new Error("dimensions must be a positive integer");
    }

    const db = new WispDB<TMetadata>(opts.dimensions, opts.metric ?? "cosine");
    await db.index.open();
    return db;
  }

  async upsert(id: string, vector: VectorInput, metadata?: TMetadata): Promise<void> {
    if (!id) throw new Error("id is required");

    const v = toFloat32Vector(vector, this.dimensions, "Vector");
    this.index.upsert(id, v, metadata);
    if (metadata !== undefined) this.metadata.set(id, metadata);
  }

  async search(query: VectorInput, opts: WispDBSearchOptions): Promise<Array<SearchResult<TMetadata>>> {
    if (!Number.isInteger(opts.k) || opts.k <= 0) {
      throw new Error("k must be a positive integer");
    }

    const q = toFloat32Vector(query, this.dimensions, "Query");
    const hits = await this.index.search(q, {
      k: opts.k,
      scoreThreshold: opts.scoreThreshold,
    });

    return hits.map((hit) => ({
      id: hit.id,
      score: hit.score,
      metadata: this.metadata.get(hit.id),
    }));
  }

  size(): number {
    return this.metadata.size;
  }
}
