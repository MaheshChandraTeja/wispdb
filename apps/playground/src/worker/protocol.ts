export type Metric = "dot" | "cosine" | "l2";

export type WhereClause = Record<string, any>;

export type Req =
  | { t: "open"; dbName: string; dim: number; metric: Metric; preferGPU?: boolean }
  | { t: "upsertBatch"; ids: string[]; vectors: ArrayBuffer; dim: number; metas?: any[] }
  | { t: "generateAndIngest"; seed: number; n: number; dim: number; scale?: number; idPrefix?: string; idOffset?: number }
  | { t: "ivfBuild"; nlist: number; iters?: number; seed?: number; sampleSize?: number; batchRows?: number }
  | { t: "ivfSearch"; query: ArrayBuffer; k: number; nprobe: number }
  | { t: "delete"; id: string }
  | { t: "search"; query: ArrayBuffer; k: number; where?: WhereClause; scoreThreshold?: number | null }
  | { t: "snapshot" }
  | { t: "stats" }
  | { t: "ping" };

export type Res =
  | { t: "res"; id: number; ok: true; v: any }
  | { t: "res"; id: number; ok: false; err: string };

export type Msg = { id: number; req: Req };
