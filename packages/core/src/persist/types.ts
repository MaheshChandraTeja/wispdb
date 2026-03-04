export type JournalOp = "upsert" | "delete";

export interface MetaState {
  key: "state";
  activeSnapshotId: string | null;
  lastSeq: number;
}

export interface SnapshotRecord {
  snapshotId: string;
  formatVersion: number;
  status: "writing" | "complete";
  createdAt: number;
  dim: number;
  chunkRows: number;
  nextId: number;
  snapshotSeq: number; // last journal seq included in this snapshot
}

export interface VectorChunkRecord {
  snapshotId: string;
  chunkIndex: number;
  vecs: ArrayBuffer; // Float32Array buffer
  tomb: ArrayBuffer; // Uint8Array buffer
}

export interface IdMapRecord {
  snapshotId: string;
  intToExt: Array<string | null>;
}

export interface JournalEntry {
  seq: number;
  ts: number;
  op: JournalOp;
  externalId: string;
  vec?: ArrayBuffer; // Float32Array buffer (dim floats)
  meta?: any;
}
