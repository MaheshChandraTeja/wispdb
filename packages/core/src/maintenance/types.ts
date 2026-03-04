import type { Metric } from "../db/BruteForceIndex";

export interface MaintenancePolicy {
  trimTail?: boolean;
  trimTailMinDead?: number;
  snapshot?: boolean;
  snapshotIfDeadRatio?: number;
  snapshotMinIntervalMs?: number;
  ivfRebuild?: boolean;
  ivfRebuildIfDeadRatio?: number;
  ivfStats?: IVFStatsConfig;
}

export interface IVFStatsConfig {
  nlist: number;
  iters?: number;
  seed?: number;
  sampleSize?: number;
  metric?: Metric;
  batchRows?: number;
}

export interface MaintenanceStats {
  liveCount: number;
  deletedCount: number;
  deadRatio: number;
  tailDeadCount: number;
  tailDeadRatio: number;
  nextId: number;
  idCount: number;
  chunkRows: number;
  chunks: number;
  listSizes?: number[];
  listImbalance?: number;
  listStd?: number;
  journalSeq?: number;
  lastSnapshotAt?: number | null;
}

export interface MaintenanceActions {
  trimmedChunks?: number;
  newNextId?: number;
  snapshotId?: string | null;
  ivfRebuild?: boolean;
}

export type MaintenanceRequest =
  | {
      type: "STATS_REQUEST";
      dbName: string;
      dim: number;
      chunkRows?: number;
      policy?: MaintenancePolicy;
      requestId?: number;
    }
  | {
      type: "RUN_MAINTENANCE";
      dbName: string;
      dim: number;
      chunkRows?: number;
      policy: MaintenancePolicy;
      requestId?: number;
    }
  | {
      type: "FORCE_SNAPSHOT";
      dbName: string;
      dim: number;
      chunkRows?: number;
      requestId?: number;
    };

export type MaintenanceResponse =
  | {
      type: "STATS";
      stats: MaintenanceStats;
      requestId?: number;
    }
  | {
      type: "MAINTENANCE_DONE";
      stats: MaintenanceStats;
      actions: MaintenanceActions;
      requestId?: number;
    };
