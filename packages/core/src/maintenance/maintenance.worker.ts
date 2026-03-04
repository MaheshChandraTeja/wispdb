import { PersistenceV1 } from "../persist/PersistenceV1";
import { StorageEngineV1 } from "../storage/StorageEngineV1";
import { MetadataStoreV1 } from "../metadata/MetadataStoreV1";
import { IVFFlatIndex } from "../ivf/IVFFlatIndex";
import type { MaintenanceRequest, MaintenanceResponse, MaintenanceStats, MaintenanceActions, IVFStatsConfig } from "./types";

let persist: PersistenceV1 | null = null;
let persistName = "";

async function ensurePersistence(dbName: string) {
  if (persist && persistName === dbName) return;
  persist = new PersistenceV1(dbName);
  await persist.open();
  persistName = dbName;
}

async function loadState(dbName: string, dim: number, chunkRows: number) {
  await ensurePersistence(dbName);
  const store = new StorageEngineV1(dim, chunkRows);
  const meta = new MetadataStoreV1({}, 4096);
  await persist!.loadInto(dim, chunkRows, store, meta);
  return { store, meta };
}

function countTrailingDead(store: StorageEngineV1, nextId: number): number {
  let count = 0;
  for (let id = nextId - 1; id >= 0; id--) {
    if (store.hasInternalId(id)) break;
    count++;
  }
  return count;
}

function listStatsFromSizes(sizes: Int32Array | number[]) {
  if (sizes.length === 0) return { listImbalance: 0, listStd: 0 };
  let sum = 0;
  let max = 0;
  for (let i = 0; i < sizes.length; i++) {
    const v = sizes[i] | 0;
    sum += v;
    if (v > max) max = v;
  }
  const avg = sum / sizes.length;
  let varSum = 0;
  for (let i = 0; i < sizes.length; i++) {
    const d = (sizes[i] as number) - avg;
    varSum += d * d;
  }
  const listStd = Math.sqrt(varSum / sizes.length);
  const listImbalance = avg > 0 ? max / avg : 0;
  return { listImbalance, listStd };
}

async function maybeBuildIvfStats(store: StorageEngineV1, meta: MetadataStoreV1, cfg?: IVFStatsConfig) {
  if (!cfg) return undefined;
  const nlist = cfg.nlist;
  if (!Number.isInteger(nlist) || nlist <= 0) return undefined;

  const ivf = new IVFFlatIndex(store, meta, null, cfg.metric ?? "dot", cfg.batchRows ?? 1024);
  ivf.trainAndBuild({
    nlist,
    iters: cfg.iters,
    seed: cfg.seed,
    sampleSize: cfg.sampleSize,
  });
  const sizes = ivf.getListSizes();
  const { listImbalance, listStd } = listStatsFromSizes(sizes);
  return { listSizes: Array.from(sizes), listImbalance, listStd };
}

async function computeStats(store: StorageEngineV1, meta: MetadataStoreV1, cfg?: IVFStatsConfig): Promise<MaintenanceStats> {
  const s = store.stats();
  const live = s.vectors.liveCount;
  const deleted = s.vectors.deletedCount;
  const total = live + deleted;
  const deadRatio = total > 0 ? deleted / total : 0;
  const tailDeadCount = countTrailingDead(store, s.vectors.nextId);
  const tailDeadRatio = s.vectors.nextId > 0 ? tailDeadCount / s.vectors.nextId : 0;

  let listSizes: number[] | undefined;
  let listImbalance: number | undefined;
  let listStd: number | undefined;
  if (cfg) {
    const ivfStats = await maybeBuildIvfStats(store, meta, cfg);
    if (ivfStats) {
      listSizes = ivfStats.listSizes;
      listImbalance = ivfStats.listImbalance;
      listStd = ivfStats.listStd;
    }
  }

  let journalSeq: number | undefined;
  let lastSnapshotAt: number | null | undefined;
  if (persist) {
    const st = await persist.getMetaState();
    journalSeq = st.lastSeq;
    const snap = await persist.getLatestCompleteSnapshotInfo();
    lastSnapshotAt = snap?.createdAt ?? null;
  }

  return {
    liveCount: live,
    deletedCount: deleted,
    deadRatio,
    tailDeadCount,
    tailDeadRatio,
    nextId: s.vectors.nextId,
    idCount: s.idCount,
    chunkRows: s.vectors.chunkRows,
    chunks: s.vectors.chunks,
    listSizes,
    listImbalance,
    listStd,
    journalSeq,
    lastSnapshotAt,
  };
}

async function runMaintenance(msg: Extract<MaintenanceRequest, { type: "RUN_MAINTENANCE" } | { type: "FORCE_SNAPSHOT" }>) {
  const chunkRows = msg.chunkRows ?? 4096;
  const { store, meta } = await loadState(msg.dbName, msg.dim, chunkRows);

  const actions: MaintenanceActions = {};
  let stats = await computeStats(store, meta, msg.type === "RUN_MAINTENANCE" ? msg.policy?.ivfStats : undefined);

  if (msg.type === "RUN_MAINTENANCE") {
    const policy = msg.policy ?? {};

    const shouldTrim = policy.trimTail ?? true;
    if (shouldTrim) {
      const minDead = policy.trimTailMinDead ?? 1;
      if (stats.tailDeadCount >= minDead) {
        const res = store.compactTrailingDead();
        if (res) {
          meta.compactToMaxId(res.newNextId);
          actions.trimmedChunks = res.trimmedChunks;
          actions.newNextId = res.newNextId;
        }
      }
    }

    let doSnapshot = !!policy.snapshot;
    if (!doSnapshot && typeof policy.snapshotIfDeadRatio === "number") {
      doSnapshot = stats.deadRatio >= policy.snapshotIfDeadRatio;
    }
    if (!doSnapshot && typeof policy.snapshotMinIntervalMs === "number") {
      const snap = await persist!.getLatestCompleteSnapshotInfo();
      const lastAt = snap?.createdAt ?? 0;
      if (Date.now() - lastAt >= policy.snapshotMinIntervalMs) doSnapshot = true;
    }

    if (doSnapshot) {
      const snapshotId = await persist!.saveSnapshot(msg.dim, chunkRows, store, meta);
      actions.snapshotId = snapshotId;
    }

    if (policy.ivfRebuild || (typeof policy.ivfRebuildIfDeadRatio === "number" && stats.deadRatio >= policy.ivfRebuildIfDeadRatio)) {
      actions.ivfRebuild = true;
    }
  } else {
    const snapshotId = await persist!.saveSnapshot(msg.dim, chunkRows, store, meta);
    actions.snapshotId = snapshotId;
  }

  stats = await computeStats(store, meta, msg.type === "RUN_MAINTENANCE" ? msg.policy?.ivfStats : undefined);
  return { stats, actions };
}

const ctx: Worker = self as any;

ctx.onmessage = async (ev: MessageEvent<MaintenanceRequest>) => {
  const msg = ev.data;
  try {
    if (msg.type === "STATS_REQUEST") {
      const { store, meta } = await loadState(msg.dbName, msg.dim, msg.chunkRows ?? 4096);
      const stats = await computeStats(store, meta, msg.policy?.ivfStats);
      const resp: MaintenanceResponse = { type: "STATS", stats, requestId: msg.requestId };
      ctx.postMessage(resp);
      return;
    }

    if (msg.type === "RUN_MAINTENANCE" || msg.type === "FORCE_SNAPSHOT") {
      const { stats, actions } = await runMaintenance(msg);
      const resp: MaintenanceResponse = { type: "MAINTENANCE_DONE", stats, actions, requestId: msg.requestId };
      ctx.postMessage(resp);
      return;
    }
  } catch (err: any) {
    const resp: MaintenanceResponse = {
      type: "MAINTENANCE_DONE",
      stats: {
        liveCount: 0,
        deletedCount: 0,
        deadRatio: 0,
        tailDeadCount: 0,
        tailDeadRatio: 0,
        nextId: 0,
        idCount: 0,
        chunkRows: msg.chunkRows ?? 4096,
        chunks: 0,
        lastSnapshotAt: null,
      },
      actions: { snapshotId: null },
      requestId: msg.requestId,
    };
    (resp as any).error = err?.message ?? String(err);
    ctx.postMessage(resp);
  }
};
