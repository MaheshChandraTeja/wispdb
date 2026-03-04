import { openWispIdb, txp, reqp, makeSnapshotId, FORMAT_VERSION } from "./idb";
import type { MetaState, SnapshotRecord, VectorChunkRecord, IdMapRecord, JournalEntry } from "./types";
import type { StorageEngineV1 } from "../storage/StorageEngineV1";
import type { MetadataStoreV1 } from "../metadata/MetadataStoreV1";

const STATE_KEY = "state" as const;

export class PersistenceV1 {
  private db!: IDBDatabase;

  constructor(private dbName: string) {}

  async open() {
    this.db = await openWispIdb(this.dbName);

    // ensure meta state exists
    await txp(this.db, "meta", "readwrite", async (tx) => {
      const meta = tx.objectStore("meta");
      const cur = await reqp(meta.get(STATE_KEY as any));
      if (!cur) {
        const init: MetaState = { key: "state", activeSnapshotId: null, lastSeq: 0 };
        meta.put(init);
      }
    });
  }

  // ---- Journal ----

  async appendUpsert(externalId: string, vec: Float32Array, metaObj: any | undefined): Promise<number> {
    const vecBuf = vec.buffer.slice(vec.byteOffset, vec.byteOffset + vec.byteLength);

    return txp(this.db, ["meta", "journal"], "readwrite", async (tx) => {
      const meta = tx.objectStore("meta");
      const journal = tx.objectStore("journal");

      const st = (await reqp(meta.get(STATE_KEY as any))) as MetaState;
      const seq = (st.lastSeq + 1) | 0;
      st.lastSeq = seq;

      meta.put(st);

      const entry: JournalEntry = {
        seq,
        ts: Date.now(),
        op: "upsert",
        externalId,
        vec: vecBuf,
        meta: metaObj,
      };
      journal.put(entry);

      return seq;
    });
  }

  async appendDelete(externalId: string): Promise<number> {
    return txp(this.db, ["meta", "journal"], "readwrite", async (tx) => {
      const meta = tx.objectStore("meta");
      const journal = tx.objectStore("journal");

      const st = (await reqp(meta.get(STATE_KEY as any))) as MetaState;
      const seq = (st.lastSeq + 1) | 0;
      st.lastSeq = seq;

      meta.put(st);

      const entry: JournalEntry = {
        seq,
        ts: Date.now(),
        op: "delete",
        externalId,
      };
      journal.put(entry);

      return seq;
    });
  }

  // ---- Snapshot ----

  async saveSnapshot(dim: number, chunkRows: number, storage: StorageEngineV1, metadata: MetadataStoreV1): Promise<string> {
    const snapshotId = makeSnapshotId();

    // Grab current lastSeq (this snapshot includes all journal entries up to that seq)
    const { lastSeq } = await txp(this.db, "meta", "readonly", async (tx) => {
      const st = (await reqp(tx.objectStore("meta").get(STATE_KEY as any))) as MetaState;
      return st;
    });

    const vecSnap = (storage as any).snapshotVectorStore() as {
      dim: number;
      chunkRows: number;
      nextId: number;
      chunks: Array<{ vecs: ArrayBuffer; tomb: ArrayBuffer }>;
    };

    const intToExt = (storage as any).snapshotIdMap() as Array<string | null>;

    const snapRec: SnapshotRecord = {
      snapshotId,
      formatVersion: FORMAT_VERSION,
      status: "writing",
      createdAt: Date.now(),
      dim,
      chunkRows,
      nextId: vecSnap.nextId,
      snapshotSeq: lastSeq,
    };

    // 1) write snapshot header + chunks + idmap + meta rows
    // do it in chunks: multiple transactions are okay because status remains "writing"
    await txp(this.db, "snapshots", "readwrite", async (tx) => {
      tx.objectStore("snapshots").put(snapRec);
    });

    if (typeof window !== "undefined" && (window as any).__WISP_CRASH_AFTER_WRITING_SNAPSHOT_HEADER__) {
      throw new Error("Simulated crash after snapshot header write");
    }

    // vector chunks
    for (let chunkIndex = 0; chunkIndex < vecSnap.chunks.length; chunkIndex++) {
      const rec: VectorChunkRecord = {
        snapshotId,
        chunkIndex,
        vecs: vecSnap.chunks[chunkIndex].vecs,
        tomb: vecSnap.chunks[chunkIndex].tomb,
      };
      await txp(this.db, "vectorChunks", "readwrite", async (tx) => {
        tx.objectStore("vectorChunks").put(rec);
      });

      if (typeof window !== "undefined" && (window as any).__WISP_CRASH_AFTER_SOME_CHUNKS__ && chunkIndex === 2) {
        throw new Error("Simulated crash after chunk 2");
      }
    }

    // id map
    const idRec: IdMapRecord = { snapshotId, intToExt };
    await txp(this.db, "idMap", "readwrite", async (tx) => {
      tx.objectStore("idMap").put(idRec);
    });

    // metadata rows (only for live ids; deterministic)
    // MetadataStoreV1 needs exportRows(): Array<[internalId, any]>
    const rows = (metadata as any).exportRows?.() as Array<[number, any]> | undefined;
    if (rows) {
      for (const [internalId, metaObj] of rows) {
        await txp(this.db, "metaRows", "readwrite", async (tx) => {
          tx.objectStore("metaRows").put({ snapshotId, internalId, meta: metaObj });
        });
      }
    }

    // 2) finalize snapshot atomically: mark complete + set activeSnapshotId
    await txp(this.db, ["snapshots", "meta"], "readwrite", async (tx) => {
      const snaps = tx.objectStore("snapshots");
      const meta = tx.objectStore("meta");

      const latest = (await reqp(snaps.get(snapshotId as any))) as SnapshotRecord;
      latest.status = "complete";
      snaps.put(latest);

      const st = (await reqp(meta.get(STATE_KEY as any))) as MetaState;
      st.activeSnapshotId = snapshotId;
      meta.put(st);
    });

    // 3) truncate journal up to snapshotSeq (best effort)
    await this.truncateJournal(lastSeq);

    return snapshotId;
  }

  private async truncateJournal(uptoSeq: number) {
    await txp(this.db, "journal", "readwrite", async (tx) => {
      const store = tx.objectStore("journal");
      const req = store.openCursor();
      await new Promise<void>((resolve, reject) => {
        req.onsuccess = () => {
          const cursor = req.result;
          if (!cursor) return resolve();
          const key = cursor.key as number;
          if (key <= uptoSeq) cursor.delete();
          cursor.continue();
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async getMetaState(): Promise<MetaState> {
    return txp(this.db, "meta", "readonly", async (tx) => {
      return (await reqp(tx.objectStore("meta").get(STATE_KEY as any))) as MetaState;
    });
  }

  async getLatestCompleteSnapshotInfo(): Promise<SnapshotRecord | null> {
    return txp(this.db, "snapshots", "readonly", async (tx) => {
      const store = tx.objectStore("snapshots");
      const all = await reqp(store.getAll() as any) as SnapshotRecord[];
      const complete = all.filter(s => s.status === "complete");
      if (complete.length === 0) return null;
      complete.sort((a, b) => b.createdAt - a.createdAt);
      return complete[0];
    });
  }

  // ---- Load + Replay ----

  async loadInto(dim: number, chunkRows: number, storage: StorageEngineV1, metadata: MetadataStoreV1): Promise<void> {
    // pick latest complete snapshot
    const snap = await this.getLatestCompleteSnapshot();
    if (snap) {
      if (snap.dim !== dim || snap.chunkRows !== chunkRows) {
        throw new Error(`Snapshot dim/chunkRows mismatch. snapshot=(${snap.dim},${snap.chunkRows}) current=(${dim},${chunkRows})`);
      }

      const chunks = await this.loadVectorChunks(snap.snapshotId);
      (storage as any).restoreVectorStore({
        dim,
        chunkRows,
        nextId: snap.nextId,
        chunks,
      });

      const idMap = await txp(this.db, "idMap", "readonly", async (tx) => {
        return (await reqp(tx.objectStore("idMap").get(snap.snapshotId as any))) as any;
      });
      if (idMap?.intToExt) (storage as any).restoreIdMap(idMap.intToExt);

      // restore metadata
      await this.loadMetadataRowsInto(snap.snapshotId, metadata);

      // replay journal after snapshotSeq
      await this.replayJournalAfter(snap.snapshotSeq, storage, metadata);
    } else {
      // no snapshot: just replay journal from 0
      await this.replayJournalAfter(0, storage, metadata);
    }
  }

  private async getLatestCompleteSnapshot(): Promise<SnapshotRecord | null> {
    return txp(this.db, "snapshots", "readonly", async (tx) => {
      const store = tx.objectStore("snapshots");
      const all = await reqp(store.getAll() as any) as SnapshotRecord[];
      const complete = all.filter(s => s.status === "complete");
      if (complete.length === 0) return null;
      complete.sort((a, b) => b.createdAt - a.createdAt);
      return complete[0];
    });
  }

  private async loadVectorChunks(snapshotId: string) {
    return txp(this.db, "vectorChunks", "readonly", async (tx) => {
      const store = tx.objectStore("vectorChunks");
      const all = (await reqp(store.getAll() as any)) as VectorChunkRecord[];
      const chunks = all
        .filter(r => r.snapshotId === snapshotId)
        .sort((a, b) => a.chunkIndex - b.chunkIndex)
        .map(r => ({ vecs: r.vecs, tomb: r.tomb }));
      return chunks;
    });
  }

  private async loadMetadataRowsInto(snapshotId: string, metadata: MetadataStoreV1) {
    await txp(this.db, "metaRows", "readonly", async (tx) => {
      const store = tx.objectStore("metaRows");
      const all = (await reqp(store.getAll() as any)) as Array<{ snapshotId: string; internalId: number; meta: any }>;
      for (const r of all) {
        if (r.snapshotId !== snapshotId) continue;
        metadata.upsert(r.internalId, r.meta);
      }
    });
  }

  private async replayJournalAfter(afterSeq: number, storage: StorageEngineV1, metadata: MetadataStoreV1) {
    const entries = await txp(this.db, "journal", "readonly", async (tx) => {
      const all = (await reqp(tx.objectStore("journal").getAll() as any)) as JournalEntry[];
      all.sort((a, b) => a.seq - b.seq);
      return all.filter(e => e.seq > afterSeq);
    });

    for (const e of entries) {
      if (e.op === "upsert") {
        const vec = new Float32Array(e.vec!);
        const internal = storage.upsert(e.externalId, vec);
        if (e.meta !== undefined) metadata.upsert(internal, e.meta);
      } else {
        const internal = storage.getInternalId(e.externalId);
        const ok = storage.delete(e.externalId);
        if (ok && internal != null) metadata.delete(internal);
      }
    }
  }
}
