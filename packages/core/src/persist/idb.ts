export type StoreName =
  | "meta"
  | "snapshots"
  | "vectorChunks"
  | "idMap"
  | "metaRows"
  | "journal";

export interface WispDBIdb extends IDBDatabase {}

export const IDB_VERSION = 1;
export const FORMAT_VERSION = 1;

export function openWispIdb(dbName: string): Promise<WispDBIdb> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(dbName, IDB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;

      if (!db.objectStoreNames.contains("meta")) {
        db.createObjectStore("meta", { keyPath: "key" });
      }
      if (!db.objectStoreNames.contains("snapshots")) {
        db.createObjectStore("snapshots", { keyPath: "snapshotId" });
      }
      if (!db.objectStoreNames.contains("vectorChunks")) {
        db.createObjectStore("vectorChunks", { keyPath: ["snapshotId", "chunkIndex"] });
      }
      if (!db.objectStoreNames.contains("idMap")) {
        db.createObjectStore("idMap", { keyPath: "snapshotId" });
      }
      if (!db.objectStoreNames.contains("metaRows")) {
        db.createObjectStore("metaRows", { keyPath: ["snapshotId", "internalId"] });
      }
      if (!db.objectStoreNames.contains("journal")) {
        db.createObjectStore("journal", { keyPath: "seq" });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function txp<T>(
  db: IDBDatabase,
  stores: StoreName[] | StoreName,
  mode: IDBTransactionMode,
  fn: (tx: IDBTransaction) => Promise<T> | T,
): Promise<T> {
  const storeList = Array.isArray(stores) ? stores : [stores];
  const tx = db.transaction(storeList, mode);
  const done = new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });

  return Promise.resolve(fn(tx)).then(async (v) => {
    await done;
    return v;
  });
}

export function reqp<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function makeSnapshotId(): string {
  // deterministic-ish, sortable-ish
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
