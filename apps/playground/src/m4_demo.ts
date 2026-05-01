import { PersistentBruteForceDB } from "wispdb";

export async function runM4Demo(log: (s: string) => void) {
  const dim = 128;
  const db = new PersistentBruteForceDB("demo", { dim, metric: "dot", preferGPU: true });
  await db.open();
  (window as any).__wispdb = db;

  const seedVec = new Float32Array(dim).fill(0.01);

  if (!db.get("hello")) {
    await db.upsert("hello", seedVec, { tag: "seed" });
  }

  const hits = await db.search(seedVec, { k: 10 });
  log(JSON.stringify(hits, null, 2));

  await db.snapshotNow();
  log("Snapshot saved.");
  log("DoD: refresh page, run search again; results should persist.");
  log("DoD: start snapshotNow(), hard refresh mid-snapshot; reload should recover from last complete snapshot + journal.");
}
