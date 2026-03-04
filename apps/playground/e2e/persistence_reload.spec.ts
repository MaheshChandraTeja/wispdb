import { test, expect } from "@playwright/test";

test("persistence reload + crash snapshot recovery", async ({ page }) => {
  const dbName = `e2e_persist_${Date.now()}`;
  await page.goto("/");

  const seed = await page.evaluate(async (name) => {
    const { PersistentBruteForceDB } = await import("/@id/@wispdb/core");
    const dim = 32;
    const v1 = new Float32Array(dim).fill(0.01);
    const v2 = new Float32Array(dim).fill(0.02);

    const db = new PersistentBruteForceDB(name, { dim, metric: "dot", preferGPU: false });
    await db.open();

    await db.upsert("hello", v1, { tag: "seed" });
    await db.snapshotNow();

    await db.upsert("world", v2, { tag: "seed" });

    // simulate crash during snapshot
    (window as any).__WISP_CRASH_AFTER_SOME_CHUNKS__ = true;
    try {
      await db.snapshotNow();
    } catch {
      // expected
    }
    (window as any).__WISP_CRASH_AFTER_SOME_CHUNKS__ = false;

    return { dim, v1: Array.from(v1), v2: Array.from(v2) };
  }, dbName);

  await page.reload();

  const result = await page.evaluate(async ({ name, seed }) => {
    const { PersistentBruteForceDB } = await import("/@id/@wispdb/core");
    const dim = seed.dim;
    const v1 = new Float32Array(seed.v1);
    const v2 = new Float32Array(seed.v2);

    const db = new PersistentBruteForceDB(name, { dim, metric: "dot", preferGPU: false });
    await db.open();

    const h = db.get("hello");
    const w = db.get("world");

    const r1 = await db.search(v1, { k: 10 });
    const r2 = await db.search(v2, { k: 10 });

    return {
      hello: !!h,
      world: !!w,
      hits1: r1.map((x) => x.id),
      hits2: r2.map((x) => x.id),
    };
  }, { name: dbName, seed });

  expect(result.hello).toBe(true);
  expect(result.world).toBe(true);
  expect(result.hits1).toContain("hello");
  expect(result.hits2).toContain("world");
});
