import { test, expect } from "@playwright/test";

test("worker smoke: open, ingest, search deterministic", async ({ page }) => {
  await page.goto("/");

  const result = await page.evaluate(async () => {
    const dbName = `e2e_worker_${Date.now()}`;
    const dim = 64;

    const worker = new Worker(new URL("/src/worker/wispdb.worker.ts", location.href), { type: "module" });
    let nextId = 1;
    const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

    worker.onmessage = (ev) => {
      const res = ev.data as { id: number; ok: boolean; v?: any; err?: string };
      const p = pending.get(res.id);
      if (!p) return;
      pending.delete(res.id);
      if (res.ok) p.resolve(res.v);
      else p.reject(new Error(res.err));
    };

    const call = (req: any, transfer?: Transferable[]) => {
      const id = nextId++;
      const msg = { id, req };
      return new Promise<any>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage(msg, transfer ?? []);
      });
    };

    const ping = await call({ t: "ping" });
    await call({ t: "open", dbName, dim, metric: "dot", preferGPU: false });
    await call({ t: "generateAndIngest", seed: 123, n: 1000, dim, scale: 0.5, idPrefix: "id_" });

    let seed = 999;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return (seed / 4294967296) * 2 - 1;
    };

    const makeQuery = () => {
      const q = new Float32Array(dim);
      for (let i = 0; i < dim; i++) q[i] = rnd() * 0.5;
      return q;
    };

    const q1 = makeQuery();
    const buf1 = q1.buffer.slice(0);
    const res1 = await call({ t: "search", query: buf1, k: 10 }, [buf1]);

    const buf2 = q1.buffer.slice(0);
    const res2 = await call({ t: "search", query: buf2, k: 10 }, [buf2]);

    const hash = (hits: any[]) =>
      hits.map((h) => `${h.internalId}:${Number(h.score).toFixed(6)}`).join("|");

    worker.terminate();
    return { ping, hash1: hash(res1.hits), hash2: hash(res2.hits), hits: res1.hits.length };
  });

  expect(result.ping.worker).toBe(true);
  expect(result.hits).toBeGreaterThan(0);
  expect(result.hash1).toBe(result.hash2);
});
