import { test, expect } from "@playwright/test";

test("scroll while searching stays within frame budget", async ({ page }) => {
  await page.goto("/");

  const stats = await page.evaluate(async () => {
    const scrollBox = document.querySelector<HTMLDivElement>("#scrollbox")!;
    if (scrollBox.childElementCount === 0) {
      const frag = document.createDocumentFragment();
      for (let i = 0; i < 5000; i++) {
        const row = document.createElement("div");
        row.textContent = `row ${i}`;
        row.style.padding = "2px 6px";
        frag.appendChild(row);
      }
      scrollBox.appendChild(frag);
    }

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

    const dim = 64;
    await call({ t: "open", dbName: `e2e_scroll_${Date.now()}`, dim, metric: "dot", preferGPU: false });
    await call({ t: "generateAndIngest", seed: 77, n: 5000, dim, scale: 0.5, idPrefix: "id_" });

    let seed = 123;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return (seed / 4294967296) * 2 - 1;
    };
    const q = new Float32Array(dim);
    for (let i = 0; i < dim; i++) q[i] = rnd() * 0.5;

    const searches = 200;
    const searchLoop = (async () => {
      for (let i = 0; i < searches; i++) {
        const buf = q.buffer.slice(0);
        await call({ t: "search", query: buf, k: 10 }, [buf]);
        if (i % 10 === 0) await new Promise((r) => setTimeout(r, 0));
      }
    })();

    const deltas: number[] = [];
    let last = performance.now();
    let running = true;
    searchLoop.finally(() => { running = false; });

    const rAFDone = new Promise<void>((resolve) => {
      const tick = (now: number) => {
        const dt = now - last;
        last = now;
        if (running) deltas.push(dt);

        const maxScroll = scrollBox.scrollHeight - scrollBox.clientHeight;
        if (maxScroll > 0) {
          let next = scrollBox.scrollTop + 4;
          if (next > maxScroll) next = 0;
          scrollBox.scrollTop = next;
        }

        if (running) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });

    await rAFDone;
    worker.terminate();

    const sorted = deltas.slice().sort((a, b) => a - b);
    const p95 = sorted.length ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0;
    return { p95, count: deltas.length };
  });

  expect(stats.count).toBeGreaterThan(0);
  expect(stats.p95).toBeLessThan(20);
});
