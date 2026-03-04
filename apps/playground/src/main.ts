import "./style.css";
import "./bench/runner";
import { runM7Bench } from "./m7_bench";
import { runM8ChurnBench } from "./m8_churn_bench";
import { WispWorkerClient } from "./worker/client";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <h1>WispDB Playground</h1>
  <div style="display: flex; gap: 8px; flex-wrap: wrap;">
    <button id="run">Run Worker Demo</button>
    <button id="bench">Run M7 Bench</button>
    <button id="m8">Run M8 Churn Bench</button>
    <button id="scroll">Run Scroll DoD</button>
  </div>
  <div id="scrollbox" style="height: 240px; overflow: auto; border: 1px solid #ccc; margin: 8px 0;"></div>
  <pre id="log" style="white-space: pre-wrap; user-select: text;"></pre>
`;

const logEl = document.querySelector<HTMLPreElement>("#log")!;
const btn = document.querySelector<HTMLButtonElement>("#run")!;
const benchBtn = document.querySelector<HTMLButtonElement>("#bench")!;
const m8Btn = document.querySelector<HTMLButtonElement>("#m8")!;
const scrollBtn = document.querySelector<HTMLButtonElement>("#scroll")!;
const scrollBox = document.querySelector<HTMLDivElement>("#scrollbox")!;
const client = new WispWorkerClient();

function log(s: string) {
  logEl.textContent += s + "\n";
}

function ensureRows() {
  if (scrollBox.childElementCount > 0) return;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 5000; i++) {
    const row = document.createElement("div");
    row.textContent = `row ${i}`;
    row.style.padding = "2px 6px";
    frag.appendChild(row);
  }
  scrollBox.appendChild(frag);
}

function percentile(sorted: number[], p: number) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

function frameStats(deltas: number[]) {
  const sorted = deltas.slice().sort((a, b) => a - b);
  const median = percentile(sorted, 0.5);
  const p95 = percentile(sorted, 0.95);
  const max = sorted.length ? sorted[sorted.length - 1] : 0;
  let overBudget = 0;
  for (const d of deltas) if (d > 16.7) overBudget++;
  return { median, p95, max, overBudget, count: deltas.length };
}

function measureFramesWhile(promise: Promise<unknown>) {
  return new Promise<ReturnType<typeof frameStats>>((resolve) => {
    let running = true;
    promise.finally(() => {
      running = false;
    });

    const deltas: number[] = [];
    let last = performance.now();

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

      if (running) {
        requestAnimationFrame(tick);
      } else {
        resolve(frameStats(deltas));
      }
    };

    requestAnimationFrame(tick);
  });
}

async function runWorkerDemo() {
  logEl.textContent = "";
  btn.disabled = true;

  try {
    const info = await client.ping();
    log("Worker ready: " + JSON.stringify(info));

    await client.open("demo", 128, "dot", true);
    log("DB opened in worker.");

    const N = 50000;
    const dim = 128;

    log(`Ingesting ${N} vectors...`);
    const ids: string[] = new Array(N);
    const vecs = new Float32Array(N * dim);

    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return (seed / 4294967296) * 2 - 1;
    };

    for (let i = 0; i < N; i++) {
      ids[i] = `id_${i}`;
      const off = i * dim;
      for (let j = 0; j < dim; j++) vecs[off + j] = rnd() * 0.5;
      if (i % 5000 === 0 && i) {
        log(`... ${i}/${N}`);
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    await client.upsertBatch(ids, vecs, dim);
    log("Ingest done.");

    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = rnd() * 0.5;

    const res = await client.search(q, 10);
    log(`Search took ${res.tookMs.toFixed(2)}ms`);
    for (const h of res.hits) log(`${h.id} score=${h.score.toFixed(6)}`);

    await client.snapshot();
    log("Snapshot saved.");
  } catch (e: any) {
    log("ERROR: " + (e?.message ?? e));
  } finally {
    btn.disabled = false;
  }
}

async function runScrollDoD() {
  logEl.textContent = "";
  btn.disabled = true;
  benchBtn.disabled = true;
  m8Btn.disabled = true;
  scrollBtn.disabled = true;

  try {
    ensureRows();

    const info = await client.ping();
    log("Worker ready: " + JSON.stringify(info));

    const dim = 128;
    await client.open("scroll_demo", dim, "dot", true);
    log("DB opened in worker.");

    const N = 50000;
    const seed = 24680;
    log(`Generating + ingesting ${N} vectors in worker...`);
    await client.generateAndIngest(seed, N, dim, 0.5, "id_");
    log("Ingest done.");

    let s = 13579;
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return (s / 4294967296) * 2 - 1;
    };

    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = rnd() * 0.5;

    const searches = 200;
    const searchLoop = (async () => {
      for (let i = 0; i < searches; i++) {
        await client.search(q, 10);
        if (i % 10 === 0) await new Promise((r) => setTimeout(r, 0));
      }
    })();

    const stats = await measureFramesWhile(searchLoop);
    log(`Frames: count=${stats.count}, median=${stats.median.toFixed(2)}ms, p95=${stats.p95.toFixed(2)}ms, max=${stats.max.toFixed(2)}ms, over16.7ms=${stats.overBudget}`);
  } catch (e: any) {
    log("ERROR: " + (e?.message ?? e));
  } finally {
    btn.disabled = false;
    benchBtn.disabled = false;
    m8Btn.disabled = false;
    scrollBtn.disabled = false;
  }
}

btn.onclick = runWorkerDemo;

benchBtn.onclick = async () => {
  btn.disabled = true;
  benchBtn.disabled = true;
  m8Btn.disabled = true;
  scrollBtn.disabled = true;
  logEl.textContent = "";
  try {
    await runM7Bench(log, null);
  } catch (e: any) {
    log("ERROR: " + (e?.message ?? e));
    console.error(e);
  } finally {
    btn.disabled = false;
    benchBtn.disabled = false;
    m8Btn.disabled = false;
    scrollBtn.disabled = false;
  }
};

m8Btn.onclick = async () => {
  btn.disabled = true;
  benchBtn.disabled = true;
  m8Btn.disabled = true;
  scrollBtn.disabled = true;
  logEl.textContent = "";
  try {
    await runM8ChurnBench(log);
  } catch (e: any) {
    log("ERROR: " + (e?.message ?? e));
    console.error(e);
  } finally {
    btn.disabled = false;
    benchBtn.disabled = false;
    m8Btn.disabled = false;
    scrollBtn.disabled = false;
  }
};

scrollBtn.onclick = runScrollDoD;
