import { BruteForceIndex } from "wispdb";
import { IVFFlatIndex } from "wispdb";

export async function runM7Bench(log: (s: string) => void, _gpuStuff: any) {
  const dim = 128;
  const N = 50_000;
  const Q = 50;
  const k = 10;

  // Build brute-force DB
  const bf = new BruteForceIndex({ dim, metric: "dot", preferGPU: true, batchRows: 16384 });
  await bf.open();

  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 4294967296) * 2 - 1;
  };

  log(`Ingesting ${N}...`);
  for (let i = 0; i < N; i++) {
    const v = new Float32Array(dim);
    for (let j = 0; j < dim; j++) v[j] = rnd() * 0.5;
    bf.upsert(`id_${i}`, v, { bucket: i % 4 });
  }

  // Train IVF (uses BF internals: store + meta + gpu)
  // If you wrapped things differently, pass the right instances.
  // @ts-ignore
  const ivf = new IVFFlatIndex(bf["store"], bf["meta"], bf["gpu"], "dot", 16384);

  log("Training IVF...");
  ivf.trainAndBuild({ nlist: 256, iters: 20, seed: 1234, sampleSize: 20000 });

  const queries: Float32Array[] = [];
  for (let i = 0; i < Q; i++) {
    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = rnd() * 0.5;
    queries.push(q);
  }

  // Warmup
  await bf.search(queries[0], k);
  await ivf.search(queries[0], { k, nprobe: 8 });

  const percentile = (arr: number[], p: number) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * (sorted.length - 1)));
    return sorted[idx];
  };

  // Baseline BF (median/p95)
  const bfTimes: number[] = [];
  const bfRes: Array<Array<number>> = [];
  for (let i = 0; i < Q; i++) {
    const t0 = performance.now();
    const res = await bf.search(queries[i], { k });
    const t1 = performance.now();
    bfTimes.push(t1 - t0);
    bfRes.push(res.map((x: any) => x.internalId));
  }

  const bfMed = percentile(bfTimes, 50);
  const bfP95 = percentile(bfTimes, 95);
  log(`BF median/p95 ms: ${bfMed.toFixed(2)} / ${bfP95.toFixed(2)}`);

  const nlist = 256;
  const nprobes = [1, 2, 4, 8, 16, 32, 64, nlist];

  for (const nprobe of nprobes) {
    const times: number[] = [];
    let hit = 0;
    for (let i = 0; i < Q; i++) {
      const t0 = performance.now();
      const res = await ivf.search(queries[i], { k, nprobe });
      const t1 = performance.now();
      times.push(t1 - t0);

      const ids = res.map((x: any) => x.internalId);
      const set = new Set(bfRes[i]);
      for (const id of ids) if (set.has(id)) hit++;
    }

    const recall = hit / (Q * k);
    const med = percentile(times, 50);
    const p95 = percentile(times, 95);
    const speedup = bfMed / med;

    log(`IVF nprobe=${nprobe} median/p95: ${med.toFixed(2)} / ${p95.toFixed(2)} ms`);
    log(`  recall@${k}: ${(recall * 100).toFixed(1)}%  speedup(vs BF median): ${speedup.toFixed(2)}x`);
  }
}
