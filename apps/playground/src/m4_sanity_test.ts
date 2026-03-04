import { BruteForceIndex, type Metric } from "@wispdb/core";
import { isWebGPUAvailable } from "@wispdb/gpu";

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export async function runM4SanityTest(log: (s: string) => void) {
  if (!isWebGPUAvailable()) {
    log("Sanity check: WebGPU not available; skipping GPU vs CPU test.");
    return;
  }

  const dim = 128;
  const N = 4096;
  const k = 10;
  const metrics: Metric[] = ["dot", "cosine", "l2"];

  const rng = mulberry32(12345);
  const vectors: Float32Array[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const v = new Float32Array(dim);
    for (let j = 0; j < dim; j++) v[j] = (rng() * 2 - 1) * 0.5;
    vectors[i] = v;
  }

  const query = new Float32Array(dim);
  for (let j = 0; j < dim; j++) query[j] = (rng() * 2 - 1) * 0.5;

  for (const metric of metrics) {
    log(`\nSanity check (GPU vs CPU) metric=${metric}...`);

    const gpu = new BruteForceIndex({ dim, metric, batchRows: 1024, preferGPU: true });
    await gpu.open();
    const cpu = new BruteForceIndex({ dim, metric, batchRows: 1024, preferGPU: false });

    for (let i = 0; i < N; i++) {
      const id = `id_${i}`;
      const v = vectors[i];
      gpu.upsert(id, v);
      cpu.upsert(id, v);
    }

    const gpuHits = await gpu.search(query, k);
    const cpuHits = await cpu.search(query, k);

    const gpuIds = gpuHits.map((h) => h.internalId);
    const cpuIds = cpuHits.map((h) => h.internalId);
    const ok =
      gpuIds.length === cpuIds.length &&
      gpuIds.every((id, i) => id === cpuIds[i]);

    if (!ok) {
      log(`❌ Mismatch metric=${metric}`);
      log(`GPU: ${gpuIds.join(",")}`);
      log(`CPU: ${cpuIds.join(",")}`);
      throw new Error(`M4 sanity check failed (metric=${metric})`);
    }

    log(`✅ TopK match: ${gpuIds.join(",")}`);
  }
}
