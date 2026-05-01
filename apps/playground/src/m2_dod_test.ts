import { DeviceManager } from "@wispdb/gpu";
import { ExactSearchGPU } from "@wispdb/gpu";

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cpuScoreAll(vectors: Float32Array, dim: number, query: Float32Array, metric: "dot"|"cosine"|"l2"): Float32Array {
  const n = vectors.length / dim;

  // For cosine CPU baseline: normalize vectors+query in float32-ish way
  let q = query;
  if (metric === "cosine") {
    const qn = new Float32Array(dim);
    let ss = 0;
    for (let j = 0; j < dim; j++) ss = Math.fround(ss + Math.fround(query[j]*query[j]));
    const inv = 1 / Math.sqrt(ss || 1e-12);
    for (let j = 0; j < dim; j++) qn[j] = Math.fround(query[j] * inv);
    q = qn;
  }

  const scores = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const base = i * dim;

    if (metric === "dot") {
      let s = 0;
      for (let j = 0; j < dim; j++) s = Math.fround(s + Math.fround(vectors[base+j] * q[j]));
      scores[i] = s;
    } else if (metric === "l2") {
      let s = 0;
      for (let j = 0; j < dim; j++) {
        const d = Math.fround(vectors[base+j] - q[j]);
        s = Math.fround(s + Math.fround(d*d));
      }
      scores[i] = -s; // match GPU ranking convention
    } else {
      // cosine: normalize vector i and dot with normalized q
      let ss = 0;
      for (let j = 0; j < dim; j++) ss = Math.fround(ss + Math.fround(vectors[base+j]*vectors[base+j]));
      const inv = 1 / Math.sqrt(ss || 1e-12);
      let s = 0;
      for (let j = 0; j < dim; j++) s = Math.fround(s + Math.fround(Math.fround(vectors[base+j] * inv) * q[j]));
      scores[i] = s;
    }
  }

  return scores;
}

function topkCPU(scores: Float32Array, k: number) {
  const idx = Array.from(scores.keys());
  idx.sort((a, b) => (scores[b] - scores[a]) || (a - b));
  return idx.slice(0, k);
}

export async function runMilestone2DoD(log: (s: string) => void) {
  const rng = mulberry32(12345);

  const n = 4096;
  const dim = 128;
  const k = 10;

  const vectors = new Float32Array(n * dim);
  for (let i = 0; i < vectors.length; i++) vectors[i] = Math.fround((rng() * 2 - 1) * 0.5);

  const query = new Float32Array(dim);
  for (let j = 0; j < dim; j++) query[j] = Math.fround((rng() * 2 - 1) * 0.5);

  const dm = new DeviceManager({ labelPrefix: "wispdb-m2-test" });
  const gpu = new ExactSearchGPU(dm, "wispdb-m2-test");
  await gpu.init();
  gpu.setDataset(vectors, dim);

  const metrics: Array<"dot"|"cosine"|"l2"> = ["dot", "cosine", "l2"];

  for (const metric of metrics) {
    log(`\nMetric: ${metric}`);

    const gpuRes = await gpu.searchExact(query, k, metric, "gpu_block");
    const gpuIdx = gpuRes.map(r => r.index);

    const cpuScores = cpuScoreAll(vectors, dim, query, metric);
    const cpuIdx = topkCPU(cpuScores, k);

    const ok = gpuIdx.join(",") === cpuIdx.join(",");
    if (!ok) {
      log(`❌ Mismatch`);
      log(`GPU: ${gpuIdx.join(",")}`);
      log(`CPU: ${cpuIdx.join(",")}`);
      throw new Error(`DoD failed for metric=${metric}`);
    } else {
      log(`✅ TopK match: ${gpuIdx.join(",")}`);
    }
  }

  log(`\n✅ Milestone 2 DoD PASSED: searchExact(v,k) matches CPU baseline`);
}
