import {
  DeviceManager,
  PipelineCache,
  BufferPool,
  CommandScheduler,
} from "@wispdb/gpu";

/**
 * WebGPU Backbone Soak Test
 * - Repeats a compute shader thousands of times
 * - Uses PipelineCache (compile once)
 * - Uses BufferPool (reuse buffers across chunks)
 * - Uses CommandScheduler (clean submits + throttling)
 * - Validates correctness via readback
 * - Reports timing + pool stats + optional JS heap deltas
 */
export async function runWebGPUBackboneSoakTest(opts?: {
  n?: number;            // elements
  iterations?: number;   // total dispatch count
  chunkSize?: number;    // submits per chunk, then flush
  maxInFlight?: number;  // scheduler throttle
  log?: (s: string) => void;
}) {
  const log = opts?.log ?? console.log;

  // Choose values that won’t make your laptop ignite.
  const n = opts?.n ?? (1 << 20); // 1,048,576
  const iterations = opts?.iterations ?? 2000;
  const chunkSize = opts?.chunkSize ?? 50;
  const maxInFlight = opts?.maxInFlight ?? 2;

  if (!navigator.gpu) {
    throw new Error("WebGPU not available (navigator.gpu missing). Use Chromium-based browser.");
  }

  // Optional heap snapshot (Chromium only)
  const heapStart =
    (performance as any).memory?.usedJSHeapSize ?? null;

  // 1) Init device
  const dm = new DeviceManager({ labelPrefix: "wispdb-test" });
  await dm.init();
  const device = dm.getDevice();
  const queue = dm.getQueue();

  // 2) Backbone pieces
  const cache = new PipelineCache(device, "wispdb-test");
  const pool = new BufferPool(device, { maxBytes: 128 * 1024 * 1024 }, "wispdb-test");
  const scheduler = new CommandScheduler(device, queue, {
    maxInFlight,
    labelPrefix: "wispdb-test",
  });

  // 3) Shader (bounds-safe, so n doesn’t need to be divisible by 256)
  const wgsl = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform> params : vec4<u32>; // params.x = n

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.x) { return; }
  out[i] = a[i] + b[i];
}
`;

  // 4) Pipeline compiled once (cache proves it)
  const pipeline = cache.getComputePipeline({ code: wgsl, label: "backboneSoak" });
  const bgl = pipeline.getBindGroupLayout(0);

  // 5) Data setup
  const bytes = n * 4;

  const aCPU = new Float32Array(n);
  const bCPU = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    aCPU[i] = i * 1.0;
    bCPU[i] = i * 2.0;
  }

  // Upload buffers (pooled or not? We pool these too for consistency.)
  const aBuf = pool.acquire(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "a");
  const bBuf = pool.acquire(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, "b");
  queue.writeBuffer(aBuf, 0, aCPU);
  queue.writeBuffer(bBuf, 0, bCPU);

  // Uniform buffer: 16 bytes (vec4<u32>)
  const paramsBuf = pool.acquire(
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    "params",
  );
  const paramsCPU = new Uint32Array([n, 0, 0, 0]);
  queue.writeBuffer(paramsBuf, 0, paramsCPU);

  // Readback buffer (not pooled, because mapping lifecycle is special)
  const readback = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "wispdb-test:readback",
  });

  const workgroups = Math.ceil(n / 256);

  // 6) Soak loop
  const chunkTimes: number[] = [];
  const totalStart = performance.now();

  // The pool reuse test: we repeatedly acquire/release the output buffer per chunk.
  // After chunk 1, it should mostly reuse the same buffer (no churn).
  let lastOut: GPUBuffer | null = null;

  for (let done = 0; done < iterations; done += chunkSize) {
    const thisChunk = Math.min(chunkSize, iterations - done);

    // Acquire output for this chunk (reuse should kick in after first chunk)
    const outBuf = pool.acquire(
      bytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      "out",
    );

    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
      label: "wispdb-test:bg",
    });

    const t0 = performance.now();

    for (let i = 0; i < thisChunk; i++) {
      await scheduler.submit((encoder) => {
        const pass = encoder.beginComputePass({ label: "wispdb-test:pass" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
      }, "soak-dispatch");
    }

    await scheduler.flush();
    const t1 = performance.now();
    chunkTimes.push(t1 - t0);

    // Keep the last output buffer for correctness check; release others.
    if (lastOut) {
      pool.release(lastOut, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    }
    lastOut = outBuf;

    log(`Chunk ${(done / chunkSize) + 1}: ${thisChunk} dispatches in ${(t1 - t0).toFixed(1)} ms`);
  }

  const totalEnd = performance.now();

  if (!lastOut) throw new Error("Internal test bug: lastOut missing.");

  // 7) Readback + correctness check
  {
    const encoder = device.createCommandEncoder({ label: "wispdb-test:copy" });
    encoder.copyBufferToBuffer(lastOut, 0, readback, 0, bytes);
    queue.submit([encoder.finish()]);
    await queue.onSubmittedWorkDone();

    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const outCPU = new Float32Array(mapped.slice(0)); // copy out
    readback.unmap();

    // Validate a few samples
    const sampleIdx = [0, 1, 2, 123, 4096, n - 1, (n / 2) | 0, (n / 3) | 0];
    let ok = true;
    for (const idx of sampleIdx) {
      const expected = aCPU[idx] + bCPU[idx];
      const got = outCPU[idx];
      if (Math.abs(got - expected) > 1e-5) {
        ok = false;
        log(`❌ Mismatch at ${idx}: got ${got}, expected ${expected}`);
        break;
      }
    }
    if (ok) log("✅ Correctness check passed (sampled indices).");
  }

  // 8) Stats + stutter score
  const poolStats = pool.getStats();
  const sorted = [...chunkTimes].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length * 0.5)] ?? 0;
  const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 0;
  const worst = sorted[sorted.length - 1] ?? 0;

  const heapEnd =
    (performance as any).memory?.usedJSHeapSize ?? null;

  // Cleanup (optional but polite)
  readback.destroy();
  pool.release(aBuf, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  pool.release(bBuf, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  pool.release(paramsBuf, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  pool.release(lastOut, bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Destroy pooled buffers so you can confirm nothing grows between runs
  pool.destroyAll();
  cache.clear();
  dm.destroy();

  const result = {
    n,
    iterations,
    chunkSize,
    totalMs: totalEnd - totalStart,
    chunks: chunkTimes.length,
    chunkMedianMs: median,
    chunkP95Ms: p95,
    chunkWorstMs: worst,
    stutterRatioWorstOverMedian: median > 0 ? worst / median : Infinity,
    poolStats,
    heapStart,
    heapEnd,
    heapDelta: heapStart != null && heapEnd != null ? heapEnd - heapStart : null,
    pipelineCacheSizeAfter: cache.size(),
  };

  return result;
}
