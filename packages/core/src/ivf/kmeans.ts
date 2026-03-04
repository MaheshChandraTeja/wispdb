export interface KMeansOptions {
  nlist: number;
  iters?: number;
  seed?: number;
  sampleSize?: number;
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function l2sq(a: Float32Array, aOff: number, b: Float32Array, bOff: number, dim: number) {
  let s = 0;
  for (let j = 0; j < dim; j++) {
    const d = a[aOff + j] - b[bOff + j];
    s += d * d;
  }
  return s;
}

export function trainKMeans(
  data: Float32Array, // packed [n][dim]
  n: number,
  dim: number,
  opts: KMeansOptions,
): Float32Array {
  const nlist = opts.nlist;
  const iters = opts.iters ?? 20;
  const seed = opts.seed ?? 1234;
  const rand = rng(seed);

  if (nlist <= 0 || nlist > n) throw new Error(`nlist invalid: ${nlist} (n=${n})`);

  // ----- kmeans++ init (deterministic) -----
  const centroids = new Float32Array(nlist * dim);

  const first = Math.floor(rand() * n);
  centroids.set(data.subarray(first * dim, first * dim + dim), 0);

  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) dist[i] = l2sq(data, i * dim, centroids, 0, dim);

  for (let c = 1; c < nlist; c++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += dist[i];
    let r = rand() * sum;

    let pick = 0;
    for (let i = 0; i < n; i++) {
      r -= dist[i];
      if (r <= 0) { pick = i; break; }
    }

    centroids.set(data.subarray(pick * dim, pick * dim + dim), c * dim);

    // update distances to nearest centroid
    for (let i = 0; i < n; i++) {
      const d = l2sq(data, i * dim, centroids, c * dim, dim);
      if (d < dist[i]) dist[i] = d;
    }
  }

  // ----- Lloyd iterations -----
  const assign = new Int32Array(n);
  const sums = new Float32Array(nlist * dim);
  const counts = new Int32Array(nlist);

  for (let it = 0; it < iters; it++) {
    sums.fill(0);
    counts.fill(0);

    // assign
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      const off = i * dim;

      for (let c = 0; c < nlist; c++) {
        const d = l2sq(data, off, centroids, c * dim, dim);
        if (d < bestD) { bestD = d; best = c; }
      }

      assign[i] = best;
      counts[best]++;

      const cOff = best * dim;
      for (let j = 0; j < dim; j++) sums[cOff + j] += data[off + j];
    }

    // recompute
    for (let c = 0; c < nlist; c++) {
      const cnt = counts[c];
      const cOff = c * dim;

      if (cnt === 0) {
        // re-seed empty centroid deterministically
        const pick = Math.floor(rand() * n);
        centroids.set(data.subarray(pick * dim, pick * dim + dim), cOff);
        continue;
      }

      const inv = 1.0 / cnt;
      for (let j = 0; j < dim; j++) centroids[cOff + j] = sums[cOff + j] * inv;
    }
  }

  return centroids;
}
