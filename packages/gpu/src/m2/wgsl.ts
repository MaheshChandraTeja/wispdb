export const WGSL = {
	normalizeBatch: /* wgsl */ `
struct Params { n: u32, dim: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read>  inputVecs  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputVecs : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let dim = params.dim;
  let base = i * dim;

  var sum: f32 = 0.0;
  var j: u32 = 0u;

  // unrolled by 4 where possible
  for (; j + 3u < dim; j += 4u) {
    let x0 = inputVecs[base + j + 0u];
    let x1 = inputVecs[base + j + 1u];
    let x2 = inputVecs[base + j + 2u];
    let x3 = inputVecs[base + j + 3u];
    sum += x0*x0 + x1*x1 + x2*x2 + x3*x3;
  }
  for (; j < dim; j += 1u) {
    let x = inputVecs[base + j];
    sum += x*x;
  }

  // avoid divide-by-zero
  let inv = inverseSqrt(max(sum, 1e-12));

  j = 0u;
  for (; j + 3u < dim; j += 4u) {
    outputVecs[base + j + 0u] = inputVecs[base + j + 0u] * inv;
    outputVecs[base + j + 1u] = inputVecs[base + j + 1u] * inv;
    outputVecs[base + j + 2u] = inputVecs[base + j + 2u] * inv;
    outputVecs[base + j + 3u] = inputVecs[base + j + 3u] * inv;
  }
  for (; j < dim; j += 1u) {
    outputVecs[base + j] = inputVecs[base + j] * inv;
  }
}
`,

	scoreDot: /* wgsl */ `
struct Params { n: u32, dim: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> vectors : array<f32>;
@group(0) @binding(1) var<storage, read> query   : array<f32>;
@group(0) @binding(2) var<storage, read_write> scores : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let dim = params.dim;
  let base = i * dim;

  var sum: f32 = 0.0;
  var j: u32 = 0u;

  for (; j + 3u < dim; j += 4u) {
    let v0 = vectors[base + j + 0u];
    let v1 = vectors[base + j + 1u];
    let v2 = vectors[base + j + 2u];
    let v3 = vectors[base + j + 3u];

    let q0 = query[j + 0u];
    let q1 = query[j + 1u];
    let q2 = query[j + 2u];
    let q3 = query[j + 3u];

    sum += v0*q0 + v1*q1 + v2*q2 + v3*q3;
  }

  for (; j < dim; j += 1u) {
    sum += vectors[base + j] * query[j];
  }

  scores[i] = sum;
}
`,

	scoreL2Neg: /* wgsl */ `
struct Params { n: u32, dim: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> vectors : array<f32>;
@group(0) @binding(1) var<storage, read> query   : array<f32>;
@group(0) @binding(2) var<storage, read_write> scores : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }

  let dim = params.dim;
  let base = i * dim;

  var sum: f32 = 0.0;
  var j: u32 = 0u;

  for (; j + 3u < dim; j += 4u) {
    let v0 = vectors[base + j + 0u];
    let v1 = vectors[base + j + 1u];
    let v2 = vectors[base + j + 2u];
    let v3 = vectors[base + j + 3u];

    let q0 = query[j + 0u];
    let q1 = query[j + 1u];
    let q2 = query[j + 2u];
    let q3 = query[j + 3u];

    let d0 = v0 - q0;
    let d1 = v1 - q1;
    let d2 = v2 - q2;
    let d3 = v3 - q3;

    sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
  }

  for (; j < dim; j += 1u) {
    let d = vectors[base + j] - query[j];
    sum += d*d;
  }

  // NEGATIVE squared L2 so "higher is better" for TopK
  scores[i] = -sum;
}
`,

	// TopK stage 1: scores -> per-block TopK pairs
	// scores -> outPairsPacked (vec2<u32>(scoreBits, idx))
	blockTopKScores: /* wgsl */ `
struct Params { n: u32, dim: u32, k: u32, _pad: u32 }
struct Pair { score: f32, idx: u32 }

@group(0) @binding(0) var<storage, read> scores : array<f32>;
@group(0) @binding(1) var<storage, read_write> outPairs : array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> sharedPairs : array<Pair, 256>;

fn better(a: Pair, b: Pair) -> bool {
  if (a.score > b.score) { return true; }
  if (a.score < b.score) { return false; }
  return a.idx < b.idx; // deterministic tie-break
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let lid = lid3.x;
  let i = gid.x;

  var p: Pair;
  if (i < params.n) {
    p.score = scores[i];
    p.idx = i;
  } else {
    p.score = -3.402823e38;
    p.idx = 0xffffffffu;
  }

  sharedPairs[lid] = p;
  workgroupBarrier();

  var k: u32 = 2u;
  loop {
    if (k > 256u) { break; }
    var j: u32 = k / 2u;
    loop {
      if (j == 0u) { break; }
      let ixj = lid ^ j;
      if (ixj > lid) {
        let dir = ((lid & k) == 0u);
        let a = sharedPairs[lid];
        let b = sharedPairs[ixj];

        var swap = false;
        if (dir) { swap = better(b, a); }
        else     { swap = better(a, b); }

        if (swap) {
          sharedPairs[lid] = b;
          sharedPairs[ixj] = a;
        }
      }
      workgroupBarrier();
      j = j / 2u;
    }
    k = k * 2u;
  }

  if (lid < params.k) {
    let outBase = wgid.x * params.k;
    outPairs[outBase + lid] = vec2<u32>(bitcast<u32>(sharedPairs[lid].score), sharedPairs[lid].idx);
  }
}
`,

	// inPairsPacked -> outPairsPacked
	blockTopKPairs: /* wgsl */ `
struct Params { n: u32, dim: u32, k: u32, _pad: u32 }
struct Pair { score: f32, idx: u32 }

@group(0) @binding(0) var<storage, read> inPairs : array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> outPairs : array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> sharedPairs : array<Pair, 256>;

fn better(a: Pair, b: Pair) -> bool {
  if (a.score > b.score) { return true; }
  if (a.score < b.score) { return false; }
  return a.idx < b.idx;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let lid = lid3.x;
  let i = gid.x;

  var p: Pair;
  if (i < params.n) {
    let v = inPairs[i];
    p.score = bitcast<f32>(v.x);
    p.idx = v.y;
  } else {
    p.score = -3.402823e38;
    p.idx = 0xffffffffu;
  }

  sharedPairs[lid] = p;
  workgroupBarrier();

  var k: u32 = 2u;
  loop {
    if (k > 256u) { break; }
    var j: u32 = k / 2u;
    loop {
      if (j == 0u) { break; }
      let ixj = lid ^ j;
      if (ixj > lid) {
        let dir = ((lid & k) == 0u);
        let a = sharedPairs[lid];
        let b = sharedPairs[ixj];

        var swap = false;
        if (dir) { swap = better(b, a); }
        else     { swap = better(a, b); }

        if (swap) {
          sharedPairs[lid] = b;
          sharedPairs[ixj] = a;
        }
      }
      workgroupBarrier();
      j = j / 2u;
    }
    k = k * 2u;
  }

  if (lid < params.k) {
    let outBase = wgid.x * params.k;
    outPairs[outBase + lid] = vec2<u32>(bitcast<u32>(sharedPairs[lid].score), sharedPairs[lid].idx);
  }
}
`,
} as const;
