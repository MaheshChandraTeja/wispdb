export type Metric = "dot" | "cosine" | "l2";

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s = Math.fround(s + Math.fround(a[i] * b[i]));
  return s;
}

export function l2sq(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.fround(a[i] - b[i]);
    s = Math.fround(s + Math.fround(d * d));
  }
  return s;
}

export function norm(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s = Math.fround(s + Math.fround(a[i] * a[i]));
  return Math.sqrt(s);
}

export function cosine(a: Float32Array, b: Float32Array): number {
  const d = dot(a, b);
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return -Infinity;
  return d / (na * nb);
}
