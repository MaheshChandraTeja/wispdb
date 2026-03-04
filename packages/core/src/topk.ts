export interface TopKItem { index: number; score: number; }

export function topk(scores: Float32Array, k: number): TopKItem[] {
  // same heap approach as GPU file; stable by index
  const n = scores.length;
  const kk = Math.min(k, n);
  const heapScore = new Float32Array(kk);
  const heapIdx = new Uint32Array(kk);
  let size = 0;

  function worse(iA: number, sA: number, iB: number, sB: number): boolean {
    if (sA < sB) return true;
    if (sA > sB) return false;
    return iA > iB;
  }

  function siftUp(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      const ci = heapIdx[i], cs = heapScore[i];
      const pi = heapIdx[p], ps = heapScore[p];
      if (worse(ci, cs, pi, ps)) {
        heapIdx[i] = pi; heapScore[i] = ps;
        heapIdx[p] = ci; heapScore[p] = cs;
        i = p;
      } else break;
    }
  }

  function siftDown(i: number) {
    while (true) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < size && worse(heapIdx[l], heapScore[l], heapIdx[m], heapScore[m])) m = l;
      if (r < size && worse(heapIdx[r], heapScore[r], heapIdx[m], heapScore[m])) m = r;
      if (m === i) break;
      const ti = heapIdx[i], ts = heapScore[i];
      heapIdx[i] = heapIdx[m]; heapScore[i] = heapScore[m];
      heapIdx[m] = ti; heapScore[m] = ts;
      i = m;
    }
  }

  for (let i = 0; i < n; i++) {
    const s = scores[i];
    if (size < kk) {
      heapIdx[size] = i;
      heapScore[size] = s;
      siftUp(size);
      size++;
    } else {
      const wi = heapIdx[0], ws = heapScore[0];
      const better = s > ws || (s === ws && i < wi);
      if (better) {
        heapIdx[0] = i;
        heapScore[0] = s;
        siftDown(0);
      }
    }
  }

  const out: TopKItem[] = [];
  for (let i = 0; i < size; i++) out.push({ index: heapIdx[i], score: heapScore[i] });
  out.sort((a, b) => (b.score - a.score) || (a.index - b.index));
  return out;
}
