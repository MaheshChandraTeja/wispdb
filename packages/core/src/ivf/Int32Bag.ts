export class Int32Bag {
  buf: Int32Array;
  len: number;

  constructor(cap = 1024) {
    this.buf = new Int32Array(cap);
    this.len = 0;
  }

  push(x: number): number {
    if (this.len >= this.buf.length) {
      const n = new Int32Array(this.buf.length * 2);
      n.set(this.buf, 0);
      this.buf = n;
    }
    this.buf[this.len] = x;
    return this.len++;
  }

  swapRemoveAt(i: number): number {
    const lastIdx = this.len - 1;
    const last = this.buf[lastIdx];
    this.buf[i] = last;
    this.len--;
    return last; // moved into i
  }

  toArray(): Int32Array {
    return this.buf.slice(0, this.len);
  }
}
