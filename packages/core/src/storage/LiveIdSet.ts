export class LiveIdSet {
  private ids = new Int32Array(1024);
  private len = 0;
  private pos = new Int32Array(0); // internalId -> index in ids, -1 if dead

  ensurePosCapacity(maxIdExclusive: number) {
    if (this.pos.length >= maxIdExclusive) return;
    const p = new Int32Array(maxIdExclusive);
    p.fill(-1);
    p.set(this.pos, 0);
    this.pos = p;
  }

  add(id: number) {
    if (id < 0) return;
    if (id >= this.pos.length) this.ensurePosCapacity(id + 1);
    if (this.pos[id] !== -1) return; // already live

    if (this.len >= this.ids.length) {
      const n = new Int32Array(this.ids.length * 2);
      n.set(this.ids, 0);
      this.ids = n;
    }
    this.pos[id] = this.len;
    this.ids[this.len++] = id;
  }

  remove(id: number) {
    if (id < 0 || id >= this.pos.length) return;
    const i = this.pos[id];
    if (i === -1) return;

    const last = this.ids[this.len - 1];
    this.ids[i] = last;
    this.pos[last] = i;

    this.len--;
    this.pos[id] = -1;
  }

  has(id: number): boolean {
    return id >= 0 && id < this.pos.length && this.pos[id] !== -1;
  }

  snapshot(): Int32Array {
    return this.ids.slice(0, this.len);
  }

  *iterateBatches(batchRows: number): Generator<Int32Array> {
    for (let i = 0; i < this.len; i += batchRows) {
      yield this.ids.slice(i, Math.min(i + batchRows, this.len));
    }
  }

  size(): number {
    return this.len;
  }
}
