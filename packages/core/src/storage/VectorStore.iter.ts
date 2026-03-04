export interface LiveBatch {
  ids: Int32Array;        // internal ids for each row in batch
  vectors: Float32Array;  // packed [row][dim], length = ids.length * dim
}

*iterateLiveBatches(batchRows = 16384): Generator<LiveBatch> {
  if (!Number.isInteger(batchRows) || batchRows <= 0) throw new Error("batchRows must be > 0");

  const idsTmp = new Int32Array(batchRows);
  const vecTmp = new Float32Array(batchRows * this.dim);

  let count = 0;
  const maxId = this.nextId;

  for (let id = 0; id < maxId; id++) {
    if (this.getTombstone(id) === 1) continue;

    idsTmp[count] = id;
    const view = this.getViewByInternalId(id)!; // live, so not null
    vecTmp.set(view, count * this.dim);
    count++;

    if (count === batchRows) {
      yield {
        ids: idsTmp.slice(0, count),
        vectors: vecTmp.slice(0, count * this.dim),
      };
      count = 0;
    }
  }

  if (count > 0) {
    yield {
      ids: idsTmp.slice(0, count),
      vectors: vecTmp.slice(0, count * this.dim),
    };
  }
}
