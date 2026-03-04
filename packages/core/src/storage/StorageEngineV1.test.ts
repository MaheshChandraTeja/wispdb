import { describe, it, expect } from "vitest";
import { StorageEngineV1 } from "./StorageEngineV1";

function makeVec(dim: number, seed: number) {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.fround((seed * 131 + i * 17) % 1000 / 1000);
  return v;
}

describe("StorageEngineV1", () => {
  it("upsert/get works", () => {
    const dim = 8;
    const db = new StorageEngineV1(dim);

    db.upsert("a", makeVec(dim, 1));
    db.upsert("b", makeVec(dim, 2));

    const a = db.get("a")!;
    const b = db.get("b")!;
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(Array.from(a)).toEqual(Array.from(makeVec(dim, 1)));
    expect(Array.from(b)).toEqual(Array.from(makeVec(dim, 2)));

    // overwrite
    db.upsert("a", makeVec(dim, 99));
    const a2 = db.get("a")!;
    expect(Array.from(a2)).toEqual(Array.from(makeVec(dim, 99)));
  });

  it("reuses internal ids after delete (free list)", () => {
    const dim = 4;
    const db = new StorageEngineV1(dim, 8);

    const idA = db.upsert("a", new Float32Array([1, 1, 1, 1]));
    const idB = db.upsert("b", new Float32Array([2, 2, 2, 2]));

    expect(db.delete("a")).toBe(true);

    const idC = db.upsert("c", new Float32Array([3, 3, 3, 3]));

    // c should reuse a's slot (LIFO free list) unless b was deleted too
    expect(idC).toBe(idA);
    expect(db.get("a")).toBeNull();
    expect(Array.from(db.get("c")!)).toEqual([3, 3, 3, 3]);

    // b still intact
    expect(idB).not.toBe(idC);
  });

  it("delete uses tombstone and get returns null", () => {
    const db = new StorageEngineV1(4);
    db.upsert("x", new Float32Array([1, 2, 3, 4]));
    expect(db.get("x")).not.toBeNull();
    expect(db.delete("x")).toBe(true);
    expect(db.get("x")).toBeNull();
    expect(db.delete("x")).toBe(false);
  });

  it("handles 100k vectors without imploding", () => {
    const dim = 32;
    const db = new StorageEngineV1(dim, 4096);

    const N = 100_000;
    for (let i = 0; i < N; i++) {
      db.upsert(`id_${i}`, makeVec(dim, i));
    }

    const s = db.stats();
    expect(s.idCount).toBe(N);
    expect(s.vectors.liveCount).toBe(N);

    // spot checks
    for (const i of [0, 1, 2, 999, 50_000, 99_999]) {
      const got = db.get(`id_${i}`)!;
      expect(Array.from(got)).toEqual(Array.from(makeVec(dim, i)));
    }
  });
});

