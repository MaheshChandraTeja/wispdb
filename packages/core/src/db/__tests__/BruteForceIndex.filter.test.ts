import { describe, it, expect } from "vitest";
import { BruteForceIndex } from "../BruteForceIndex";

function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed / 4294967296) * 2 - 1;
  };
}

describe("Milestone 5 filtering", () => {
  it("search(where) works and is deterministic", async () => {
    const dim = 32;
    const db = new BruteForceIndex({
      dim,
      preferGPU: false,
      metadataSchema: {
        year: { type: "number" },
        premium: { type: "boolean" },
        lang: { type: "enum", values: ["en", "fr", "de"] },
      },
    });

    const r = rng(123);
    const N = 5000;

    for (let i = 0; i < N; i++) {
      const v = new Float32Array(dim);
      for (let j = 0; j < dim; j++) v[j] = r() * 0.5;

      db.upsert(`id_${i}`, v, {
        year: 2000 + (i % 25),
        premium: (i % 2) === 0,
        lang: (i % 3) === 0 ? "en" : (i % 3) === 1 ? "fr" : "de",
        junk: "ignored",
      });
    }

    const q = new Float32Array(dim);
    for (let j = 0; j < dim; j++) q[j] = r() * 0.5;

    const where = { premium: true, year: { gte: 2010, lte: 2015 }, lang: { in: ["en", "fr"] } } as const;

    const a = await db.search(q, { k: 10, where });
    const b = await db.search(q, { k: 10, where });

    expect(a.map(x => x.internalId)).toEqual(b.map(x => x.internalId));
    expect(a.map(x => x.score)).toEqual(b.map(x => x.score));
  });
});
