import { describe, it, expect } from "vitest";

type SnapshotRecord = {
  snapshotId: string;
  status: "writing" | "complete";
  createdAt: number;
  snapshotSeq: number;
};

type JournalEntry = { seq: number; op: "upsert" | "delete"; externalId: string; value?: number };

function pickLatestCompleteSnapshot(snaps: SnapshotRecord[]) {
  const complete = snaps.filter((s) => s.status === "complete");
  if (complete.length === 0) return null;
  complete.sort((a, b) => b.createdAt - a.createdAt);
  return complete[0];
}

function applyJournalAfter(
  base: Map<string, number>,
  journal: JournalEntry[],
  afterSeq: number,
) {
  const entries = journal.slice().sort((a, b) => a.seq - b.seq).filter((e) => e.seq > afterSeq);
  for (const e of entries) {
    if (e.op === "upsert") base.set(e.externalId, e.value ?? 0);
    else base.delete(e.externalId);
  }
  return base;
}

describe("PersistenceV1 logic (snapshot + journal)", () => {
  it("ignores incomplete snapshot and replays from last complete", () => {
    const snaps: SnapshotRecord[] = [
      { snapshotId: "s1", status: "complete", createdAt: 1000, snapshotSeq: 5 },
      { snapshotId: "s2", status: "writing", createdAt: 2000, snapshotSeq: 9 }, // incomplete
    ];

    const journal: JournalEntry[] = [
      { seq: 1, op: "upsert", externalId: "a", value: 1 },
      { seq: 3, op: "upsert", externalId: "b", value: 2 },
      { seq: 6, op: "delete", externalId: "a" },
      { seq: 7, op: "upsert", externalId: "c", value: 3 },
    ];

    // snapshot s1 already contains seq<=5
    const snapshotState = new Map<string, number>();
    snapshotState.set("a", 1);
    snapshotState.set("b", 2);

    const snap = pickLatestCompleteSnapshot(snaps);
    expect(snap?.snapshotId).toBe("s1");

    const restored = applyJournalAfter(snapshotState, journal, snap!.snapshotSeq);
    expect(restored.has("a")).toBe(false);
    expect(restored.get("b")).toBe(2);
    expect(restored.get("c")).toBe(3);
  });

  it("when no snapshot exists, replays from seq 0", () => {
    const snaps: SnapshotRecord[] = [
      { snapshotId: "s1", status: "writing", createdAt: 1000, snapshotSeq: 5 },
    ];
    const journal: JournalEntry[] = [
      { seq: 1, op: "upsert", externalId: "a", value: 1 },
      { seq: 2, op: "upsert", externalId: "b", value: 2 },
      { seq: 3, op: "delete", externalId: "a" },
    ];

    const snap = pickLatestCompleteSnapshot(snaps);
    expect(snap).toBe(null);

    const restored = applyJournalAfter(new Map(), journal, 0);
    expect(restored.has("a")).toBe(false);
    expect(restored.get("b")).toBe(2);
  });
});
