import type { MetadataSchema, WhereClause, WhereValue } from "./types";
import { ColumnStore } from "./ColumnStore";

export class MetadataStoreV1 {
  private row = new Map<number, any>();
  private cols: ColumnStore;

  constructor(schema: MetadataSchema = {}, chunkRows = 4096) {
    this.cols = new ColumnStore(schema, chunkRows);
  }

  get(internalId: number): any | null {
    return this.row.get(internalId) ?? null;
  }

  upsert(internalId: number, meta: any) {
    const m = meta ?? {};
    this.row.set(internalId, m);
    this.cols.setHot(internalId, m);
  }

  delete(internalId: number) {
    this.row.delete(internalId);
    // keep columns tidy (optional)
    this.cols.ensureCapacityForId(internalId);
    this.cols.clearHot(internalId);
  }

  exportRows(): Array<[number, any]> {
    // deterministic by internalId ascending
    const out: Array<[number, any]> = [];
    for (const [k, v] of this.row) out.push([k, v]);
    out.sort((a, b) => a[0] - b[0]);
    return out;
  }

  compactToMaxId(maxIdExclusive: number) {
    for (const k of this.row.keys()) {
      if (k >= maxIdExclusive) this.row.delete(k);
    }
    this.cols.trimToMaxId(maxIdExclusive);
  }

  /** Deterministic prefilter: returns internal IDs in ascending order. */
  prefilter(where: WhereClause, maxIdExclusive: number, isLive: (id: number) => boolean): Int32Array {
    const out: number[] = [];

    for (let id = 0; id < maxIdExclusive; id++) {
      if (!isLive(id)) continue;
      if (this.matches(where, id)) out.push(id);
    }

    return Int32Array.from(out);
  }

  private matches(where: WhereClause, internalId: number): boolean {
    for (const [field, cond] of Object.entries(where)) {
      if (!this.matchField(field, cond, internalId)) return false;
    }
    return true;
  }

  private matchField(field: string, cond: WhereValue, internalId: number): boolean {
    const spec = this.cols.getFieldSpec(field);

    // Hot path: column store
    if (spec) {
      if (spec.type === "number") {
        const v = this.cols.getNumber(field, internalId);
        return matchNumber(v, cond);
      }
      if (spec.type === "boolean") {
        const v = this.cols.getBool(field, internalId);
        return matchBool(v, cond);
      }
      if (spec.type === "enum") {
        const code = this.cols.getEnumCode(field, internalId);
        return matchEnum(code, cond, (s) => this.cols.enumValueToCode(field, s));
      }
    }

    // Cold path: row json
    const meta = this.row.get(internalId);
    const v = meta?.[field];
    return matchAny(v, cond);
  }
}

// ---- predicate helpers ----

function isRangeObj(x: any): x is { gt?: number; gte?: number; lt?: number; lte?: number } {
  return x && typeof x === "object" && ("gt" in x || "gte" in x || "lt" in x || "lte" in x);
}
function isInObj(x: any): x is { in: any[] } {
  return x && typeof x === "object" && Array.isArray(x.in);
}

function matchNumber(v: number, cond: any): boolean {
  if (!Number.isFinite(v)) return false;
  if (typeof cond === "number") return v === cond;
  if (isRangeObj(cond)) {
    if (typeof cond.gt === "number" && !(v > cond.gt)) return false;
    if (typeof cond.gte === "number" && !(v >= cond.gte)) return false;
    if (typeof cond.lt === "number" && !(v < cond.lt)) return false;
    if (typeof cond.lte === "number" && !(v <= cond.lte)) return false;
    return true;
  }
  if (isInObj(cond)) return cond.in.includes(v);
  return false;
}

function matchBool(v: 0 | 1 | 2, cond: any): boolean {
  if (typeof cond === "boolean") return v !== 2 && (v === 1) === cond;
  if (isInObj(cond)) return cond.in.includes((v === 1));
  return false;
}

function matchEnum(code: number, cond: any, toCode: (s: string) => number): boolean {
  if (code === 0xffff) return false;
  if (typeof cond === "string") return code === toCode(cond);
  if (isInObj(cond)) {
    for (const x of cond.in) {
      if (typeof x === "string" && code === toCode(x)) return true;
    }
    return false;
  }
  return false;
}

function matchAny(v: any, cond: any): boolean {
  if (isInObj(cond)) return cond.in.includes(v);
  if (isRangeObj(cond)) return (typeof v === "number") && matchNumber(v, cond);
  return v === cond;
}
