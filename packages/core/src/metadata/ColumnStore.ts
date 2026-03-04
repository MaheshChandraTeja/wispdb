import type { HotField, MetadataSchema } from "./types";

const BOOL_MISSING = 2;           // 0=false, 1=true, 2=missing
const ENUM_MISSING = 0xffff;      // u16 sentinel
const NUM_MISSING = NaN;          // NaN sentinel

function isPow2(x: number) { return (x & (x - 1)) === 0; }

export class ColumnStore {
  private readonly chunkRows: number;
  private readonly CHUNK_SHIFT: number;
  private readonly CHUNK_MASK: number;

  private numCols = new Map<string, Float32Array[]>();
  private boolCols = new Map<string, Uint8Array[]>();
  private enumCols = new Map<string, Uint16Array[]>();

  private enumDict = new Map<string, Map<string, number>>(); // field -> value->code

  constructor(private schema: MetadataSchema, chunkRows = 4096) {
    if (!isPow2(chunkRows)) throw new Error("chunkRows must be power of 2");
    this.chunkRows = chunkRows;
    this.CHUNK_SHIFT = Math.log2(chunkRows) | 0;
    this.CHUNK_MASK = chunkRows - 1;

    for (const [field, spec] of Object.entries(schema)) {
      if (spec.type === "number") this.numCols.set(field, []);
      if (spec.type === "boolean") this.boolCols.set(field, []);
      if (spec.type === "enum") {
        this.enumCols.set(field, []);
        const dict = new Map<string, number>();
        spec.values.forEach((v, i) => dict.set(v, i));
        this.enumDict.set(field, dict);
      }
    }
  }

  ensureCapacityForId(id: number) {
    const needChunk = id >> this.CHUNK_SHIFT;

    for (const [field] of this.numCols) {
      const arr = this.numCols.get(field)!;
      while (arr.length <= needChunk) {
        const c = new Float32Array(this.chunkRows);
        c.fill(NUM_MISSING);
        arr.push(c);
      }
    }

    for (const [field] of this.boolCols) {
      const arr = this.boolCols.get(field)!;
      while (arr.length <= needChunk) {
        const c = new Uint8Array(this.chunkRows);
        c.fill(BOOL_MISSING);
        arr.push(c);
      }
    }

    for (const [field] of this.enumCols) {
      const arr = this.enumCols.get(field)!;
      while (arr.length <= needChunk) {
        const c = new Uint16Array(this.chunkRows);
        c.fill(ENUM_MISSING);
        arr.push(c);
      }
    }
  }

  // Write metadata into hot columns if present
  setHot(internalId: number, meta: any) {
    this.ensureCapacityForId(internalId);
    const chunk = internalId >> this.CHUNK_SHIFT;
    const row = internalId & this.CHUNK_MASK;

    for (const [field, spec] of Object.entries(this.schema)) {
      const v = meta?.[field];

      if (spec.type === "number") {
        const col = this.numCols.get(field)!;
        col[chunk][row] = (typeof v === "number" && Number.isFinite(v)) ? v : NUM_MISSING;
      } else if (spec.type === "boolean") {
        const col = this.boolCols.get(field)!;
        col[chunk][row] = (typeof v === "boolean") ? (v ? 1 : 0) : BOOL_MISSING;
      } else if (spec.type === "enum") {
        const col = this.enumCols.get(field)!;
        const dict = this.enumDict.get(field)!;
        const code = (typeof v === "string" && dict.has(v)) ? dict.get(v)! : ENUM_MISSING;
        col[chunk][row] = code;
      }
    }
  }

  // Clear hot fields for deleted ids (optional but nice)
  clearHot(internalId: number) {
    const chunk = internalId >> this.CHUNK_SHIFT;
    const row = internalId & this.CHUNK_MASK;

    for (const [field] of this.numCols) this.numCols.get(field)![chunk][row] = NUM_MISSING;
    for (const [field] of this.boolCols) this.boolCols.get(field)![chunk][row] = BOOL_MISSING;
    for (const [field] of this.enumCols) this.enumCols.get(field)![chunk][row] = ENUM_MISSING;
  }

  // Reads (fast)
  getNumber(field: string, id: number): number {
    const col = this.numCols.get(field);
    if (!col) return NaN;
    const chunk = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    return col[chunk]?.[row] ?? NaN;
  }

  getBool(field: string, id: number): 0 | 1 | 2 {
    const col = this.boolCols.get(field);
    if (!col) return 2;
    const chunk = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    return (col[chunk]?.[row] ?? 2) as 0 | 1 | 2;
  }

  getEnumCode(field: string, id: number): number {
    const col = this.enumCols.get(field);
    if (!col) return ENUM_MISSING;
    const chunk = id >> this.CHUNK_SHIFT;
    const row = id & this.CHUNK_MASK;
    return col[chunk]?.[row] ?? ENUM_MISSING;
  }

  enumValueToCode(field: string, value: string): number {
    const dict = this.enumDict.get(field);
    if (!dict) return ENUM_MISSING;
    return dict.get(value) ?? ENUM_MISSING;
  }

  hasField(field: string): boolean {
    return this.schema[field] !== undefined;
  }

  getFieldSpec(field: string): HotField | undefined {
    return this.schema[field];
  }

  trimToMaxId(maxIdExclusive: number) {
    const newChunkCount = maxIdExclusive <= 0 ? 0 : ((maxIdExclusive - 1) >> this.CHUNK_SHIFT) + 1;
    for (const [field] of this.numCols) {
      const arr = this.numCols.get(field)!;
      if (arr.length > newChunkCount) arr.length = newChunkCount;
    }
    for (const [field] of this.boolCols) {
      const arr = this.boolCols.get(field)!;
      if (arr.length > newChunkCount) arr.length = newChunkCount;
    }
    for (const [field] of this.enumCols) {
      const arr = this.enumCols.get(field)!;
      if (arr.length > newChunkCount) arr.length = newChunkCount;
    }
  }
}
