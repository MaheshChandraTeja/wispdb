export type HotField =
  | { type: "number" }   // stored as float32
  | { type: "boolean" }  // stored as u8
  | { type: "enum"; values: readonly string[] }; // stored as u16 codes

export type MetadataSchema = Record<string, HotField>;

export type WhereValue =
  | string
  | number
  | boolean
  | { in: readonly (string | number | boolean)[] }
  | { gt?: number; gte?: number; lt?: number; lte?: number };

export type WhereClause = Record<string, WhereValue>;
