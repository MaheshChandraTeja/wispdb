import fs from "node:fs";

const [resultsPath, gatesPath] = process.argv.slice(2);
if (!resultsPath || !gatesPath) {
  console.error("Usage: node scripts/bench/check_gates.ts <results.json> <gates.json>");
  process.exit(1);
}

const results = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
const gates = JSON.parse(fs.readFileSync(gatesPath, "utf8"));

const mapValue = (key: string): number | null => {
  switch (key) {
    case "search_p95_ms_max":
      return results.search_ms_p95 ?? null;
    case "ingest_p95_ms_max":
      return results.ingest_ms_p95 ?? null;
    case "ivf_recall_at_10_min":
      return results.ivf_recall_at_10 ?? null;
    case "ivf_speedup_min":
      return results.ivf_speedup ?? null;
    case "warm_start_ms_max":
      return results.warm_start_ms ?? null;
    case "frame_p95_ms_max":
      return results.frame_p95_ms ?? null;
    default:
      return null;
  }
};

const failures: string[] = [];

for (const [key, gateValue] of Object.entries(gates)) {
  const v = mapValue(key);
  if (v == null || Number.isNaN(v)) {
    console.warn(`SKIP ${key}: missing in results`);
    continue;
  }
  if (key.endsWith("_max")) {
    if (v > (gateValue as number)) failures.push(`${key}=${v} > ${gateValue}`);
  } else if (key.endsWith("_min")) {
    if (v < (gateValue as number)) failures.push(`${key}=${v} < ${gateValue}`);
  } else {
    console.warn(`SKIP ${key}: unknown gate suffix`);
  }
}

if (failures.length) {
  console.error("Bench gates failed:\n" + failures.join("\n"));
  process.exit(1);
} else {
  console.log("Bench gates passed");
}