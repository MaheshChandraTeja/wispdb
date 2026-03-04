import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

test("browser bench", async ({ page }) => {
  await page.goto("/");

  const ci = !!process.env.CI || process.argv.includes("--ci");
  const opts = ci
    ? { N: 5000, queries: 10, batchSize: 500, frameSearches: 100 }
    : {};

  const res = await page.evaluate(async (o) => {
    // @ts-ignore
    return await window.__WISP_BENCH_RUN__?.(o);
  }, opts);

  expect(res).toBeTruthy();
  expect(res.search_ms_p50).toBeGreaterThan(0);
  expect(res.search_ms_p95).toBeGreaterThan(0);

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const outPath = path.resolve(__dirname, "../../../bench-results.json");
  fs.writeFileSync(outPath, JSON.stringify(res, null, 2), "utf8");
  console.log(JSON.stringify(res));
});
