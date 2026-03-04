import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  resolve: {
    alias: {
      "@wispdb/gpu": path.resolve(
        __dirname,
        "../../packages/gpu/src/index.ts",
      ),
      "@wispdb/core": path.resolve(
        __dirname,
        "../../packages/core/src/index.ts",
      ),
      "@wispdb/utils": path.resolve(
        __dirname,
        "../../packages/utils/src/index.ts",
      ),
    },
  },
});
