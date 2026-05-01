import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  noExternal: ["@wispdb/gpu"],
  sourcemap: true,
  clean: true,
  esbuildOptions(options) {
    options.loader = {
      ...options.loader,
      ".wgsl": "text",
    };
  },
});
