import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.cwd();
const gpuStub = pathToFileURL(path.join(root, "packages/bench/src/gpu_stub.ts")).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "@wispdb/gpu") {
    return { url: gpuStub, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}