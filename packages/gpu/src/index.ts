// packages/gpu/src/index.ts

export * from "./kernels/VectorAddKernel";
export type {
	ExactSearchResult,
	Metric,
	TopKMode,
} from "./m2/ExactSearchGPU";
export {
	ExactSearchGPU,
	topkFromScoresCPU,
} from "./m2/ExactSearchGPU";
export type {
	Metric as ChunkScannerMetric,
	Pair,
} from "./m4/ChunkScannerGPU";
export { ChunkScannerGPU } from "./m4/ChunkScannerGPU";
export * from "./runtime/BufferPool";
export * from "./runtime/CommandScheduler";
export * from "./runtime/DeviceManager";
export * from "./runtime/errors";
export * from "./runtime/PipelineCache";
export * from "./runtime/types";

export function isWebGPUAvailable(): boolean {
	return (
		typeof navigator !== "undefined" &&
		"gpu" in navigator &&
		!!(navigator as any).gpu
	);
}
