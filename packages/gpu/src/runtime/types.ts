// packages/gpu/src/runtime/types.ts
export type PowerPreference = "low-power" | "high-performance";

export interface DeviceManagerOptions {
	powerPreference?: PowerPreference;
	forceFallbackAdapter?: boolean;
	requiredFeatures?: GPUFeatureName[];
	requiredLimits?: Partial<GPUSupportedLimits>;
	labelPrefix?: string;
}

export interface BufferPoolOptions {
	maxBytes?: number; // hard cap before we trim aggressively
	bucketAlignment?: number; // size rounding, default 256
}

export interface BufferPoolStats {
	liveBuffers: number;
	liveBytes: number;
	freeBuffers: number;
	freeBytes: number;
	allocations: number;
	reuses: number;
	destroys: number;
}

export interface SchedulerOptions {
	maxInFlight?: number; // throttle via onSubmittedWorkDone
	labelPrefix?: string;
}
