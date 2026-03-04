// packages/gpu/src/runtime/errors.ts
export class WebGPUUnavailableError extends Error {
	name = "WebGPUUnavailableError";
	constructor(msg = "WebGPU is not available in this environment.") {
		super(msg);
	}
}

export class DeviceRequestError extends Error {
	name = "DeviceRequestError";
	constructor(msg: string) {
		super(msg);
	}
}

export class PipelineCompileError extends Error {
	name = "PipelineCompileError";
	constructor(msg: string) {
		super(msg);
	}
}
