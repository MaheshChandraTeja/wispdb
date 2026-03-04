declare module "*.wgsl?raw" {
	const code: string;
	export default code;
}

declare module "*.wgsl" {
	const code: string;
	export default code;
}

// Minimal WebGPU typings for DTS builds (avoid full dependency on @webgpu/types).
type GPUAdapter = any;
type GPUDevice = any;
type GPUQueue = any;
type GPUBuffer = any;
type GPUCommandEncoder = any;
type GPUComputePipeline = any;
type GPUBindGroupLayout = any;
type GPUBindGroup = any;
type GPUShaderModule = any;
type GPUPipelineLayout = any;
type GPUComputePassEncoder = any;
type GPUDeviceLostInfo = any;

type GPUBufferUsageFlags = number;
type GPUFeatureName = string;
interface GPUSupportedLimits {
	[key: string]: number;
}

interface GPUUncapturedErrorEvent extends Event {
	error: unknown;
}

declare const GPUBufferUsage: {
	STORAGE: number;
	COPY_DST: number;
	COPY_SRC: number;
	UNIFORM: number;
	MAP_READ: number;
};

declare const GPUMapMode: {
	READ: number;
	WRITE: number;
};

interface Navigator {
	gpu?: any;
}
