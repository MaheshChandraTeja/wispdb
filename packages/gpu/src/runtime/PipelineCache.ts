// packages/gpu/src/runtime/PipelineCache.ts
import { PipelineCompileError } from "./errors";

type ComputeKey = string;

function hashKey(parts: string[]): string {
	// Deterministic, cheap. Not crypto. We just want stable cache keys.
	let h = 2166136261;
	for (const p of parts) {
		for (let i = 0; i < p.length; i++) {
			h ^= p.charCodeAt(i);
			h = Math.imul(h, 16777619);
		}
	}
	return (h >>> 0).toString(16);
}

export interface ComputePipelineDesc {
	code: string;
	entryPoint?: string;
	bindGroupLayouts?: GPUBindGroupLayout[]; // optional explicit layout
	label?: string;
}

export class PipelineCache {
	private compute = new Map<ComputeKey, GPUComputePipeline>();

	constructor(
		private device: GPUDevice,
		private labelPrefix = "wispdb",
	) {}

	getComputePipeline(desc: ComputePipelineDesc): GPUComputePipeline {
		const entryPoint = desc.entryPoint ?? "main";
		const layoutSig = (desc.bindGroupLayouts ?? []).length.toString();
		const key = hashKey([entryPoint, layoutSig, desc.code]);

		const cached = this.compute.get(key);
		if (cached) return cached;

		const module = this.device.createShaderModule({
			code: desc.code,
			label: `${this.labelPrefix}:shader:${desc.label ?? key}`,
		});
		const getInfo = (module as any).getCompilationInfo?.bind(module);
		if (getInfo) {
			getInfo()
				.then((info: any) => {
					const messages = info?.messages ?? [];
					for (const m of messages) {
						const type = m.type ?? "info";
						const line = m.lineNum ?? m.line ?? "?";
						const col = m.linePos ?? m.column ?? "?";
						const text = m.message ?? String(m);
						if (type === "error") {
							console.error(
								`[${this.labelPrefix}] WGSL ${type} (${desc.label ?? key}) at ${line}:${col}: ${text}`,
							);
						} else if (type === "warning") {
							console.warn(
								`[${this.labelPrefix}] WGSL ${type} (${desc.label ?? key}) at ${line}:${col}: ${text}`,
							);
						}
					}
				})
				.catch(() => {});
		}

		const layout =
			desc.bindGroupLayouts && desc.bindGroupLayouts.length > 0
				? this.device.createPipelineLayout({
						bindGroupLayouts: desc.bindGroupLayouts,
						label: `${this.labelPrefix}:pipelineLayout:${desc.label ?? key}`,
					})
				: "auto";

		try {
			const pipeline = this.device.createComputePipeline({
				label: `${this.labelPrefix}:computePipeline:${desc.label ?? key}`,
				layout,
				compute: { module, entryPoint },
			});

			this.compute.set(key, pipeline);
			return pipeline;
		} catch (e: any) {
			throw new PipelineCompileError(
				`Compute pipeline compile failed: ${e?.message ?? e}`,
			);
		}
	}

	clear(): void {
		this.compute.clear();
	}

	size(): number {
		return this.compute.size;
	}
}
