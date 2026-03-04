// packages/gpu/src/kernels/VectorAddKernel.ts

import type { BufferPool } from "../runtime/BufferPool";
import type { CommandScheduler } from "../runtime/CommandScheduler";
import type { PipelineCache } from "../runtime/PipelineCache";

import code from "./vectorAdd.wgsl?raw";

export class VectorAddKernel {
	private pipeline: GPUComputePipeline;
	private bgl: GPUBindGroupLayout;

	constructor(
		private device: GPUDevice,
		private pipelineCache: PipelineCache,
		private pool: BufferPool,
		private scheduler: CommandScheduler,
		private labelPrefix = "wispdb",
	) {
		// Use pipeline layout auto for now; read layout back via getBindGroupLayout.
		this.pipeline = this.pipelineCache.getComputePipeline({
			code,
			label: "vectorAdd",
		});
		this.bgl = this.pipeline.getBindGroupLayout(0);
	}

	async run(
		a: GPUBuffer,
		b: GPUBuffer,
		out: GPUBuffer,
		n: number,
	): Promise<void> {
		const bindGroup = this.device.createBindGroup({
			layout: this.bgl,
			entries: [
				{ binding: 0, resource: { buffer: a } },
				{ binding: 1, resource: { buffer: b } },
				{ binding: 2, resource: { buffer: out } },
			],
			label: `${this.labelPrefix}:bindGroup:vectorAdd`,
		});

		const workgroups = Math.ceil(n / 256);

		await this.scheduler.submit((encoder) => {
			const pass = encoder.beginComputePass({
				label: `${this.labelPrefix}:pass:vectorAdd`,
			});
			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.dispatchWorkgroups(workgroups);
			pass.end();
		}, "vectorAdd");

		// Bind groups are lightweight; let GC do it. Buffers are pooled, not bind groups.
	}
}
