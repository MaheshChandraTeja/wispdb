// packages/gpu/src/runtime/CommandScheduler.ts
import type { SchedulerOptions } from "./types";

export type ComputeWork = (encoder: GPUCommandEncoder) => void;

export class CommandScheduler {
	private inFlight = 0;
	private maxInFlight: number;
	private labelPrefix: string;

	constructor(
		private device: GPUDevice,
		private queue: GPUQueue,
		opts: SchedulerOptions = {},
	) {
		this.maxInFlight = opts.maxInFlight ?? 2;
		this.labelPrefix = opts.labelPrefix ?? "wispdb";
	}

	async submit(work: ComputeWork, label?: string): Promise<void> {
		// Throttle if requested (prevents queue ballooning in tight loops)
		while (this.inFlight >= this.maxInFlight) {
			await this.queue.onSubmittedWorkDone();
			this.inFlight = 0; // conservative reset; keeps behavior stable
		}

		const encoder = this.device.createCommandEncoder({
			label: `${this.labelPrefix}:encoder:${label ?? "job"}`,
		});

		work(encoder);

		const cmd = encoder.finish();
		this.queue.submit([cmd]);

		this.inFlight++;

		// Optional: if you want strict determinism, await completion here.
		// await this.queue.onSubmittedWorkDone();
	}

	async flush(): Promise<void> {
		await this.queue.onSubmittedWorkDone();
		this.inFlight = 0;
	}
}
