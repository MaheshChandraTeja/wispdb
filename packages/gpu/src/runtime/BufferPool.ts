// packages/gpu/src/runtime/BufferPool.ts
import type { BufferPoolOptions, BufferPoolStats } from "./types";

function alignUp(n: number, a: number): number {
	return Math.ceil(n / a) * a;
}

type PoolKey = string;

function makeKey(usage: GPUBufferUsageFlags, size: number): PoolKey {
	return `${usage}:${size}`;
}

export class BufferPool {
	private free = new Map<PoolKey, GPUBuffer[]>();
	private stats: BufferPoolStats = {
		liveBuffers: 0,
		liveBytes: 0,
		freeBuffers: 0,
		freeBytes: 0,
		allocations: 0,
		reuses: 0,
		destroys: 0,
	};

	private maxBytes: number;
	private alignment: number;

	constructor(
		private device: GPUDevice,
		opts: BufferPoolOptions = {},
		private labelPrefix = "wispdb",
	) {
		this.maxBytes = opts.maxBytes ?? 256 * 1024 * 1024; // 256MB default cap (tune later)
		this.alignment = opts.bucketAlignment ?? 256;
	}

	getStats(): BufferPoolStats {
		return { ...this.stats };
	}

	acquire(size: number, usage: GPUBufferUsageFlags, label?: string): GPUBuffer {
		const aligned = alignUp(size, this.alignment);
		const key = makeKey(usage, aligned);
		const bucket = this.free.get(key);

		if (bucket && bucket.length > 0) {
			const buf = bucket.pop()!;
			this.stats.freeBuffers--;
			this.stats.freeBytes -= aligned;
			this.stats.reuses++;
			return buf;
		}

		const buf = this.device.createBuffer({
			size: aligned,
			usage,
			label: `${this.labelPrefix}:buf:${label ?? key}`,
		});

		this.stats.allocations++;
		this.stats.liveBuffers++;
		this.stats.liveBytes += aligned;
		return buf;
	}

	release(buffer: GPUBuffer, size: number, usage: GPUBufferUsageFlags): void {
		const aligned = alignUp(size, this.alignment);
		const key = makeKey(usage, aligned);

		if (!this.free.has(key)) this.free.set(key, []);
		this.free.get(key)!.push(buffer);

		this.stats.freeBuffers++;
		this.stats.freeBytes += aligned;

		// Trim if we’re hoarding like a dragon
		if (this.stats.freeBytes > this.maxBytes) this.trim();
	}

	trim(targetFreeBytes = Math.floor(this.maxBytes * 0.5)): void {
		// Brutal but effective: destroy from largest buckets first.
		const entries = [...this.free.entries()]
			.map(([k, arr]) => {
				const size = Number(k.split(":")[1]);
				return { k, arr, size };
			})
			.sort((a, b) => b.size - a.size);

		for (const { k, arr, size } of entries) {
			while (arr.length > 0 && this.stats.freeBytes > targetFreeBytes) {
				const buf = arr.pop()!;
				buf.destroy();
				this.stats.destroys++;
				this.stats.freeBuffers--;
				this.stats.freeBytes -= size;

				this.stats.liveBuffers--;
				this.stats.liveBytes -= size;
			}
			if (arr.length === 0) this.free.delete(k);
			if (this.stats.freeBytes <= targetFreeBytes) break;
		}
	}

	destroyAll(): void {
		for (const [k, arr] of this.free.entries()) {
			const size = Number(k.split(":")[1]);
			for (const b of arr) {
				b.destroy();
				this.stats.destroys++;
				this.stats.freeBuffers--;
				this.stats.freeBytes -= size;

				this.stats.liveBuffers--;
				this.stats.liveBytes -= size;
			}
		}
		this.free.clear();
	}
}
