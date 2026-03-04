// packages/gpu/src/runtime/DeviceManager.ts

import { DeviceRequestError, WebGPUUnavailableError } from "./errors";
import type { DeviceManagerOptions } from "./types";

export class DeviceManager {
	private adapter: GPUAdapter | null = null;
	private device: GPUDevice | null = null;
	private queue: GPUQueue | null = null;

	private initPromise: Promise<void> | null = null;
	private destroyed = false;

	private labelPrefix: string;

	constructor(private opts: DeviceManagerOptions = {}) {
		this.labelPrefix = opts.labelPrefix ?? "wispdb";
	}

	isReady(): boolean {
		return !!this.device && !!this.queue;
	}

	getDevice(): GPUDevice {
		if (!this.device)
			throw new Error("DeviceManager not initialized. Call init().");
		return this.device;
	}

	getQueue(): GPUQueue {
		if (!this.queue)
			throw new Error("DeviceManager not initialized. Call init().");
		return this.queue;
	}

	getAdapter(): GPUAdapter {
		if (!this.adapter)
			throw new Error("DeviceManager not initialized. Call init().");
		return this.adapter;
	}

	async init(): Promise<void> {
		if (this.destroyed) throw new Error("DeviceManager is destroyed.");
		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			const nav = typeof navigator === "undefined" ? null : (navigator as any);
			if (!nav || !nav.gpu) {
				throw new WebGPUUnavailableError();
			}

			const powerPreference = this.opts.powerPreference ?? "high-performance";

			// Adapter request
			const adapter = await nav.gpu.requestAdapter({
				powerPreference,
				forceFallbackAdapter: this.opts.forceFallbackAdapter ?? false,
			});

			if (!adapter)
				throw new DeviceRequestError("Failed to acquire GPUAdapter.");

			// Feature/limit negotiation (don’t demand fantasy limits)
			const requiredFeatures = this.opts.requiredFeatures ?? [];
			for (const f of requiredFeatures) {
				if (!adapter.features.has(f)) {
					throw new DeviceRequestError(`Required feature not supported: ${f}`);
				}
			}

			const requiredLimits = this.opts.requiredLimits ?? {};
			const limits: Record<string, number> = {};
			for (const [k, v] of Object.entries(requiredLimits)) {
				const supported = (adapter.limits as Record<string, number>)[k];
				if (typeof supported === "number" && typeof v === "number") {
					if (v > supported) {
						throw new DeviceRequestError(
							`Required limit ${k}=${v} exceeds supported ${supported}`,
						);
					}
					limits[k] = v;
				}
			}

			const device = await adapter.requestDevice({
				requiredFeatures,
				requiredLimits: limits,
				label: `${this.labelPrefix}:device`,
			});

			// Capture “uncaptured” errors (WebGPU will happily fail silently otherwise)
			device.addEventListener(
				"uncapturederror",
				(ev: GPUUncapturedErrorEvent) => {
					// Keep it loud in dev, controllable later.
					console.error(
						`[${this.labelPrefix}] Uncaptured WebGPU error:`,
						ev.error,
					);
				},
			);

			// Lost device handling: you can rebuild higher-level systems on callback.
			device.lost.then((info: GPUDeviceLostInfo) => {
				console.warn(`[${this.labelPrefix}] Device lost:`, info);
				// mark unusable; upstream can recreate DeviceManager (or we can auto-reinit later)
				this.device = null;
				this.queue = null;
			});

			this.adapter = adapter;
			this.device = device;
			this.queue = device.queue;
		})();

		return this.initPromise;
	}

	destroy(): void {
		this.destroyed = true;
		this.adapter = null;
		this.device = null;
		this.queue = null;
		this.initPromise = null;
	}
}
