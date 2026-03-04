import { BufferPool } from "../runtime/BufferPool";
import { CommandScheduler } from "../runtime/CommandScheduler";
import type { DeviceManager } from "../runtime/DeviceManager";
import { PipelineCache } from "../runtime/PipelineCache";
import { WGSL } from "./wgsl";

export type Metric = "dot" | "cosine" | "l2";
export type TopKMode = "cpu" | "gpu_block";

export interface ExactSearchResult {
	index: number;
	score: number;
}

function ceilDiv(a: number, b: number) {
	return Math.floor((a + b - 1) / b);
}

function u32bytes(n: number) {
	return n * 4;
}

function pairBytes(nPairs: number) {
	return nPairs * 8; // f32 + u32
}

function writeParams(
	queue: GPUQueue,
	buf: GPUBuffer,
	n: number,
	dim: number,
	k: number,
) {
	const p = new Uint32Array([n >>> 0, dim >>> 0, k >>> 0, 0]);
	queue.writeBuffer(buf, 0, p);
}

// CPU TopK on scores (descending). Stable tie-break by index.
export function topkFromScoresCPU(
	scores: Float32Array,
	k: number,
): ExactSearchResult[] {
	const n = scores.length;
	const kk = Math.min(k, n);

	// min-heap of size k over (score, idx) with "worst" at root
	const heapScore = new Float32Array(kk);
	const heapIdx = new Uint32Array(kk);
	let size = 0;

	function worse(iA: number, sA: number, iB: number, sB: number): boolean {
		// worse means smaller score; tie: larger idx worse
		if (sA < sB) return true;
		if (sA > sB) return false;
		return iA > iB;
	}

	function siftUp(i: number) {
		while (i > 0) {
			const p = (i - 1) >> 1;
			// if parent is worse-than child? we want worst at root, so parent should be worse or equal
			// if child is worse than parent, swap (to keep worst nearer root)
			const ci = heapIdx[i],
				cs = heapScore[i];
			const pi = heapIdx[p],
				ps = heapScore[p];
			if (worse(ci, cs, pi, ps)) {
				heapIdx[i] = pi;
				heapScore[i] = ps;
				heapIdx[p] = ci;
				heapScore[p] = cs;
				i = p;
			} else break;
		}
	}

	function siftDown(i: number) {
		while (true) {
			const l = i * 2 + 1;
			const r = l + 1;
			let smallest = i;

			if (
				l < size &&
				worse(heapIdx[l], heapScore[l], heapIdx[smallest], heapScore[smallest])
			) {
				smallest = l;
			}
			if (
				r < size &&
				worse(heapIdx[r], heapScore[r], heapIdx[smallest], heapScore[smallest])
			) {
				smallest = r;
			}
			if (smallest === i) break;

			const ti = heapIdx[i],
				ts = heapScore[i];
			heapIdx[i] = heapIdx[smallest];
			heapScore[i] = heapScore[smallest];
			heapIdx[smallest] = ti;
			heapScore[smallest] = ts;

			i = smallest;
		}
	}

	for (let i = 0; i < n; i++) {
		const s = scores[i];
		if (size < kk) {
			heapIdx[size] = i;
			heapScore[size] = s;
			siftUp(size);
			size++;
		} else {
			// if current better than worst at root, replace root
			const wi = heapIdx[0],
				ws = heapScore[0];
			// better means higher score; tie smaller idx
			const better = s > ws || (s === ws && i < wi);
			if (better) {
				heapIdx[0] = i;
				heapScore[0] = s;
				siftDown(0);
			}
		}
	}

	// heap contains top-k but unordered; sort descending
	const out: ExactSearchResult[] = [];
	for (let i = 0; i < size; i++) {
		out.push({ index: heapIdx[i], score: heapScore[i] });
	}
	out.sort((a, b) => b.score - a.score || a.index - b.index);
	return out;
}

export class ExactSearchGPU {
	private device!: GPUDevice;
	private queue!: GPUQueue;

	private cache!: PipelineCache;
	private pool!: BufferPool;
	private scheduler!: CommandScheduler;

	private dim = 0;
	private n = 0;

	private datasetRaw: GPUBuffer | null = null;
	private datasetNorm: GPUBuffer | null = null;

	constructor(
		private dm: DeviceManager,
		private labelPrefix = "wispdb-m2",
	) {}

	async init() {
		await this.dm.init();
		this.device = this.dm.getDevice();
		this.queue = this.dm.getQueue();

		this.cache = new PipelineCache(this.device, this.labelPrefix);
		this.pool = new BufferPool(
			this.device,
			{ maxBytes: 256 * 1024 * 1024 },
			this.labelPrefix,
		);
		this.scheduler = new CommandScheduler(this.device, this.queue, {
			maxInFlight: 2,
			labelPrefix: this.labelPrefix,
		});
	}

	setDataset(vectors: Float32Array, dim: number) {
		if (dim <= 0) throw new Error("dim must be > 0");
		if (vectors.length % dim !== 0)
			throw new Error("vectors length must be multiple of dim");

		this.dim = dim;
		this.n = vectors.length / dim;

		// Destroy old buffers
		this.datasetRaw?.destroy();
		this.datasetNorm?.destroy();
		this.datasetRaw = null;
		this.datasetNorm = null;

		const bytes = vectors.byteLength;

		this.datasetRaw = this.device.createBuffer({
			size: bytes,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:datasetRaw`,
		});

		this.queue.writeBuffer(this.datasetRaw, 0, vectors);
	}

	async ensureDatasetNormalized() {
		if (!this.datasetRaw) throw new Error("Dataset not set");
		if (this.datasetNorm) return;

		const bytes = this.n * this.dim * 4;
		this.datasetNorm = this.device.createBuffer({
			size: bytes,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			label: `${this.labelPrefix}:datasetNorm`,
		});

		const paramsBuf = this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:paramsNormDataset`,
		});
		writeParams(this.queue, paramsBuf, this.n, this.dim, 0);

		const pipeline = this.cache.getComputePipeline({
			code: WGSL.normalizeBatch,
			label: "normalizeBatch",
		});
		const bgl = pipeline.getBindGroupLayout(0);

		const bg = this.device.createBindGroup({
			layout: bgl,
			entries: [
				{ binding: 0, resource: { buffer: this.datasetRaw } },
				{ binding: 1, resource: { buffer: this.datasetNorm } },
				{ binding: 2, resource: { buffer: paramsBuf } },
			],
			label: `${this.labelPrefix}:bgNormDataset`,
		});

		const workgroups = ceilDiv(this.n, 256);

		await this.scheduler.submit((encoder) => {
			const pass = encoder.beginComputePass({
				label: `${this.labelPrefix}:passNormDataset`,
			});
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(workgroups);
			pass.end();
		}, "normDataset");
		await this.scheduler.flush();

		paramsBuf.destroy();
	}

	async searchExact(
		query: Float32Array,
		k: number,
		metric: Metric,
		mode: TopKMode = "gpu_block",
	) {
		if (!this.datasetRaw) throw new Error("Dataset not set");
		if (query.length !== this.dim)
			throw new Error(`Query dim mismatch: expected ${this.dim}`);
		if (k <= 0) return [];

		const kk = Math.min(k, this.n);
		const n = this.n;
		const dim = this.dim;

		// Prepare query buffer (raw)
		const qBytes = query.byteLength;
		const queryBuf = this.pool.acquire(
			qBytes,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			"query",
		);
		this.queue.writeBuffer(queryBuf, 0, query);

		// If cosine: normalize query + ensure dataset normalized
		let queryUsed = queryBuf;
		let datasetUsed = this.datasetRaw;

		let queryNormBuf: GPUBuffer | null = null;

		if (metric === "cosine") {
			await this.ensureDatasetNormalized();
			if (!this.datasetNorm) throw new Error("Normalization failed");
			datasetUsed = this.datasetNorm;

			queryNormBuf = this.pool.acquire(
				qBytes,
				GPUBufferUsage.STORAGE,
				"queryNorm",
			);
			const paramsBuf = this.pool.acquire(
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
				"paramsNormQuery",
			);
			writeParams(this.queue, paramsBuf, 1, dim, 0);

			const pipeline = this.cache.getComputePipeline({
				code: WGSL.normalizeBatch,
				label: "normalizeBatch",
			});
			const bgl = pipeline.getBindGroupLayout(0);

			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: queryBuf } },
					{ binding: 1, resource: { buffer: queryNormBuf } },
					{ binding: 2, resource: { buffer: paramsBuf } },
				],
				label: `${this.labelPrefix}:bgNormQuery`,
			});

			await this.scheduler.submit((encoder) => {
				const pass = encoder.beginComputePass({
					label: `${this.labelPrefix}:passNormQuery`,
				});
				pass.setPipeline(pipeline);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(1);
				pass.end();
			}, "normQuery");
			await this.scheduler.flush();

			this.pool.release(
				paramsBuf,
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			);
			queryUsed = queryNormBuf;
		}

		// Score buffer
		const scoresBuf = this.pool.acquire(
			u32bytes(n),
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			"scores",
		);
		const paramsBufScore = this.pool.acquire(
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			"paramsScore",
		);
		writeParams(this.queue, paramsBufScore, n, dim, kk);

		const scoreCode = metric === "l2" ? WGSL.scoreL2Neg : WGSL.scoreDot; // cosine uses dot on normalized
		const scorePipeline = this.cache.getComputePipeline({
			code: scoreCode,
			label: `score_${metric}`,
		});
		const scoreBgl = scorePipeline.getBindGroupLayout(0);

		const scoreBg = this.device.createBindGroup({
			layout: scoreBgl,
			entries: [
				{ binding: 0, resource: { buffer: datasetUsed } },
				{ binding: 1, resource: { buffer: queryUsed } },
				{ binding: 2, resource: { buffer: scoresBuf } },
				{ binding: 3, resource: { buffer: paramsBufScore } },
			],
			label: `${this.labelPrefix}:bgScore`,
		});

		const workgroups = ceilDiv(n, 256);

		await this.scheduler.submit((encoder) => {
			const pass = encoder.beginComputePass({
				label: `${this.labelPrefix}:passScore`,
			});
			pass.setPipeline(scorePipeline);
			pass.setBindGroup(0, scoreBg);
			pass.dispatchWorkgroups(workgroups);
			pass.end();
		}, "score");
		await this.scheduler.flush();

		let results: ExactSearchResult[] = [];

		const readScoresCPU = async (): Promise<ExactSearchResult[]> => {
			const readback = this.device.createBuffer({
				size: u32bytes(n),
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
				label: `${this.labelPrefix}:readbackScores`,
			});

			const enc = this.device.createCommandEncoder({
				label: `${this.labelPrefix}:encCopyScores`,
			});
			enc.copyBufferToBuffer(scoresBuf, 0, readback, 0, u32bytes(n));
			this.queue.submit([enc.finish()]);
			await this.queue.onSubmittedWorkDone();

			await readback.mapAsync(GPUMapMode.READ);
			const raw = readback.getMappedRange();
			const scores = new Float32Array(raw.slice(0));
			readback.unmap();
			readback.destroy();

			return topkFromScoresCPU(scores, kk);
		};

		const isValidTopK = (items: ExactSearchResult[]): boolean => {
			if (items.length !== kk) return false;
			const seen = new Set<number>();
			for (const r of items) {
				if (!Number.isFinite(r.score)) return false;
				if (r.index >= n) return false;
				if (seen.has(r.index)) return false;
				seen.add(r.index);
			}
			return true;
		};

		if (mode === "cpu") {
			// Read scores back and do CPU TopK (useful baseline)
			results = await readScoresCPU();
		} else {
			// GPU TopK: scores -> block topk -> reduce until <= 256 -> final topk
			// Stage 1
			const currentN = n;
			let inIsScores = true;

			const blockTopKScoresPipe = this.cache.getComputePipeline({
				code: WGSL.blockTopKScores,
				label: "blockTopKScores",
			});
			const blockTopKPairsPipe = this.cache.getComputePipeline({
				code: WGSL.blockTopKPairs,
				label: "blockTopKPairs",
			});

			const paramsBufTopK = this.pool.acquire(
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
				"paramsTopK",
			);
			writeParams(this.queue, paramsBufTopK, currentN, dim, kk);

			const numBlocks1 = ceilDiv(currentN, 256);
			let pairsBufA = this.pool.acquire(
				pairBytes(numBlocks1 * kk),
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
				"pairsA",
			);

			{
				const bgl = blockTopKScoresPipe.getBindGroupLayout(0);
				const bg = this.device.createBindGroup({
					layout: bgl,
					entries: [
						{ binding: 0, resource: { buffer: scoresBuf } },
						{ binding: 1, resource: { buffer: pairsBufA } },
						{ binding: 2, resource: { buffer: paramsBufTopK } },
					],
					label: `${this.labelPrefix}:bgTopK1`,
				});

				await this.scheduler.submit((encoder) => {
					const pass = encoder.beginComputePass({
						label: `${this.labelPrefix}:passTopK1`,
					});
					pass.setPipeline(blockTopKScoresPipe);
					pass.setBindGroup(0, bg);
					pass.dispatchWorkgroups(numBlocks1);
					pass.end();
				}, "topk1");
				await this.scheduler.flush();
			}

			// Reduce
			let currentPairs = numBlocks1 * kk;
			inIsScores = false;

			while (currentPairs > 256) {
				const nextBlocks = ceilDiv(currentPairs, 256);
				const pairsBufB = this.pool.acquire(
					pairBytes(nextBlocks * kk),
					GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
					"pairsB",
				);

				writeParams(this.queue, paramsBufTopK, currentPairs, dim, kk);

				const bgl = blockTopKPairsPipe.getBindGroupLayout(0);
				const bg = this.device.createBindGroup({
					layout: bgl,
					entries: [
						{ binding: 0, resource: { buffer: pairsBufA } },
						{ binding: 1, resource: { buffer: pairsBufB } },
						{ binding: 2, resource: { buffer: paramsBufTopK } },
					],
					label: `${this.labelPrefix}:bgReduce`,
				});

				await this.scheduler.submit((encoder) => {
					const pass = encoder.beginComputePass({
						label: `${this.labelPrefix}:passReduce`,
					});
					pass.setPipeline(blockTopKPairsPipe);
					pass.setBindGroup(0, bg);
					pass.dispatchWorkgroups(nextBlocks);
					pass.end();
				}, "reduce");
				await this.scheduler.flush();

				// release old A, swap
				this.pool.release(
					pairsBufA,
					pairBytes(currentPairs),
					GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
				);
				pairsBufA = pairsBufB;
				currentPairs = nextBlocks * kk;
			}

			// Final: one more blockTopKPairs pass to output exactly k pairs (one block)
			const finalOut = this.pool.acquire(
				pairBytes(kk),
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
				"pairsFinal",
			);
			writeParams(this.queue, paramsBufTopK, currentPairs, dim, kk);

			{
				const bgl = blockTopKPairsPipe.getBindGroupLayout(0);
				const bg = this.device.createBindGroup({
					layout: bgl,
					entries: [
						{ binding: 0, resource: { buffer: pairsBufA } },
						{ binding: 1, resource: { buffer: finalOut } },
						{ binding: 2, resource: { buffer: paramsBufTopK } },
					],
					label: `${this.labelPrefix}:bgFinal`,
				});

				await this.scheduler.submit((encoder) => {
					const pass = encoder.beginComputePass({
						label: `${this.labelPrefix}:passFinal`,
					});
					pass.setPipeline(blockTopKPairsPipe);
					pass.setBindGroup(0, bg);
					pass.dispatchWorkgroups(1);
					pass.end();
				}, "final");
				await this.scheduler.flush();
			}

			// Read back final k pairs
			const readback = this.device.createBuffer({
				size: pairBytes(kk),
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
				label: `${this.labelPrefix}:readbackTopK`,
			});

			const enc = this.device.createCommandEncoder({
				label: `${this.labelPrefix}:encCopyTopK`,
			});
			enc.copyBufferToBuffer(finalOut, 0, readback, 0, pairBytes(kk));
			this.queue.submit([enc.finish()]);
			await this.queue.onSubmittedWorkDone();

			await readback.mapAsync(GPUMapMode.READ);
			const raw = readback.getMappedRange();
			const dv = new DataView(raw.slice(0)); // copy
			readback.unmap();
			readback.destroy();

			function bitsToF32(u: number): number {
				const buf = new ArrayBuffer(4);
				new DataView(buf).setUint32(0, u, true);
				return new DataView(buf).getFloat32(0, true);
			}

			results = [];
			for (let i = 0; i < kk; i++) {
				const scoreBits = dv.getUint32(i * 8 + 0, true);
				const idx = dv.getUint32(i * 8 + 4, true);
				const score = bitsToF32(scoreBits);
				results.push({ index: idx, score });
			}

			results.sort((a, b) => b.score - a.score || a.index - b.index);

			if (!isValidTopK(results)) {
				console.warn(
					`[${this.labelPrefix}] GPU TopK invalid; falling back to CPU TopK.`,
				);
				results = await readScoresCPU();
			}

			// cleanup intermediates
			this.pool.release(
				finalOut,
				pairBytes(kk),
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			);
			this.pool.release(
				pairsBufA,
				pairBytes(currentPairs),
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			);
			this.pool.release(
				paramsBufTopK,
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			);
		}

		// Cleanup common buffers
		this.pool.release(
			paramsBufScore,
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		);
		this.pool.release(
			scoresBuf,
			u32bytes(n),
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		);
		this.pool.release(
			queryBuf,
			qBytes,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);
		if (queryNormBuf)
			this.pool.release(queryNormBuf, qBytes, GPUBufferUsage.STORAGE);

		return results;
	}
}
