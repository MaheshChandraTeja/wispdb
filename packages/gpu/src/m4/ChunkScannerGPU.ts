import { WGSL } from "../m2/wgsl"; // must include packed vec2<u32> TopK kernels
import { BufferPool } from "../runtime/BufferPool";
import { CommandScheduler } from "../runtime/CommandScheduler";
import type { DeviceManager } from "../runtime/DeviceManager";
import { PipelineCache } from "../runtime/PipelineCache";

export type Metric = "dot" | "cosine" | "l2";

export interface Pair {
	idx: number; // index within current batch
	score: number; // score (higher is better; for l2 we use -l2sq)
}

function ceilDiv(a: number, b: number) {
	return Math.floor((a + b - 1) / b);
}

function writeParams(
	queue: GPUQueue,
	buf: GPUBuffer,
	n: number,
	dim: number,
	k: number,
) {
	queue.writeBuffer(buf, 0, new Uint32Array([n >>> 0, dim >>> 0, k >>> 0, 0]));
}

function bitsToF32(u: number): number {
	const b = new ArrayBuffer(4);
	const dv = new DataView(b);
	dv.setUint32(0, u, true);
	return dv.getFloat32(0, true);
}

function topkFromScoresCPU(scores: Float32Array, k: number): Pair[] {
	const n = scores.length;
	const kk = Math.min(k, n);
	if (kk <= 0) return [];

	const heapScore = new Float32Array(kk);
	const heapIdx = new Uint32Array(kk);
	let size = 0;

	const worse = (iA: number, sA: number, iB: number, sB: number) => {
		if (sA < sB) return true;
		if (sA > sB) return false;
		return iA > iB;
	};

	const siftUp = (i: number) => {
		while (i > 0) {
			const p = (i - 1) >> 1;
			const ci = heapIdx[i], cs = heapScore[i];
			const pi = heapIdx[p], ps = heapScore[p];
			if (worse(ci, cs, pi, ps)) {
				heapIdx[i] = pi; heapScore[i] = ps;
				heapIdx[p] = ci; heapScore[p] = cs;
				i = p;
			} else break;
		}
	};

	const siftDown = (i: number) => {
		while (true) {
			const l = i * 2 + 1;
			const r = l + 1;
			let m = i;
			if (l < size && worse(heapIdx[l], heapScore[l], heapIdx[m], heapScore[m])) m = l;
			if (r < size && worse(heapIdx[r], heapScore[r], heapIdx[m], heapScore[m])) m = r;
			if (m === i) break;
			const ti = heapIdx[i], ts = heapScore[i];
			heapIdx[i] = heapIdx[m]; heapScore[i] = heapScore[m];
			heapIdx[m] = ti; heapScore[m] = ts;
			i = m;
		}
	};

	for (let i = 0; i < n; i++) {
		const s = scores[i];
		if (size < kk) {
			heapIdx[size] = i;
			heapScore[size] = s;
			siftUp(size);
			size++;
		} else {
			const wi = heapIdx[0], ws = heapScore[0];
			const better = s > ws || (s === ws && i < wi);
			if (better) {
				heapIdx[0] = i;
				heapScore[0] = s;
				siftDown(0);
			}
		}
	}

	const out: Pair[] = [];
	for (let i = 0; i < size; i++) out.push({ idx: heapIdx[i], score: heapScore[i] });
	out.sort((a, b) => (b.score - a.score) || (a.idx - b.idx));
	return out;
}

function isValidTopK(pairs: Pair[], n: number, kk: number): boolean {
	if (pairs.length !== kk) return false;
	const seen = new Set<number>();
	for (const p of pairs) {
		if (!Number.isFinite(p.score)) return false;
		if (p.idx < 0 || p.idx >= n) return false;
		if (seen.has(p.idx)) return false;
		seen.add(p.idx);
	}
	return true;
}

export class ChunkScannerGPU {
	private device!: GPUDevice;
	private queue!: GPUQueue;

	private cache!: PipelineCache;
	private pool!: BufferPool;
	private scheduler!: CommandScheduler;

	constructor(
		private dm: DeviceManager,
		private labelPrefix = "wispdb-m4",
	) {}

	async open() {
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

		await this.warmupPipelines();
	}

	private async warmupPipelines(): Promise<void> {
		const normalizePipe = this.cache.getComputePipeline({
			code: WGSL.normalizeBatch,
			label: "normalizeBatch_warmup",
		});
		const scoreDot = this.cache.getComputePipeline({
			code: WGSL.scoreDot,
			label: "scoreDot_warmup",
		});
		const scoreL2 = this.cache.getComputePipeline({
			code: WGSL.scoreL2Neg,
			label: "scoreL2_warmup",
		});
		const topK1 = this.cache.getComputePipeline({
			code: WGSL.blockTopKScores,
			label: "blockTopKScores_warmup",
		});
		const topKReduce = this.cache.getComputePipeline({
			code: WGSL.blockTopKPairs,
			label: "blockTopKPairs_warmup",
		});

		const vec = this.device.createBuffer({
			size: 4 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:vec`,
		});
		const vecOut = this.device.createBuffer({
			size: 4 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:vecOut`,
		});
		const query = this.device.createBuffer({
			size: 4 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:query`,
		});
		const scores = this.device.createBuffer({
			size: 4 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:scores`,
		});
		const params = this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:params`,
		});
		this.queue.writeBuffer(params, 0, new Uint32Array([1, 4, 1, 0]));

		const pairsIn = this.device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:pairsIn`,
		});
		const pairsOut = this.device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			label: `${this.labelPrefix}:warmup:pairsOut`,
		});

		const encoder = this.device.createCommandEncoder({
			label: `${this.labelPrefix}:encWarmup`,
		});
		const pass = encoder.beginComputePass({
			label: `${this.labelPrefix}:passWarmup`,
		});

		{
			const bgl = normalizePipe.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: vec } },
					{ binding: 1, resource: { buffer: vecOut } },
					{ binding: 2, resource: { buffer: params } },
				],
			});
			pass.setPipeline(normalizePipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
		}

		{
			const bgl = scoreDot.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: vec } },
					{ binding: 1, resource: { buffer: query } },
					{ binding: 2, resource: { buffer: scores } },
					{ binding: 3, resource: { buffer: params } },
				],
			});
			pass.setPipeline(scoreDot);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
		}

		{
			const bgl = scoreL2.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: vec } },
					{ binding: 1, resource: { buffer: query } },
					{ binding: 2, resource: { buffer: scores } },
					{ binding: 3, resource: { buffer: params } },
				],
			});
			pass.setPipeline(scoreL2);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
		}

		{
			const bgl = topK1.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: scores } },
					{ binding: 1, resource: { buffer: pairsOut } },
					{ binding: 2, resource: { buffer: params } },
				],
			});
			pass.setPipeline(topK1);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
		}

		{
			const bgl = topKReduce.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: pairsIn } },
					{ binding: 1, resource: { buffer: pairsOut } },
					{ binding: 2, resource: { buffer: params } },
				],
			});
			pass.setPipeline(topKReduce);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
		}

		pass.end();
		this.queue.submit([encoder.finish()]);
		await this.queue.onSubmittedWorkDone();

		vec.destroy();
		vecOut.destroy();
		query.destroy();
		scores.destroy();
		params.destroy();
		pairsIn.destroy();
		pairsOut.destroy();
	}

	async prepareQuery(query: Float32Array, dim: number, metric: Metric) {
		if (query.length !== dim)
			throw new Error(
				`Query dim mismatch: expected ${dim}, got ${query.length}`,
			);

		const qBytes = query.byteLength;
		const queryBuf = this.pool.acquire(
			qBytes,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			"query",
		);
		this.queue.writeBuffer(queryBuf, 0, query);

		if (metric !== "cosine") {
			return {
				queryUsed: queryBuf,
				release: () =>
					this.pool.release(
						queryBuf,
						qBytes,
						GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
					),
			};
		}

		// cosine: normalize query on GPU
		const queryNorm = this.pool.acquire(
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

		const pipe = this.cache.getComputePipeline({
			code: WGSL.normalizeBatch,
			label: "normalizeBatch",
		});
		const bgl = pipe.getBindGroupLayout(0);

		const bg = this.device.createBindGroup({
			layout: bgl,
			entries: [
				{ binding: 0, resource: { buffer: queryBuf } },
				{ binding: 1, resource: { buffer: queryNorm } },
				{ binding: 2, resource: { buffer: paramsBuf } },
			],
			label: `${this.labelPrefix}:bgNormQuery`,
		});

		await this.scheduler.submit((encoder) => {
			const pass = encoder.beginComputePass({
				label: `${this.labelPrefix}:passNormQuery`,
			});
			pass.setPipeline(pipe);
			pass.setBindGroup(0, bg);
			pass.dispatchWorkgroups(1);
			pass.end();
		}, "normQuery");
		await this.scheduler.flush();

		return {
			queryUsed: queryNorm,
			release: () => {
				this.pool.release(
					queryBuf,
					qBytes,
					GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
				);
				this.pool.release(queryNorm, qBytes, GPUBufferUsage.STORAGE);
				this.pool.release(
					paramsBuf,
					16,
					GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
				);
			},
		};
	}

	async topKBatch(
		vectors: Float32Array, // packed [row][dim]
		dim: number,
		queryUsed: GPUBuffer,
		metric: Metric,
		k: number,
		vectorsNormalized = false,
	): Promise<Pair[]> {
		if (k <= 0) return [];
		if (k > 256)
			throw new Error("k > 256 not supported in GPU TopK v1 (bitonic block).");
		if (vectors.length % dim !== 0)
			throw new Error("vectors length must be multiple of dim");

		const n = vectors.length / dim;
		const kk = Math.min(k, n);

		const vecBytes = vectors.byteLength;

		// Upload batch vectors
		const vecBuf = this.pool.acquire(
			vecBytes,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			"vecBatch",
		);
		this.queue.writeBuffer(vecBuf, 0, vectors);

		// For cosine: normalize vectors batch into vecNorm, then dot with queryUsed
		let vecUsed = vecBuf;
		let vecNorm: GPUBuffer | null = null;
		let paramsNorm: GPUBuffer | null = null;

		if (metric === "cosine" && !vectorsNormalized) {
			vecNorm = this.pool.acquire(
				vecBytes,
				GPUBufferUsage.STORAGE,
				"vecNormBatch",
			);
			paramsNorm = this.pool.acquire(
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
				"paramsNormBatch",
			);
			writeParams(this.queue, paramsNorm, n, dim, 0);

			const pipe = this.cache.getComputePipeline({
				code: WGSL.normalizeBatch,
				label: "normalizeBatch",
			});
			const bgl = pipe.getBindGroupLayout(0);

			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: vecBuf } },
					{ binding: 1, resource: { buffer: vecNorm } },
					{ binding: 2, resource: { buffer: paramsNorm } },
				],
			});

			await this.scheduler.submit((encoder) => {
				const pass = encoder.beginComputePass({
					label: `${this.labelPrefix}:passNormBatch`,
				});
				pass.setPipeline(pipe);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(ceilDiv(n, 256));
				pass.end();
			}, "normBatch");
			await this.scheduler.flush();

			vecUsed = vecNorm;
		}

		// Score buffer
		const scoresBuf = this.pool.acquire(
			n * 4,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			"scores",
		);
		const paramsScore = this.pool.acquire(
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			"paramsScore",
		);
		writeParams(this.queue, paramsScore, n, dim, kk);

		const scoreCode = metric === "l2" ? WGSL.scoreL2Neg : WGSL.scoreDot; // cosine uses dot on normalized vecs
		const scorePipe = this.cache.getComputePipeline({
			code: scoreCode,
			label: `score_${metric}`,
		});
		const scoreBgl = scorePipe.getBindGroupLayout(0);

		const scoreBg = this.device.createBindGroup({
			layout: scoreBgl,
			entries: [
				{ binding: 0, resource: { buffer: vecUsed } },
				{ binding: 1, resource: { buffer: queryUsed } },
				{ binding: 2, resource: { buffer: scoresBuf } },
				{ binding: 3, resource: { buffer: paramsScore } },
			],
		});

		await this.scheduler.submit((encoder) => {
			const pass = encoder.beginComputePass({
				label: `${this.labelPrefix}:passScore`,
			});
			pass.setPipeline(scorePipe);
			pass.setBindGroup(0, scoreBg);
			pass.dispatchWorkgroups(ceilDiv(n, 256));
			pass.end();
		}, "score");
		await this.scheduler.flush();

		// GPU TopK reduce: scores -> pairs -> reduce -> finalPairs
		const topK1 = this.cache.getComputePipeline({
			code: WGSL.blockTopKScores,
			label: "blockTopKScores_packed_v1",
		});

		const topKReduce = this.cache.getComputePipeline({
			code: WGSL.blockTopKPairs,
			label: "blockTopKPairs_packed_v1",
		});

		const paramsTopK = this.pool.acquire(
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			"paramsTopK",
		);

		const numBlocks1 = ceilDiv(n, 256);
		const currentN = n;
		let currentPairs = numBlocks1 * kk;

		writeParams(this.queue, paramsTopK, currentN, dim, kk);

		let pairsA = this.pool.acquire(
			currentPairs * 8,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			"pairsA",
		);

		{
			const bgl = topK1.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: scoresBuf } },
					{ binding: 1, resource: { buffer: pairsA } },
					{ binding: 2, resource: { buffer: paramsTopK } },
				],
			});

			await this.scheduler.submit((encoder) => {
				const pass = encoder.beginComputePass({
					label: `${this.labelPrefix}:passTopK1`,
				});
				pass.setPipeline(topK1);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(numBlocks1);
				pass.end();
			}, "topk1");
			await this.scheduler.flush();
		}

		while (currentPairs > 256) {
			const nextBlocks = ceilDiv(currentPairs, 256);
			const nextPairs = nextBlocks * kk;

			const pairsB = this.pool.acquire(
				nextPairs * 8,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
				"pairsB",
			);
			writeParams(this.queue, paramsTopK, currentPairs, dim, kk);

			const bgl = topKReduce.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: pairsA } },
					{ binding: 1, resource: { buffer: pairsB } },
					{ binding: 2, resource: { buffer: paramsTopK } },
				],
			});

			await this.scheduler.submit((encoder) => {
				const pass = encoder.beginComputePass({
					label: `${this.labelPrefix}:passReduce`,
				});
				pass.setPipeline(topKReduce);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(nextBlocks);
				pass.end();
			}, "reduce");
			await this.scheduler.flush();

			this.pool.release(
				pairsA,
				currentPairs * 8,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			);
			pairsA = pairsB;
			currentPairs = nextPairs;
		}

		const finalOut = this.pool.acquire(
			kk * 8,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
			"finalPairs",
		);
		writeParams(this.queue, paramsTopK, currentPairs, dim, kk);

		{
			const bgl = topKReduce.getBindGroupLayout(0);
			const bg = this.device.createBindGroup({
				layout: bgl,
				entries: [
					{ binding: 0, resource: { buffer: pairsA } },
					{ binding: 1, resource: { buffer: finalOut } },
					{ binding: 2, resource: { buffer: paramsTopK } },
				],
			});

			await this.scheduler.submit((encoder) => {
				const pass = encoder.beginComputePass({
					label: `${this.labelPrefix}:passFinal`,
				});
				pass.setPipeline(topKReduce);
				pass.setBindGroup(0, bg);
				pass.dispatchWorkgroups(1);
				pass.end();
			}, "final");
			await this.scheduler.flush();
		}

		// Read back finalOut
		const readback = this.device.createBuffer({
			size: kk * 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			label: `${this.labelPrefix}:readbackTopK`,
		});

		const enc = this.device.createCommandEncoder({
			label: `${this.labelPrefix}:encCopyTopK`,
		});
		enc.copyBufferToBuffer(finalOut, 0, readback, 0, kk * 8);
		this.queue.submit([enc.finish()]);
		await this.queue.onSubmittedWorkDone();

		await readback.mapAsync(GPUMapMode.READ);
		const raw = readback.getMappedRange();
		const dv = new DataView(raw.slice(0));
		readback.unmap();
		readback.destroy();

		let out: Pair[] = [];
		for (let i = 0; i < kk; i++) {
			const scoreBits = dv.getUint32(i * 8 + 0, true);
			const idx = dv.getUint32(i * 8 + 4, true);
			out.push({ idx, score: bitsToF32(scoreBits) });
		}
		out.sort((a, b) => (b.score - a.score) || (a.idx - b.idx));

		if (!isValidTopK(out, n, kk)) {
			const readbackScores = this.device.createBuffer({
				size: n * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
				label: `${this.labelPrefix}:readbackScores`,
			});
			const encScores = this.device.createCommandEncoder({
				label: `${this.labelPrefix}:encCopyScores`,
			});
			encScores.copyBufferToBuffer(scoresBuf, 0, readbackScores, 0, n * 4);
			this.queue.submit([encScores.finish()]);
			await this.queue.onSubmittedWorkDone();

			await readbackScores.mapAsync(GPUMapMode.READ);
			const rawScores = readbackScores.getMappedRange();
			const scores = new Float32Array(rawScores.slice(0));
			readbackScores.unmap();
			readbackScores.destroy();

			out = topkFromScoresCPU(scores, kk);
			console.warn(`[${this.labelPrefix}] GPU TopK invalid; fell back to CPU TopK for batch.`);
		}

		// Cleanup pooled buffers
		this.pool.release(
			finalOut,
			kk * 8,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		);
		this.pool.release(
			pairsA,
			currentPairs * 8,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		);
		this.pool.release(
			paramsTopK,
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		);

		this.pool.release(
			paramsScore,
			16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		);
		this.pool.release(
			scoresBuf,
			n * 4,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		);

		if (vecNorm && paramsNorm) {
			this.pool.release(vecNorm, vecBytes, GPUBufferUsage.STORAGE);
			this.pool.release(
				paramsNorm,
				16,
				GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			);
		}

		this.pool.release(
			vecBuf,
			vecBytes,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);

		return out;
	}
}
