export class DeviceManager {
  constructor(_: any) {}
  async init() {}
}

export class ChunkScannerGPU {
  constructor(_: any, __?: string) {}
  async open() {}
  async prepareQuery(query: Float32Array) {
    return { queryUsed: query, release() {} };
  }
  async topKBatch() {
    return [];
  }
}

export function isWebGPUAvailable() {
  return false;
}

