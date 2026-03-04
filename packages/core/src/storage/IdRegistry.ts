export class IdRegistry {
  private extToInt = new Map<string, number>();
  private intToExt: (string | null)[] = [];

  hasExternal(id: string): boolean {
    return this.extToInt.has(id);
  }

  getInternal(id: string): number | null {
    const v = this.extToInt.get(id);
    return v === undefined ? null : v;
  }

  getExternal(internalId: number): string | null {
    const v = this.intToExt[internalId];
    return v ?? null;
  }

  bind(externalId: string, internalId: number): void {
    this.extToInt.set(externalId, internalId);
    this.intToExt[internalId] = externalId;
  }

  unbind(externalId: string): number | null {
    const internal = this.extToInt.get(externalId);
    if (internal === undefined) return null;

    this.extToInt.delete(externalId);
    this.intToExt[internal] = null;
    return internal;
  }

  size(): number {
    return this.extToInt.size;
  }

  exportIntToExt(): Array<string | null> {
    return this.intToExt.slice(0);
  }

  restoreFromIntToExt(intToExt: Array<string | null>) {
    this.intToExt = intToExt.slice(0);
    this.extToInt = new Map<string, number>();
    for (let i = 0; i < this.intToExt.length; i++) {
      const ext = this.intToExt[i];
      if (ext) this.extToInt.set(ext, i);
    }
  }

  compactToMaxId(maxIdExclusive: number) {
    if (maxIdExclusive < 0) maxIdExclusive = 0;
    if (this.intToExt.length <= maxIdExclusive) return;
    this.intToExt = this.intToExt.slice(0, maxIdExclusive);
    this.extToInt = new Map<string, number>();
    for (let i = 0; i < this.intToExt.length; i++) {
      const ext = this.intToExt[i];
      if (ext) this.extToInt.set(ext, i);
    }
  }
}
