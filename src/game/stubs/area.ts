// A STAND-IN for src/game/area.ts (agent B's area layer, docs/spectacle-ca-hybrid.md §3) with the frozen surface
// of §4.1: one HexNCA per player on the strand board placed in the NCA's S×S frame (§3.1), walls = that player's
// line tiles. Replaced by the real thing at integration.

import type { Board } from '../../strand.js';
import { HexNCA, type NCAWeights } from '../../nca.js';

export interface AreaLayer {
  readonly owners: number;
  update(lines: { ch: { owner: Int32Array; D: Int32Array } }, changed: Int32Array): void;
  step(n?: number): void;
  fill(owner: number): Uint8Array;
  territory(): Int8Array;
}

export class CpuArea implements AreaLayer {
  readonly S: number;
  readonly slot: Int32Array;
  readonly ncas: HexNCA[] = [];
  private readonly lineOwner: Int32Array;
  private readonly walls: Int32Array;

  constructor(readonly board: Board, weights: NCAWeights, readonly owners: number) {
    let S = Math.max(board.h, board.w) + 2;
    if (S % 2 === 0) S++;
    this.S = S;
    this.slot = new Int32Array(board.n);
    const mask = new Uint8Array(S * S);
    for (let i = 0; i < board.n; i++) {
      this.slot[i] = (Math.floor(board.pos[i] / board.w) + 1) * S + (board.pos[i] % board.w) + 1;
      mask[this.slot[i]] = 1;
    }
    for (let p = 0; p < owners; p++) {
      const m = new HexNCA(weights, (S - 1) / 2, mask);
      m.reset();
      this.ncas.push(m);
    }
    this.lineOwner = new Int32Array(board.n);
    this.walls = new Int32Array(owners);
  }

  update(lines: { ch: { owner: Int32Array; D: Int32Array } }, changed: Int32Array): void {
    for (const c of changed) {
      const o = lines.ch.D[c] ? lines.ch.owner[c] : 0;
      const was = this.lineOwner[c];
      if (o === was) continue;
      this.lineOwner[c] = o;
      if (was && this.ncas[was - 1].setWall(this.slot[c], 0) && --this.walls[was - 1] === 0) this.ncas[was - 1].reset();
      if (o && this.ncas[o - 1].setWall(this.slot[c], 1)) this.walls[o - 1]++;
    }
  }

  step(n = 1): void {
    for (let p = 0; p < this.owners; p++) if (this.walls[p] > 0) this.ncas[p].step(n);
  }

  fill(owner: number): Uint8Array {
    const out = new Uint8Array(this.board.n);
    if (!this.walls[owner - 1]) return out;
    const ch1 = this.ncas[owner - 1].channel(1);
    for (let i = 0; i < out.length; i++) out[i] = ch1[this.slot[i]] > 0.5 ? 1 : 0;
    return out;
  }

  territory(): Int8Array {
    const out = new Int8Array(this.board.n);
    for (let p = 1; p <= this.owners; p++) {
      const f = this.fill(p);
      for (let i = 0; i < out.length; i++) if (f[i]) out[i] = out[i] === 0 || out[i] === p ? p : -1;
    }
    for (let i = 0; i < out.length; i++) if (this.lineOwner[i]) out[i] = this.lineOwner[i];
    return out;
  }
}

export function cpuArea(board: Board, weights: NCAWeights, owners: number): AreaLayer {
  return new CpuArea(board, weights, owners);
}
