// Area control for the hybrid game (docs/spectacle-ca-hybrid.md §3): the trained flood fill (src/nca.ts) with each
// player's line tiles as its walls. One HexNCA per player over one shared set of weights; rival lines are
// transparent to each (Spectacle's "other lines ignored"). The fill of player p is everything p's walls enclose,
// plus every rim region but the largest — for one rim-to-rim claim, its smaller side (§3.3).
//
// The frame (§3.1): the strand Board's (row, col) is axial with the same six neighbour vectors as the NCA's S×S
// array, so a board cell goes to slot (row + 1)·S + col + 1 with S = max(h, w) + 2 rounded up to odd (a one-cell
// margin all round) and R = (S − 1) / 2; the mask is the board. The flood sees the board's own outline as its rim,
// as it was trained on ragged masks.

import type { Board } from '../strand.js';
import { HexNCA, type NCAWeights } from '../nca.js';

/** What `update` reads of the line CA (src/game/line-ca.ts's `LineCA.ch`, §4.1): per board cell, the owner and
 * the drawn chord ends `D` (6 bits, 0 = no chord). */
export interface LineChannels {
  readonly ch: { readonly owner: Int32Array; readonly D: Int32Array };
}

export interface AreaLayer {
  /** Players 1..owners each have a flood. */
  readonly owners: number;
  /** Re-derives walls_p (owner_c = p and a chord on c) for the changed cells. */
  update(lines: LineChannels, changed: Int32Array): void;
  /** n flood steps for every player (default 1). */
  step(n?: number): void;
  /** Per board cell, 1 where the owner's flood reads filled (channel 1 > 0.5). */
  fill(owner: number): Uint8Array;
  /** Per board cell: the owner of a line on it; else the one player whose fill covers it; else 0; -1 when two or
   * more fills cover it (contested). */
  territory(): Int8Array;
}

/** The strand Board placed in the NCA's S×S array (§3.1). */
export interface AreaFrame {
  readonly S: number;
  readonly R: number;
  /** slot[i] = (row + 1)·S + col + 1 for board cell i. */
  readonly slot: Int32Array;
  /** 1 at every board cell's slot, 0 elsewhere (S² long). */
  readonly mask: Uint8Array;
}

export function areaFrame(board: Board): AreaFrame {
  let S = Math.max(board.h, board.w) + 2;
  if (S % 2 === 0) S++;
  const slot = new Int32Array(board.n);
  const mask = new Uint8Array(S * S);
  for (let i = 0; i < board.n; i++) {
    const row = Math.floor(board.pos[i] / board.w);
    const col = board.pos[i] % board.w;
    slot[i] = (row + 1) * S + col + 1;
    mask[slot[i]] = 1;
  }
  return { S, R: (S - 1) / 2, slot, mask };
}

/** The flood layer on the CPU: K `HexNCA`s, scalar JS. A player with no wall at all is dormant: their flood is
 * reset (the fresh state, which fills nothing — the oracle's answer for no walls) and not stepped, so idle seats
 * cost nothing; it starts from the fresh state when their first wall lands, exactly as in training. */
export class CpuArea implements AreaLayer {
  readonly frame: AreaFrame;
  readonly ncas: HexNCA[];
  /** Per board cell, the owner of the line on it (0 none), as of the last `update`. */
  readonly lineOwner: Int32Array;
  /** Per player (index p − 1), how many board cells are their walls. */
  readonly wallCount: Int32Array;

  constructor(readonly board: Board, weights: NCAWeights, readonly owners: number) {
    if (!Number.isInteger(owners) || owners < 1 || owners > 15) throw new RangeError(`area: owners must be 1..15, got ${owners}`);
    this.frame = areaFrame(board);
    this.ncas = [];
    for (let p = 0; p < owners; p++) {
      const nca = new HexNCA(weights, this.frame.R, this.frame.mask);
      nca.reset();
      this.ncas.push(nca);
    }
    this.lineOwner = new Int32Array(board.n);
    this.wallCount = new Int32Array(owners);
  }

  update(lines: LineChannels, changed: Int32Array): void {
    const { owner, D } = lines.ch;
    const { slot } = this.frame;
    for (let k = 0; k < changed.length; k++) {
      const c = changed[k];
      const o = D[c] !== 0 ? owner[c] : 0;
      if (o < 0 || o > this.owners) throw new RangeError(`area: owner ${o} on cell ${c}, but the layer has ${this.owners}`);
      const was = this.lineOwner[c];
      if (o === was) continue;
      this.lineOwner[c] = o;
      if (was) this.setWall(was, slot[c], 0);
      if (o) this.setWall(o, slot[c], 1);
    }
  }

  private setWall(p: number, s: number, v: 0 | 1): void {
    const nca = this.ncas[p - 1];
    if (!nca.setWall(s, v)) return;
    this.wallCount[p - 1] += v ? 1 : -1;
    if (this.wallCount[p - 1] === 0) nca.reset(); // dormant: the fresh state (no walls) fills nothing
  }

  step(n = 1): void {
    for (let p = 0; p < this.owners; p++) if (this.wallCount[p] > 0) this.ncas[p].step(n);
  }

  fill(owner: number): Uint8Array {
    if (!Number.isInteger(owner) || owner < 1 || owner > this.owners) throw new RangeError(`area: no owner ${owner}`);
    const out = new Uint8Array(this.board.n);
    if (this.wallCount[owner - 1] === 0) return out;
    const ch1 = this.ncas[owner - 1].channel(1);
    const { slot } = this.frame;
    for (let i = 0; i < out.length; i++) out[i] = ch1[slot[i]] > 0.5 ? 1 : 0;
    return out;
  }

  territory(): Int8Array {
    return territoryOf(this.lineOwner, (p) => this.fill(p), this.owners);
  }
}

/** territory() from the line owners and the fills (shared with the GPU layer). */
export function territoryOf(lineOwner: Int32Array, fill: (owner: number) => Uint8Array, owners: number): Int8Array {
  const out = new Int8Array(lineOwner.length);
  for (let p = 1; p <= owners; p++) {
    const f = fill(p);
    for (let i = 0; i < out.length; i++) {
      if (!f[i]) continue;
      out[i] = out[i] === 0 ? p : out[i] === p ? p : -1;
    }
  }
  for (let i = 0; i < out.length; i++) if (lineOwner[i]) out[i] = lineOwner[i];
  return out;
}

export function cpuArea(board: Board, weights: NCAWeights, owners: number): AreaLayer {
  return new CpuArea(board, weights, owners);
}
