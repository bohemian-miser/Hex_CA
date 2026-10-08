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
  /** n flood steps for every player whose flood still runs (default 1). */
  step(n?: number): void;
  /** Per board cell, 1 where the owner's flood reads filled (channel 1 > 0.5; a capped flood: its core). */
  fill(owner: number): Uint8Array;
  /** Per board cell: the owner of a line on it; else the one player whose fill covers it; else 0; -1 when two or
   * more fills cover it (contested). */
  territory(): Int8Array;
  /** Whether any flood still runs (an addition to §4.1: both layers here have it; see AreaOptions). */
  settling?(): boolean;
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

/**
 * When a player's flood runs, and what `fill` reads (both layers use this, so the CPU and GL paths agree):
 *
 * - LIVE: while the player's walls change (a line growing, a wipe), the flood runs on the walls as they come — the fill
 *   spreads as you go — and `fill` reads it as it is.
 * - REFRESHING: once the walls have not changed for `refreshPerR`·R flood steps, the flood starts again from the fresh
 *   state on the finished walls, and `fill` keeps reading the live fill as it was then. A flood that grew up with its
 *   walls is worse at near-even splits than a fresh one — it can fill every side (docs/spectacle-ca-hybrid.md §3.5) —
 *   and fresh is what was measured (§3.6).
 * - SETTLED: the fresh flood's thresholded fill (channel 1 > 0.5) has not changed for `settlePerR`·R steps. It sleeps
 *   (not stepped) and `fill` reads that fill. The state itself never reaches an exact fixed point (every cell drifts a
 *   little every step), so "settled" is about the readout.
 * - CAPPED: not settled `capPerR`·R steps into the fresh run (a flood swinging between answers). It sleeps too, and
 *   `fill` reads the conservative core: the cells it filled at every look from `settlePerR`·R steps into the run on.
 *
 * Any change to the player's walls makes it live again. The fill is looked at every `checkEvery` flood steps of that
 * player (on the GPU a look is a `readPixels`). `refreshPerR` 0: no re-flood (the live flood settles and caps instead);
 * `settlePerR` and `capPerR` 0: that rule is off; all three 0: the bare flood, every step.
 */
export interface AreaOptions {
  settlePerR: number;
  capPerR: number;
  checkEvery: number;
  refreshPerR: number;
}
export const DEFAULT_AREA_OPTIONS: AreaOptions = { settlePerR: 8, capPerR: 64, checkEvery: 4, refreshPerR: 2 };

export type FloodStatus = 'dormant' | 'live' | 'refreshing' | 'settled' | 'capped';

/** One player's flood as the rules of AreaOptions see it. */
export class Settle {
  /** Flood steps since the player's walls last changed. */
  sinceWall = 0;
  /** Flood steps since the run being judged began: the wall change, or the fresh re-flood. */
  sinceRun = 0;
  /** Flood steps the thresholded fill has stayed the same (counted at looks). */
  quiet = 0;
  /** Re-flooding from the fresh state. */
  fresh = false;
  asleep = false;
  capped = false;
  /** What `fill` reads while asleep or re-flooding. */
  frozen: Uint8Array | null = null;
  private readonly last: Uint8Array;
  private core: Uint8Array | null = null;

  constructor(n: number, readonly settleSteps: number, readonly capSteps: number, readonly refreshSteps: number,
    readonly every: number) {
    this.last = new Uint8Array(n);
  }

  /** Whether any rule needs looks at the fill. */
  get looks(): boolean {
    return this.settleSteps > 0 || this.capSteps > 0 || this.refreshSteps > 0;
  }

  /** The walls changed: live, counting afresh (the fill before the change is what "no change" is measured from). */
  wake(): void {
    this.fresh = false;
    this.asleep = false;
    this.capped = false;
    this.frozen = null;
    this.sinceWall = 0;
    this.sinceRun = 0;
    this.quiet = 0;
    this.core = null;
  }

  /** The fresh state (no walls). */
  clear(): void {
    this.wake();
    this.last.fill(0);
  }

  /** Flood steps to the next look (at least 1). */
  untilLook(): number {
    return this.every - (this.sinceWall % this.every);
  }

  /** k steps were run; whether a look is due now. */
  advanced(k: number): boolean {
    this.sinceWall += k;
    this.sinceRun += k;
    return this.sinceWall % this.every === 0;
  }

  /** A look at the thresholded fill. True when the re-flood begins now: the caller resets the flood. */
  look(fill: Uint8Array): boolean {
    if (this.refreshSteps > 0 && !this.fresh) {
      if (this.sinceWall < this.refreshSteps) return false; // live: the walls may still be coming
      this.fresh = true;
      this.frozen = fill.slice();
      this.sinceRun = 0;
      this.quiet = 0;
      this.last.fill(0); // the fresh state fills nothing
      this.core = null;
      return true;
    }
    let same = true;
    for (let i = 0; i < fill.length; i++) if (fill[i] !== this.last[i]) { same = false; break; }
    if (same) this.quiet += this.every;
    else {
      this.quiet = 0;
      this.last.set(fill);
    }
    if (this.sinceRun >= this.settleSteps) {
      if (!this.core) this.core = fill.slice();
      else for (let i = 0; i < fill.length; i++) this.core[i] &= fill[i];
    }
    if (this.settleSteps > 0 && this.quiet >= this.settleSteps) {
      this.asleep = true;
      this.frozen = this.last.slice();
    } else if (this.capSteps > 0 && this.sinceRun >= this.capSteps) {
      this.asleep = true;
      this.capped = true;
      this.frozen = (this.core ?? fill).slice();
    }
    return false;
  }
}

/**
 * What the CPU and GPU layers share: walls from the line channels, dormant and sleeping floods (AreaOptions), fill
 * and territory. A player with no wall at all is dormant: their flood is reset (the fresh state, which fills nothing
 * — the oracle's answer for no walls) and not stepped; it starts from the fresh state when their first wall lands,
 * exactly as in training. Subclasses run the floods.
 */
export abstract class FloodLayer implements AreaLayer {
  readonly frame: AreaFrame;
  readonly options: AreaOptions;
  /** Per board cell, the owner of the line on it (0 none), as of the last `update`. */
  readonly lineOwner: Int32Array;
  /** Per player (index p − 1), how many board cells are their walls. */
  readonly wallCount: Int32Array;
  /** Per player (index p − 1), the settle rule's view of their flood. */
  readonly settle: Settle[];

  constructor(readonly board: Board, readonly owners: number, options: Partial<AreaOptions> = {}) {
    if (!Number.isInteger(owners) || owners < 1 || owners > 15) throw new RangeError(`area: owners must be 1..15, got ${owners}`);
    this.frame = areaFrame(board);
    this.options = { ...DEFAULT_AREA_OPTIONS, ...options };
    const { settlePerR, capPerR, checkEvery, refreshPerR } = this.options;
    if (!Number.isInteger(checkEvery) || checkEvery < 1) throw new RangeError(`area: checkEvery must be an integer ≥ 1`);
    this.lineOwner = new Int32Array(board.n);
    this.wallCount = new Int32Array(owners);
    const R = this.frame.R;
    this.settle = Array.from({ length: owners }, () => new Settle(board.n, Math.round(settlePerR * R), Math.round(capPerR * R),
      Math.round(refreshPerR * R), checkEvery));
  }

  /** Paints or erases player p's wall at slot s; whether it changed. */
  protected abstract setWallAt(p: number, s: number, v: 0 | 1): boolean;
  /** Player p's flood back to the fresh state. */
  protected abstract resetFlood(p: number): void;
  /** k flood steps of player p. */
  protected abstract runFlood(p: number, k: number): void;
  /** Player p's thresholded fill now, per board cell. */
  protected abstract readFill(p: number): Uint8Array;

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
    if (!this.setWallAt(p, s, v)) return;
    this.wallCount[p - 1] += v ? 1 : -1;
    if (this.wallCount[p - 1] === 0) {
      this.resetFlood(p); // dormant: the fresh state (no walls) fills nothing
      this.settle[p - 1].clear();
    } else this.settle[p - 1].wake();
  }

  step(n = 1): void {
    for (let p = 1; p <= this.owners; p++) {
      const st = this.settle[p - 1];
      if (this.wallCount[p - 1] === 0 || st.asleep) continue;
      if (!st.looks) { // no rule: no looks either
        this.runFlood(p, n);
        st.advanced(n);
        continue;
      }
      let left = n;
      while (left > 0 && !st.asleep) {
        const k = Math.min(left, st.untilLook());
        this.runFlood(p, k);
        left -= k;
        if (st.advanced(k) && st.look(this.readFill(p))) this.resetFlood(p);
      }
    }
  }

  /** Whether any player's flood still runs (has walls and is neither settled nor capped). */
  settling(): boolean {
    for (let p = 0; p < this.owners; p++) if (this.wallCount[p] > 0 && !this.settle[p].asleep) return true;
    return false;
  }

  status(owner: number): FloodStatus {
    const st = this.settle[owner - 1];
    if (this.wallCount[owner - 1] === 0) return 'dormant';
    return st.asleep ? (st.capped ? 'capped' : 'settled') : st.fresh ? 'refreshing' : 'live';
  }

  fill(owner: number): Uint8Array {
    if (!Number.isInteger(owner) || owner < 1 || owner > this.owners) throw new RangeError(`area: no owner ${owner}`);
    if (this.wallCount[owner - 1] === 0) return new Uint8Array(this.board.n);
    const st = this.settle[owner - 1];
    return st.frozen ? st.frozen.slice() : this.readFill(owner);
  }

  territory(): Int8Array {
    return territoryOf(this.lineOwner, (p) => this.fill(p), this.owners);
  }
}

/** The flood layer on the CPU: K `HexNCA`s, scalar JS. */
export class CpuArea extends FloodLayer {
  readonly ncas: HexNCA[];

  constructor(board: Board, weights: NCAWeights, owners: number, options: Partial<AreaOptions> = {}) {
    super(board, owners, options);
    this.ncas = [];
    for (let p = 0; p < owners; p++) {
      const nca = new HexNCA(weights, this.frame.R, this.frame.mask);
      nca.reset();
      this.ncas.push(nca);
    }
  }

  protected setWallAt(p: number, s: number, v: 0 | 1): boolean {
    return this.ncas[p - 1].setWall(s, v);
  }

  protected resetFlood(p: number): void {
    this.ncas[p - 1].reset();
  }

  protected runFlood(p: number, k: number): void {
    this.ncas[p - 1].step(k);
  }

  protected readFill(p: number): Uint8Array {
    const out = new Uint8Array(this.board.n);
    const ch1 = this.ncas[p - 1].channel(1);
    const { slot } = this.frame;
    for (let i = 0; i < out.length; i++) out[i] = ch1[slot[i]] > 0.5 ? 1 : 0;
    return out;
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

export function cpuArea(board: Board, weights: NCAWeights, owners: number, options: Partial<AreaOptions> = {}): AreaLayer {
  return new CpuArea(board, weights, owners, options);
}
