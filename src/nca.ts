// The trained hex neural cellular automaton (nca/, the spec's "Model"), run in
// the browser: pure TypeScript, no DOM.
//
// Board: the hexagon of radius R in axial (q, r), stored in an S×S array,
// S = 2R + 1, cell (q, r) at [row = r + R][col = q + R]. Off-board slots exist
// but are dead: every channel is 0 there, always. Neighbours as (drow, dcol):
// (0,+1) (0,-1) (+1,0) (-1,0) (-1,+1) (+1,-1). This is NOT src/hex.ts's layout
// (that one pads a ring); it is the Python side's, nca/hexgrid.py.
//
// The board need not be the full hexagon: HexNCA takes an optional custom mask (the same
// S×S array, see `fieldMask`) in its place — a ragged shape, such as a crop of Spectacle's
// own hex field (web/nca.ts's map selector). "Off board" then means off that shape, not
// off the hexagon; the model was trained on ragged masks too (nca/masks.py), so this needs
// no new input — a cell's rim-ness already falls out of which of its 7 taps are on-mask.
//
// One step, every on-board cell at once (synchronous, double-buffered):
//   x = concat(state, consts)                C + K channels (K consts: version 1
//                                             is just "mask"; version 2 and 3 are
//                                             "mask", "theta1", "theta2" — spec v3's
//                                             angle round the board centre)
//   pool                                      version 3 only ("taps+pool", spec v4
//                                             §1): per state channel, the max and the
//                                             min over the on-board taps among the 7
//                                             (self + 6 neighbours; off-board taps
//                                             excluded, not zero; corners never taps)
//   h = relu(conv3x3_hexmasked(x) + w1pool·pool + b1)   C+K -> H, 7 taps (corners
//                                             k=0, 8 are zero)
//   d = conv1x1(h) + b2                      H -> C
//   state = state + d · fire                 fire = 1 when fireRate is 1
//   state = clamp(state, lo, hi)             when the export has a clamp
//   state[ch0] = wall; state *= mask
//
// Every const is the same at every step (it depends only on the board, not the
// state), so its whole contribution — every tap, every const channel — folds
// into a per-cell bias once, at construction; only the state channels (taps and,
// in version 3, their per-channel pools) are gathered fresh out of `src` on
// every tap (see HexNCA's `bias` and `w1r`).

import { coordsOf, indexOf, makeBoard } from './hex.js';
import { randomLine } from './lines.js';

/** What the weights were trained on (nca/export.py writes it); every field optional. */
export interface NCAMeta {
  /** One radius, or the radii of mixed-size training. */
  trainedR?: number | number[];
  steps?: [number, number];
  iterations?: number;
  /** Trained with the persistent pool and wall edits: the state may be kept across edits. */
  pool?: boolean;
  note?: string;
  [key: string]: unknown;
}

export interface NCAWeights {
  /** 1 (consts implied ["mask"]), 2 (consts carried explicitly, spec v3), or 3 (+ w1pool, spec v4). */
  version: number;
  channels: number;
  hidden: number;
  /** The constant input channels, in the order baked into w1's extra columns. Version 1: ["mask"]. */
  consts: string[];
  /** 'taps' (versions 1–2) or 'taps+pool' (version 3, spec v4 §1: `w1pool` on top of the tap convolution). */
  perception: 'taps' | 'taps+pool';
  clamp: [number, number] | null;
  fireRate: number;
  /**
   * H × (C + K) × 9, index ((h·(C+K)) + c)·9 + k, k = (drow+1)·3 + (dcol+1);
   * c < C is a state channel, c >= C is the (c − C)-th const (`consts` order).
   */
  w1: Float32Array;
  /** Version 3 only: H × 2C, index h·2C + j; j < C is the max of state channel j over the on-board taps, j >= C the min of channel j − C. */
  w1pool?: Float32Array;
  b1: Float32Array;
  /** C × H, index o·H + h. */
  w2: Float32Array;
  b2: Float32Array;
  meta: NCAMeta;
}

/** The six neighbours as (drow, dcol), in the spec's order. */
export const NEIGHBOURS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [0, -1], [1, 0], [-1, 0], [-1, 1], [1, -1],
];

/** The 7 taps of the hex kernel: self and the six neighbours, as kernel index k = (drow+1)·3 + (dcol+1). */
const TAPS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0],
];
const tapK = ([dr, dc]: readonly [number, number]) => (dr + 1) * 3 + (dc + 1);
/** Index of the self tap ([0, 0]) within TAPS: always on board, so a cell's pool always has at least this one value. */
const SELF_TAP = TAPS.findIndex(([dr, dc]) => dr === 0 && dc === 0);

export const side = (R: number): number => 2 * R + 1;

export const hexDist = (q: number, r: number): number => Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r));

/** Array index of axial (q, r) on a radius-R board. */
export const cellIndex = (R: number, q: number, r: number): number => (r + R) * side(R) + (q + R);

/** Axial (q, r) of array index i on a radius-R board. */
export function cellCoords(R: number, i: number): [number, number] {
  const S = side(R);
  return [(i % S) - R, Math.floor(i / S) - R];
}

/** 1 on the board, 0 off it, per array slot. */
export function boardMask(R: number): Uint8Array {
  const S = side(R);
  const m = new Uint8Array(S * S);
  for (let row = 0; row < S; row++) {
    for (let col = 0; col < S; col++) m[row * S + col] = hexDist(col - R, row - R) <= R ? 1 : 0;
  }
  return m;
}

/**
 * A ragged board shape from a list of axial (q, r) cells (e.g. Spectacle's own hex fields,
 * web/nca.ts's map selector): R is one past the farthest cell's hex distance from the
 * centre, so every listed cell lands strictly inside the S×S array with room for at least
 * one off-board neighbour all round (the model's rim input). `mask` is 1 at each listed
 * cell, 0 everywhere else in the array — HexNCA takes it as a custom board shape in place
 * of the default full hexagon.
 */
export function fieldMask(qr: ReadonlyArray<readonly [number, number]>): { R: number; mask: Uint8Array } {
  let maxD = 0;
  for (const [q, r] of qr) maxD = Math.max(maxD, hexDist(q, r));
  const R = Math.max(1, maxD + 1);
  const S = side(R);
  const mask = new Uint8Array(S * S);
  for (const [q, r] of qr) mask[cellIndex(R, q, r)] = 1;
  return { R, mask };
}

// ── Weights ─────────────────────────────────────────────────────────────────

function floats(j: Record<string, unknown>, key: string, n: number): Float32Array {
  const a = j[key];
  if (!Array.isArray(a)) throw new Error(`nca weights: ${key} is not an array`);
  if (a.length !== n) throw new Error(`nca weights: ${key} has ${a.length} values, expected ${n}`);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = a[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`nca weights: ${key}[${i}] is not a finite number`);
    out[i] = v;
  }
  return out;
}

function posInt(j: Record<string, unknown>, key: string, min: number): number {
  const v = j[key];
  if (typeof v !== 'number' || !Number.isInteger(v) || v < min) throw new Error(`nca weights: ${key} must be an integer >= ${min}`);
  return v;
}

/** The const inputs this reader knows how to build (see `buildConst`). */
const KNOWN_CONSTS = new Set(['mask', 'theta1', 'theta2', 'src1', 'src1c', 'src2', 'src2c']);

/** The exported JSON (web/nca-weights.json) as typed arrays, checked against the spec's format. */
export function loadWeights(json: unknown): NCAWeights {
  if (typeof json !== 'object' || json === null || Array.isArray(json)) throw new Error('nca weights: not an object');
  const j = json as Record<string, unknown>;
  const version = j.version;
  if (version !== 1 && version !== 2 && version !== 3) throw new Error(`nca weights: unsupported version ${String(version)}`);
  let consts: string[];
  if (version === 1) {
    consts = ['mask']; // v1 files don't carry a `consts` field: the mask is implied.
  } else {
    const rawConsts = j.consts;
    if (!Array.isArray(rawConsts) || rawConsts.length === 0 || !rawConsts.every((v) => typeof v === 'string')) {
      throw new Error('nca weights: consts must be a non-empty array of strings');
    }
    for (const name of rawConsts) if (!KNOWN_CONSTS.has(name)) throw new Error(`nca weights: unknown const '${name}'`);
    consts = rawConsts as string[];
  }
  const perception: 'taps' | 'taps+pool' = version === 3 ? 'taps+pool' : 'taps';
  if (version === 3 && j.perception !== 'taps+pool') {
    throw new Error(`nca weights: version 3 requires perception "taps+pool", got ${JSON.stringify(j.perception)}`);
  }
  const C = posInt(j, 'channels', 2);
  const H = posInt(j, 'hidden', 1);
  let clamp: [number, number] | null = null;
  if (j.clamp !== null && j.clamp !== undefined) {
    const cl = j.clamp;
    if (!Array.isArray(cl) || cl.length !== 2 || !cl.every((v) => typeof v === 'number' && Number.isFinite(v)) || !(cl[0] < cl[1])) {
      throw new Error('nca weights: clamp must be null or [lo, hi] with lo < hi');
    }
    clamp = [cl[0], cl[1]];
  }
  const fireRate = j.fireRate ?? 1;
  if (typeof fireRate !== 'number' || !(fireRate > 0 && fireRate <= 1)) throw new Error('nca weights: fireRate must be in (0, 1]');
  const K = consts.length;
  const w1 = floats(j, 'w1', H * (C + K) * 9);
  for (let i = 0; i < w1.length; i += 9) {
    if (w1[i] !== 0 || w1[i + 8] !== 0) throw new Error('nca weights: a corner tap (k = 0 or 8) is not zero');
  }
  const w1pool = version === 3 ? floats(j, 'w1pool', H * 2 * C) : undefined;
  const meta = j.meta;
  if (meta !== undefined && (typeof meta !== 'object' || meta === null || Array.isArray(meta))) {
    throw new Error('nca weights: meta must be an object');
  }
  return {
    version, channels: C, hidden: H, consts, perception, clamp, fireRate,
    w1, w1pool, b1: floats(j, 'b1', H), w2: floats(j, 'w2', C * H), b2: floats(j, 'b2', C),
    meta: (meta ?? {}) as NCAMeta,
  };
}

/**
 * One constant input's values over a radius-R board's S×S array (spec v3,
 * src* added in v6 §1): "mask" is 1 on board else 0; "theta1"/"theta2" encode
 * a cell's angle round the board centre (pointy-top axial → Cartesian, atan2
 * in double precision, stored float32; theta2 = theta1 + 0.5 mod 1).
 * "src1"/"src1c"/"src2"/"src2c" gate theta1/theta2 (and its complement) by
 * rim(cell) = 1 iff on board and hex distance == R: src1 = rim·theta1,
 * src1c = rim·(1 − theta1), src2 = rim·theta2, src2c = rim·(1 − theta2) — all
 * computed in double, stored float32. Off-board slots are 0 for every const;
 * the src* consts are additionally 0 on-board off the rim.
 */
export function buildConst(name: string, R: number): Float32Array {
  const S = side(R);
  const N = S * S;
  const out = new Float32Array(N);
  const mask = boardMask(R);
  if (name === 'mask') {
    for (let i = 0; i < N; i++) out[i] = mask[i];
    return out;
  }
  if (name === 'theta1' || name === 'theta2' || name === 'src1' || name === 'src1c' || name === 'src2' || name === 'src2c') {
    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const i = row * S + col;
        if (!mask[i]) continue; // stays 0
        const q = col - R;
        const r = row - R;
        const x = Math.sqrt(3) * (q + r / 2);
        const y = 1.5 * r;
        const theta1 = (Math.atan2(y, x) / (2 * Math.PI) + 1) % 1;
        if (name === 'theta1') { out[i] = theta1; continue; }
        const theta2 = (theta1 + 0.5) % 1;
        if (name === 'theta2') { out[i] = theta2; continue; }
        if (hexDist(q, r) !== R) continue; // off the rim: stays 0
        if (name === 'src1') out[i] = theta1;
        else if (name === 'src1c') out[i] = 1 - theta1;
        else if (name === 'src2') out[i] = theta2;
        else out[i] = 1 - theta2; // src2c
      }
    }
    return out;
  }
  throw new Error(`nca weights: unknown const '${name}'`);
}

// ── The automaton ───────────────────────────────────────────────────────────

export class HexNCA {
  readonly R: number;
  readonly S: number;
  /** Slots per channel, S². */
  readonly N: number;
  readonly channels: number;
  readonly hidden: number;
  /** 1 on the board, 0 off it. */
  readonly mask: Uint8Array;
  /** The on-board slots, row by row. */
  readonly cells: Int32Array;
  /** The wall picture, 0/1 per slot (always 0 off the board). */
  readonly walls: Uint8Array;
  /** The const inputs (`weights.consts` order), each S² floats, 0 off the board. */
  readonly consts: Float32Array[];
  /** C × S² floats, channel-major: channel c of slot i at c·N + i. */
  state: Float32Array;
  /** Steps since the last reset. */
  steps = 0;

  private next: Float32Array;
  private readonly clamp: [number, number] | null;
  private readonly fireRate: number;
  private readonly rand: () => number;
  /** Per cell, the 7 taps' slots (−1 off the array or off the board: those read zero). */
  private readonly taps: Int32Array;
  /** Per cell and hidden unit: b1 plus every const channel's taps, which never change. */
  private readonly bias: Float32Array;
  /** 2C when `perception` is 'taps+pool' (max then min per state channel), else 0. */
  private readonly poolWidth: number;
  /**
   * w1 (and, when `poolWidth` > 0, w1pool) without the const channels, eight hidden units
   * interleaved: [h / 8][feature][h % 8], H padded to a multiple of 8 with zero units (doubles:
   * faster to read here). Feature index f < T·C is tap t = f / C, channel c = f % C; f >= T·C is
   * the pool feature f − T·C (max of channel j for f − T·C = j < C, min of channel j for
   * f − T·C = C + j).
   */
  private readonly w1r: Float64Array;
  /** w2 transposed, [h][o]. */
  private readonly w2t: Float32Array;
  private readonly b2: Float32Array;
  /** T·C tap features, then (when pooling) 2C pool features: `w1r`'s feature layout. */
  private readonly x: Float64Array;
  /** The hidden units' dot products for one cell, before bias and relu. */
  private readonly z: Float64Array;
  private readonly d: Float64Array;
  /** Per-channel max/min scratch for the pool features (length 0 when `perception` is 'taps'). */
  private readonly pmax: Float64Array;
  private readonly pmin: Float64Array;

  /**
   * `mask`, when given, replaces the default full hexagon as the board shape (same S×S
   * layout, 1 on board else 0 — see `fieldMask`): a ragged field, not just a hexagon of
   * radius R. It must be S² long. `rand` is only drawn on when fireRate < 1 (each cell
   * fires with that chance, all channels together).
   */
  constructor(w: NCAWeights, R: number, mask?: Uint8Array, rand: () => number = Math.random) {
    if (!Number.isInteger(R) || R < 1) throw new Error('HexNCA: R must be an integer >= 1');
    const C = w.channels;
    const H = w.hidden;
    const S = side(R);
    const N = S * S;
    if (mask && mask.length !== N) throw new Error(`HexNCA: mask has ${mask.length} slots, expected ${N} (S=${S})`);
    this.R = R;
    this.S = S;
    this.N = N;
    this.channels = C;
    this.hidden = H;
    this.clamp = w.clamp;
    this.fireRate = w.fireRate;
    this.rand = rand;
    this.mask = mask ? Uint8Array.from(mask) : boardMask(R);
    this.walls = new Uint8Array(N);
    // The "mask" const input is this board's own shape, not necessarily the full hexagon
    // (buildConst('mask', R) would be); every other const (none in the shipped weights)
    // is positional and shape-independent, so it still comes from buildConst.
    this.consts = w.consts.map((name) => (name === 'mask' ? Float32Array.from(this.mask) : buildConst(name, R)));
    this.state = new Float32Array(C * N);
    this.next = new Float32Array(C * N);

    const cells: number[] = [];
    for (let i = 0; i < N; i++) if (this.mask[i]) cells.push(i);
    this.cells = Int32Array.from(cells);
    const M = cells.length;
    const T = TAPS.length;

    this.taps = new Int32Array(M * T);
    for (let ci = 0; ci < M; ci++) {
      const row = Math.floor(cells[ci] / S);
      const col = cells[ci] % S;
      for (let t = 0; t < T; t++) {
        const r2 = row + TAPS[t][0];
        const c2 = col + TAPS[t][1];
        const j = r2 * S + c2;
        this.taps[ci * T + t] = r2 >= 0 && r2 < S && c2 >= 0 && c2 < S && this.mask[j] ? j : -1;
      }
    }

    const K = w.consts.length;
    const CK = C + K;
    const H8 = Math.ceil(H / 8) * 8;
    const TC = T * C;
    const PW = this.poolWidth = w.perception === 'taps+pool' ? 2 * C : 0;
    const W = TC + PW;
    this.w1r = new Float64Array(H8 * W);
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < T; t++) {
        const k = tapK(TAPS[t]);
        for (let c = 0; c < C; c++) this.w1r[((h >> 3) * W + t * C + c) * 8 + (h & 7)] = w.w1[(h * CK + c) * 9 + k];
      }
      if (PW > 0) {
        const w1pool = w.w1pool!;
        for (let f = 0; f < PW; f++) this.w1r[((h >> 3) * W + TC + f) * 8 + (h & 7)] = w1pool[h * PW + f];
      }
    }
    // Every const (mask, and in versions 2–3 theta1/theta2) is the same at every step: fold its
    // whole contribution — every tap, every const channel — into a per-cell bias. The pool
    // features depend on the state, so they are not foldable and are computed fresh each step.
    this.bias = new Float32Array(M * H);
    for (let ci = 0; ci < M; ci++) {
      for (let h = 0; h < H; h++) {
        let b = w.b1[h];
        for (let t = 0; t < T; t++) {
          const j = this.taps[ci * T + t];
          if (j < 0) continue;
          const k = tapK(TAPS[t]);
          for (let c = 0; c < K; c++) b += w.w1[(h * CK + (C + c)) * 9 + k] * this.consts[c][j];
        }
        this.bias[ci * H + h] = b;
      }
    }
    this.w2t = new Float32Array(H * C);
    for (let o = 0; o < C; o++) for (let h = 0; h < H; h++) this.w2t[h * C + o] = w.w2[o * H + h];
    this.b2 = w.b2;
    this.x = new Float64Array(W);
    this.z = new Float64Array(H8);
    this.d = new Float64Array(C);
    this.pmax = new Float64Array(PW > 0 ? C : 0);
    this.pmin = new Float64Array(PW > 0 ? C : 0);
  }

  /** The fresh state: zeros, except channel 0 = the walls. */
  reset(): void {
    this.state.fill(0);
    this.next.fill(0);
    for (const i of this.cells) this.state[i] = this.walls[i];
    this.steps = 0;
  }

  /** Paint or erase one wall; channel 0 follows at once. Off-board slots are ignored. Returns whether it changed. */
  setWall(i: number, v: 0 | 1): boolean {
    if (i < 0 || i >= this.N || !this.mask[i] || this.walls[i] === v) return false;
    this.walls[i] = v;
    this.state[i] = v;
    return true;
  }

  /** Channel c as a view into the state (S² floats). */
  channel(c: number): Float32Array {
    return this.state.subarray(c * this.N, (c + 1) * this.N);
  }

  step(n = 1): void {
    const C = this.channels;
    const H = this.hidden;
    const N = this.N;
    const T = TAPS.length;
    const TC = T * C;
    const PW = this.poolWidth;
    const { cells, taps, bias, w1r, w2t, b2, walls, x, z, d, pmax, pmin } = this;
    const H8 = z.length;
    const W = TC + PW;
    const lo = this.clamp ? this.clamp[0] : -Infinity;
    const hi = this.clamp ? this.clamp[1] : Infinity;
    const fireRate = this.fireRate;
    for (let s = 0; s < n; s++) {
      const src = this.state;
      const dst = this.next;
      for (let ci = 0; ci < cells.length; ci++) {
        const i = cells[ci];
        const fire = fireRate >= 1 || this.rand() < fireRate;
        if (fire) {
          // Gather the 7 taps' channels, tap-major.
          for (let t = 0; t < T; t++) {
            const j = taps[ci * T + t];
            const base = t * C;
            if (j < 0) for (let c = 0; c < C; c++) x[base + c] = 0;
            else for (let c = 0; c < C; c++) x[base + c] = src[c * N + j];
          }
          if (PW > 0) {
            // Per state channel, the max and min over the on-board taps among the 7 (self
            // always on board; off-board taps excluded, not the 0 the gather above filled in).
            const selfBase = SELF_TAP * C;
            for (let c = 0; c < C; c++) pmax[c] = pmin[c] = x[selfBase + c];
            for (let t = 0; t < T; t++) {
              if (t === SELF_TAP || taps[ci * T + t] < 0) continue;
              const base = t * C;
              for (let c = 0; c < C; c++) {
                const v = x[base + c];
                if (v > pmax[c]) pmax[c] = v;
                if (v < pmin[c]) pmin[c] = v;
              }
            }
            for (let c = 0; c < C; c++) {
              x[TC + c] = pmax[c];
              x[TC + C + c] = pmin[c];
            }
          }
          for (let o = 0; o < C; o++) d[o] = b2[o];
          const bOff = ci * H;
          // The W-input dot products (W = 7·C tap features, + 2C pool features when
          // pooling), eight hidden units at a time: each input is read once for all
          // eight (about 1.5× faster in V8 than a unit at a time).
          for (let u = 0, wo = 0; u < H8; u += 8) {
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            for (let f = 0; f < W; f++, wo += 8) {
              const xf = x[f];
              a0 += w1r[wo] * xf;
              a1 += w1r[wo + 1] * xf;
              a2 += w1r[wo + 2] * xf;
              a3 += w1r[wo + 3] * xf;
              a4 += w1r[wo + 4] * xf;
              a5 += w1r[wo + 5] * xf;
              a6 += w1r[wo + 6] * xf;
              a7 += w1r[wo + 7] * xf;
            }
            z[u] = a0;
            z[u + 1] = a1;
            z[u + 2] = a2;
            z[u + 3] = a3;
            z[u + 4] = a4;
            z[u + 5] = a5;
            z[u + 6] = a6;
            z[u + 7] = a7;
          }
          for (let u = 0; u < H; u++) {
            const a = bias[bOff + u] + z[u];
            if (a > 0) {
              const w2Off = u * C;
              for (let o = 0; o < C; o++) d[o] += w2t[w2Off + o] * a;
            }
          }
        }
        for (let c = 0; c < C; c++) {
          let v = src[c * N + i] + (fire ? d[c] : 0);
          if (v < lo) v = lo;
          else if (v > hi) v = hi;
          dst[c * N + i] = v;
        }
        dst[i] = walls[i];
      }
      this.state = dst;
      this.next = src;
      this.steps++;
    }
  }
}

// ── The oracle ──────────────────────────────────────────────────────────────

/**
 * The spec's targets (v4 §2), the primary first. Regions are the 6-connected
 * components of on-board non-wall cells; a region holding a rim cell is a rim
 * region, the rest are enclosed. Every enclosed region fills. Every rim region
 * fills too, except one: the one with the most cells, or whose rim cells span
 * the widest angular arc round the board's centre (within `SPAN_EPS` of the
 * widest) — the second measure replaces v2's rimcount with the angular span
 * of a region's rim cells (`buildConst`'s theta1/theta2: the narrower of the
 * two, so the branch cut in whichever one wraps through the region's arc
 * never inflates it). Each such region gives one acceptable target; the
 * primary leaves out the one with the most cells, then the widest span, then
 * the lowest cell index. With a single rim region nothing but the enclosed
 * regions fills. Walls and off-board slots are 0 in every target.
 */
const SPAN_EPS = 0.02;

export function targets(walls: ArrayLike<number>, R: number, customMask?: Uint8Array): Uint8Array[] {
  const S = side(R);
  const N = S * S;
  const mask = customMask ?? boardMask(R);
  const theta1 = buildConst('theta1', R);
  const theta2 = buildConst('theta2', R);
  // Regions by breadth-first search, numbered in order of their lowest cell index.
  const region = new Int32Array(N).fill(-1);
  const area: number[] = [];
  const rim: number[] = [];
  const span: number[] = [];
  const queue = new Int32Array(N);
  for (let s0 = 0; s0 < N; s0++) {
    if (!mask[s0] || walls[s0] || region[s0] >= 0) continue;
    const id = area.length;
    let head = 0;
    let tail = 0;
    let cells = 0;
    let rimCells = 0;
    let t1min = Infinity;
    let t1max = -Infinity;
    let t2min = Infinity;
    let t2max = -Infinity;
    region[s0] = id;
    queue[tail++] = s0;
    while (head < tail) {
      const i = queue[head++];
      const row = Math.floor(i / S);
      const col = i % S;
      cells++;
      // Rim: an on-board cell with an off-board neighbour (off the array, or off `mask` —
      // the board's own shape, the full hexagon on the default board). For the default
      // board this is exactly hexDist(col - R, row - R) === R; for a ragged one (a
      // Spectacle field) it is the field's own outline, with no extra input needed.
      let onRim = false;
      for (const [dr, dc] of NEIGHBOURS) {
        const r2 = row + dr;
        const c2 = col + dc;
        if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S || !mask[r2 * S + c2]) {
          onRim = true;
          break;
        }
      }
      if (onRim) {
        rimCells++;
        if (theta1[i] < t1min) t1min = theta1[i];
        if (theta1[i] > t1max) t1max = theta1[i];
        if (theta2[i] < t2min) t2min = theta2[i];
        if (theta2[i] > t2max) t2max = theta2[i];
      }
      for (const [dr, dc] of NEIGHBOURS) {
        const r2 = row + dr;
        const c2 = col + dc;
        if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S) continue;
        const j = r2 * S + c2;
        if (!mask[j] || walls[j] || region[j] >= 0) continue;
        region[j] = id;
        queue[tail++] = j;
      }
    }
    area.push(cells);
    rim.push(rimCells);
    span.push(rimCells > 0 ? Math.min(t1max - t1min, t2max - t2min) : 0);
  }
  // The rim regions in the primary's order: most cells, then widest span, then lowest index.
  const rims = area.map((_, id) => id).filter((id) => rim[id] > 0)
    .sort((a, b) => area[b] - area[a] || span[b] - span[a] || a - b);
  let maxArea = 0;
  let maxSpan = 0;
  for (const id of rims) {
    maxArea = Math.max(maxArea, area[id]);
    maxSpan = Math.max(maxSpan, span[id]);
  }
  const unfilled = rims.filter((id) => area[id] === maxArea || span[id] >= maxSpan - SPAN_EPS);
  if (!unfilled.length) unfilled.push(-1); // no rim region at all: everything fills
  return unfilled.map((x) => {
    const fill = new Uint8Array(N);
    for (let i = 0; i < N; i++) if (region[i] >= 0 && region[i] !== x) fill[i] = 1;
    return fill;
  });
}

/**
 * A random bridge for the page: src/lines.ts's randomLine (a cheapest path
 * under random costs, so straight, curved or wandering) from one rim cell to
 * another through the interior, round the walls already there. Its cells in
 * drawing order, or null when there is no way across.
 */
export function randomBridge(walls: ArrayLike<number>, R: number, rand: () => number): number[] | null {
  const board = makeBoard(R);
  const blocked = new Set<number>();
  for (let i = 0; i < walls.length; i++) {
    if (!walls[i]) continue;
    const [q, r] = cellCoords(R, i);
    if (hexDist(q, r) <= R) blocked.add(indexOf(board, q, r));
  }
  const line = randomLine(board, rand, { blocked, wiggle: rand() * 3 });
  return line && line.map((slot) => {
    const [q, r] = coordsOf(board, slot);
    return cellIndex(R, q, r);
  });
}
