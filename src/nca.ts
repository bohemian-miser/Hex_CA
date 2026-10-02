// The trained hex neural cellular automaton (nca/, the spec's "Model"), run in
// the browser: pure TypeScript, no DOM.
//
// Board: the hexagon of radius R in axial (q, r), stored in an S×S array,
// S = 2R + 1, cell (q, r) at [row = r + R][col = q + R]. Off-board slots exist
// but are dead: every channel is 0 there, always. Neighbours as (drow, dcol):
// (0,+1) (0,-1) (+1,0) (-1,0) (-1,+1) (+1,-1). This is NOT src/hex.ts's layout
// (that one pads a ring); it is the Python side's, nca/hexgrid.py.
//
// One step, every on-board cell at once (synchronous, double-buffered):
//   x = concat(state, mask)                  17 channels
//   h = relu(conv3x3_hexmasked(x) + b1)      17 -> H, 7 taps (corners k=0, 8 are zero)
//   d = conv1x1(h) + b2                      H -> C
//   state = state + d · fire                 fire = 1 when fireRate is 1
//   state = clamp(state, lo, hi)             when the export has a clamp
//   state[ch0] = wall; state *= mask

/** What the weights were trained on (nca/export.py writes it); every field optional. */
export interface NCAMeta {
  trainedR?: number;
  steps?: [number, number];
  iterations?: number;
  /** Trained with the persistent pool and wall edits: the state may be kept across edits. */
  pool?: boolean;
  note?: string;
  [key: string]: unknown;
}

export interface NCAWeights {
  channels: number;
  hidden: number;
  clamp: [number, number] | null;
  fireRate: number;
  /** H × (C + 1) × 9, index ((h·(C+1)) + c)·9 + k, k = (drow+1)·3 + (dcol+1); c = C is the mask. */
  w1: Float32Array;
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

/** The exported JSON (web/nca-weights.json) as typed arrays, checked against the spec's format. */
export function loadWeights(json: unknown): NCAWeights {
  if (typeof json !== 'object' || json === null || Array.isArray(json)) throw new Error('nca weights: not an object');
  const j = json as Record<string, unknown>;
  if (j.version !== 1) throw new Error(`nca weights: unsupported version ${String(j.version)}`);
  const C = posInt(j, 'channels', 2);
  const H = posInt(j, 'hidden', 1);
  let clamp: [number, number] | null = null;
  if (j.clamp !== null && j.clamp !== undefined) {
    const c = j.clamp;
    if (!Array.isArray(c) || c.length !== 2 || !c.every((v) => typeof v === 'number' && Number.isFinite(v)) || !(c[0] < c[1])) {
      throw new Error('nca weights: clamp must be null or [lo, hi] with lo < hi');
    }
    clamp = [c[0], c[1]];
  }
  const fireRate = j.fireRate ?? 1;
  if (typeof fireRate !== 'number' || !(fireRate > 0 && fireRate <= 1)) throw new Error('nca weights: fireRate must be in (0, 1]');
  const w1 = floats(j, 'w1', H * (C + 1) * 9);
  for (let i = 0; i < w1.length; i += 9) {
    if (w1[i] !== 0 || w1[i + 8] !== 0) throw new Error('nca weights: a corner tap (k = 0 or 8) is not zero');
  }
  const meta = j.meta;
  if (meta !== undefined && (typeof meta !== 'object' || meta === null || Array.isArray(meta))) {
    throw new Error('nca weights: meta must be an object');
  }
  return {
    channels: C, hidden: H, clamp, fireRate,
    w1, b1: floats(j, 'b1', H), w2: floats(j, 'w2', C * H), b2: floats(j, 'b2', C),
    meta: (meta ?? {}) as NCAMeta,
  };
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
  /** Per cell and hidden unit: b1 plus the mask channel's taps, which never change. */
  private readonly bias: Float32Array;
  /** w1 without the mask channel, regrouped [h][tap][c] to match the gathered input (doubles: faster to read here). */
  private readonly w1r: Float64Array;
  /** w2 transposed, [h][o]. */
  private readonly w2t: Float32Array;
  private readonly b2: Float32Array;
  private readonly x: Float64Array;
  private readonly d: Float64Array;

  /** `rand` is only drawn on when fireRate < 1 (each cell fires with that chance, all channels together). */
  constructor(w: NCAWeights, R: number, rand: () => number = Math.random) {
    if (!Number.isInteger(R) || R < 1) throw new Error('HexNCA: R must be an integer >= 1');
    const C = w.channels;
    const H = w.hidden;
    const S = side(R);
    const N = S * S;
    this.R = R;
    this.S = S;
    this.N = N;
    this.channels = C;
    this.hidden = H;
    this.clamp = w.clamp;
    this.fireRate = w.fireRate;
    this.rand = rand;
    this.mask = boardMask(R);
    this.walls = new Uint8Array(N);
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

    const C1 = C + 1;
    this.w1r = new Float64Array(H * T * C);
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < T; t++) {
        const k = tapK(TAPS[t]);
        for (let c = 0; c < C; c++) this.w1r[(h * T + t) * C + c] = w.w1[(h * C1 + c) * 9 + k];
      }
    }
    // The mask channel is 1 at every on-board tap and 0 elsewhere: fold it into a per-cell bias.
    this.bias = new Float32Array(M * H);
    for (let ci = 0; ci < M; ci++) {
      for (let h = 0; h < H; h++) {
        let b = w.b1[h];
        for (let t = 0; t < T; t++) if (this.taps[ci * T + t] >= 0) b += w.w1[(h * C1 + C) * 9 + tapK(TAPS[t])];
        this.bias[ci * H + h] = b;
      }
    }
    this.w2t = new Float32Array(H * C);
    for (let o = 0; o < C; o++) for (let h = 0; h < H; h++) this.w2t[h * C + o] = w.w2[o * H + h];
    this.b2 = w.b2;
    this.x = new Float64Array(T * C);
    this.d = new Float64Array(C);
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
    const { cells, taps, bias, w1r, w2t, b2, walls, x, d } = this;
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
          for (let o = 0; o < C; o++) d[o] = b2[o];
          const bOff = ci * H;
          for (let u = 0; u < H; u++) {
            // The dot product of 7·C inputs, eight running sums (about 1.5× faster in V8 than one).
            const wOff = u * TC;
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            let f = 0;
            for (; f + 7 < TC; f += 8) {
              a0 += w1r[wOff + f] * x[f];
              a1 += w1r[wOff + f + 1] * x[f + 1];
              a2 += w1r[wOff + f + 2] * x[f + 2];
              a3 += w1r[wOff + f + 3] * x[f + 3];
              a4 += w1r[wOff + f + 4] * x[f + 4];
              a5 += w1r[wOff + f + 5] * x[f + 5];
              a6 += w1r[wOff + f + 6] * x[f + 6];
              a7 += w1r[wOff + f + 7] * x[f + 7];
            }
            for (; f < TC; f++) a0 += w1r[wOff + f] * x[f];
            const a = bias[bOff + u] + (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7);
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
 * The spec's target: 1 on every on-board non-wall cell that is NOT joined,
 * through on-board non-wall cells (6-connected), to a non-wall rim cell; 0 on
 * walls, outside cells and off the board. A breadth-first search from the rim.
 */
export function enclosed(walls: ArrayLike<number>, R: number): Uint8Array {
  const S = side(R);
  const N = S * S;
  const outside = new Uint8Array(N);
  const queue = new Int32Array(N);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < N; i++) {
    const [q, r] = cellCoords(R, i);
    if (hexDist(q, r) === R && !walls[i]) {
      outside[i] = 1;
      queue[tail++] = i;
    }
  }
  while (head < tail) {
    const i = queue[head++];
    const row = Math.floor(i / S);
    const col = i % S;
    for (const [dr, dc] of NEIGHBOURS) {
      const r2 = row + dr;
      const c2 = col + dc;
      if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S) continue;
      const j = r2 * S + c2;
      if (outside[j] || walls[j] || hexDist(c2 - R, r2 - R) > R) continue;
      outside[j] = 1;
      queue[tail++] = j;
    }
  }
  const fill = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const [q, r] = cellCoords(R, i);
    if (hexDist(q, r) <= R && !walls[i] && !outside[i]) fill[i] = 1;
  }
  return fill;
}
