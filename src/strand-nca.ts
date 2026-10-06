// The trained strand NCAs (nca/strand: train2.py's FrameNCA and StrandNCA, train.py's v1 HexNCA) run in
// TypeScript, pure, no DOM: one synchronous step of every on-board cell, matching PyTorch's float32 step to
// ~1e-6 (tests/strand.test.ts against tests/fixtures/strand-parity.json). The weights come from
// nca/strand/export.py ("hexca-strand" version 1; its docstring has the format).
//
// One step, per cell i in frame f (arch "frame": f = the cell's geo % 12 = rot * 2 + mirror bit; arch "taps":
// every cell in frame 0, which permutes nothing):
//   taps     local tap j = 0 is the cell, 1 + k the neighbour across board direction sigma_f(k) (0 off the board);
//            local channel c of a directional group g (dirIn) reads board channel g + sigma_f(c - g), any
//            other channel itself. sigma_f(k) = (m k + rot) mod 6, m = 1 - 2 * mirror bit.
//   pre      b1 + w1 [H, 7 · Cx] · taps          (Cx = C state + nIn consts; the consts never change, so their
//                                                  part is folded into a per-cell bias once per set of taps)
//   h        relu(pre); then relu(w h + b) per hidden layer (depth - 1)
//   d        w2 [C, H] · h + b2                  (local output channels; board channel g + d of a directional
//                                                  group takes local output g + sigma_f^-1(d))
//   state    clamp(state + d); state[0] = 1 (the mask); off-board cells don't exist here.
// The state is cell-major: state[i * C + c].

import { Board, CODE_BITS, PAIRS, type Rule, RuleTable } from './strand.js';

export const STRAND_FORMAT = 'hexca-strand';

export interface StrandMeta {
  run?: string | null;
  file?: string;
  iteration?: number;
  prevIterations?: number;
  levels?: number[];
  init?: string | null;
  best?: number;
  heldOut?: {
    iteration: number;
    exact?: number;
    balanced?: number;
    iou?: number;
    byLevel?: Record<string, { exact?: number; balanced?: number; iou?: number }>;
    byLen?: Record<string, number>;
    bySet?: Record<string, { exact?: number; balanced?: number }>;
    exit?: number;
  } | null;
  note?: string;
  [key: string]: unknown;
}

export interface StrandWeights {
  arch: 'frame' | 'taps';
  kind: 'v1' | 'v2';
  task: string;
  inputs: string;
  channels: number;
  hidden: number;
  depth: number;
  nIn: number;
  clamp: [number, number] | null;
  dirIn: number[];
  consts: [string, number][];
  /** The consts carry one rule on every cell (bc inputs, v1's chord planes): one tap per board. */
  ruleEverywhere: boolean;
  /** [H, 7, Cx] */
  w1: Float32Array;
  b1: Float32Array;
  mids: { w: Float32Array; b: Float32Array }[];
  /** [C, H] */
  w2: Float32Array;
  b2: Float32Array;
  /** [108, planes]: the static input planes per geo. */
  staticLut: { planes: number; data: Float32Array } | null;
  meta: StrandMeta;
}

/** Whether a parsed JSON file is strand weights (and not, say, the flood NCA's). */
export function isStrandWeights(json: unknown): boolean {
  return !!json && typeof json === 'object' && (json as { format?: unknown }).format === STRAND_FORMAT;
}

/** Little-endian float32 from base64 (atob in the browser, Buffer in Node). */
export function decodeF32(b64: string): Float32Array {
  let bytes: Uint8Array;
  const B = (globalThis as { Buffer?: { from(s: string, enc: string): Uint8Array } }).Buffer;
  if (B) bytes = Uint8Array.from(B.from(b64, 'base64'));
  else {
    const s = atob(b64);
    bytes = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  }
  const out = new Float32Array(bytes.length / 4);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

interface TensorJson { shape: number[]; data: string }

function tensor(t: TensorJson, shape: number[], what: string): Float32Array {
  if (!t || t.shape.length !== shape.length || t.shape.some((v, i) => v !== shape[i])) {
    throw new Error(`strand weights: ${what} has shape [${t?.shape}], expected [${shape}]`);
  }
  const a = decodeF32(t.data);
  if (a.length !== shape.reduce((x, y) => x * y, 1)) throw new Error(`strand weights: ${what} is truncated`);
  return a;
}

/** Decode and check a weights file (nca/strand/export.py). Throws with the reason if it isn't one. */
export function loadStrandWeights(json: unknown): StrandWeights {
  if (!isStrandWeights(json)) throw new Error('not strand weights (no "format": "hexca-strand"): a flood model?');
  const j = json as Record<string, unknown> & { tensors: Record<string, unknown> };
  if (j.version !== 1) throw new Error(`strand weights version ${String(j.version)}: this page reads version 1`);
  const C = j.channels as number;
  const H = j.hidden as number;
  const nIn = j.nIn as number;
  const consts = j.consts as [string, number][];
  if (consts.reduce((a, [, n]) => a + n, 0) !== nIn || consts[0]?.[0] !== 'mask') {
    throw new Error('strand weights: the consts layout does not add up');
  }
  const T = j.tensors as { w1: TensorJson; b1: TensorJson; mids: { w: TensorJson; b: TensorJson }[]; w2: TensorJson; b2: TensorJson };
  const lut = j.staticLut as { planes: number; data: string } | undefined;
  return {
    arch: j.arch as 'frame' | 'taps',
    kind: j.kind as 'v1' | 'v2',
    task: (j.task as string) ?? 'm1a',
    inputs: j.inputs as string,
    channels: C,
    hidden: H,
    depth: j.depth as number,
    nIn,
    clamp: (j.clamp as [number, number] | null) ?? null,
    dirIn: (j.dirIn as number[]) ?? [],
    consts,
    ruleEverywhere: !!j.ruleEverywhere,
    w1: tensor(T.w1, [H, 7, C + nIn], 'w1'),
    b1: tensor(T.b1, [H], 'b1'),
    mids: T.mids.map((m, k) => ({ w: tensor(m.w, [H, H], `mids[${k}].w`), b: tensor(m.b, [H], `mids[${k}].b`) })),
    w2: tensor(T.w2, [C, H], 'w2'),
    b2: tensor(T.b2, [C], 'b2'),
    staticLut: lut ? { planes: lut.planes, data: decodeF32(lut.data) } : null,
    meta: (j.meta as StrandMeta) ?? {},
  };
}

/** A tap: a rule, and the tapped chord (d0, d1) of cell (row, col). */
export interface Tap {
  rule: Rule;
  row: number;
  col: number;
  d0: number;
  d1: number;
}

const N_FRAMES = 12;

/** sigma[f * 6 + k]: the board direction of local direction k in frame f = rot * 2 + mirror bit. */
const SIGMA: Int8Array = (() => {
  const out = new Int8Array(N_FRAMES * 6);
  for (let rot = 0; rot < 6; rot++) {
    for (let mb = 0; mb < 2; mb++) {
      for (let k = 0; k < 6; k++) out[(rot * 2 + mb) * 6 + k] = ((((1 - 2 * mb) * k + rot) % 6) + 6) % 6;
    }
  }
  return out;
})();

/** One weights file's per-frame tables. */
class Frames {
  readonly n: number;
  /** stateChan[f * C + c]: the board state channel local channel c reads. */
  readonly stateChan: Int32Array;
  /** constLocal[f * nIn + b]: the local const channel that board const channel b lands in. */
  readonly constLocal: Int32Array;
  /** outLocal[f * C + c]: the local output that board channel c takes. */
  readonly outLocal: Int32Array;
  /** tapDir[f * 7 + j]: the board direction of local tap j (-1: the cell itself). */
  readonly tapDir: Int8Array;

  constructor(W: StrandWeights) {
    const C = W.channels;
    const Cx = C + W.nIn;
    this.n = W.arch === 'frame' ? N_FRAMES : 1;
    const isDir = new Int32Array(Cx).fill(-1); // channel -> its group's first channel
    for (const g of W.dirIn) for (let k = 0; k < 6; k++) isDir[g + k] = g;
    this.stateChan = new Int32Array(this.n * C);
    this.constLocal = new Int32Array(this.n * W.nIn);
    this.outLocal = new Int32Array(this.n * C);
    this.tapDir = new Int8Array(this.n * 7);
    for (let f = 0; f < this.n; f++) {
      const sig = (k: number) => SIGMA[f * 6 + k];
      const inv = (d: number) => { for (let k = 0; k < 6; k++) if (sig(k) === d) return k; return -1; };
      // local channel c reads board channel boardOf(c)
      const boardOf = (c: number) => (isDir[c] >= 0 ? isDir[c] + sig(c - isDir[c]) : c);
      for (let c = 0; c < C; c++) this.stateChan[f * C + c] = boardOf(c);
      for (let c = C; c < Cx; c++) this.constLocal[f * W.nIn + (boardOf(c) - C)] = c - C;
      for (let c = 0; c < C; c++) this.outLocal[f * C + c] = isDir[c] >= 0 ? isDir[c] + inv(c - isDir[c]) : c;
      this.tapDir[f * 7] = -1;
      for (let k = 0; k < 6; k++) this.tapDir[f * 7 + 1 + k] = sig(k);
    }
  }
}

/**
 * out[c * outStride + r] = sum_k M[r * K + k] * X[c * xStride + k] for the 4 vectors c of X and every row r: four
 * cells at a time, four rows at a time (each weight read once per four cells; sixteen independent sums).
 */
function gemm4(M: Float32Array, rows: number, K: number, X: Float32Array, xStride: number,
  out: Float32Array, outStride: number): void {
  const x1 = xStride;
  const x2 = 2 * xStride;
  const x3 = 3 * xStride;
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    let c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    const wa = r * K;
    const wb = wa + K;
    const wc = wb + K;
    const wd = wc + K;
    for (let k = 0; k < K; k++) {
      const y0 = X[k];
      const y1 = X[x1 + k];
      const y2 = X[x2 + k];
      const y3 = X[x3 + k];
      let w = M[wa + k];
      a0 += w * y0; a1 += w * y1; a2 += w * y2; a3 += w * y3;
      w = M[wb + k];
      b0 += w * y0; b1 += w * y1; b2 += w * y2; b3 += w * y3;
      w = M[wc + k];
      c0 += w * y0; c1 += w * y1; c2 += w * y2; c3 += w * y3;
      w = M[wd + k];
      d0 += w * y0; d1 += w * y1; d2 += w * y2; d3 += w * y3;
    }
    let o = r;
    out[o] = a0; out[o + 1] = b0; out[o + 2] = c0; out[o + 3] = d0;
    o += outStride;
    out[o] = a1; out[o + 1] = b1; out[o + 2] = c1; out[o + 3] = d1;
    o += outStride;
    out[o] = a2; out[o + 1] = b2; out[o + 2] = c2; out[o + 3] = d2;
    o += outStride;
    out[o] = a3; out[o + 1] = b3; out[o + 2] = c3; out[o + 3] = d3;
  }
  for (; r < rows; r++) {
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const wa = r * K;
    for (let k = 0; k < K; k++) {
      const w = M[wa + k];
      a0 += w * X[k]; a1 += w * X[x1 + k]; a2 += w * X[x2 + k]; a3 += w * X[x3 + k];
    }
    out[r] = a0; out[outStride + r] = a1; out[2 * outStride + r] = a2; out[3 * outStride + r] = a3;
  }
}

/** The const channels of one board cell, sparse: [channel, value] pairs (channel indexes the consts, mask = 0). */
type SparseConsts = [number, number][];

/** The trained strand CA on one board, with a set of taps held on their cells. */
export class StrandNCA {
  readonly C: number;
  readonly H: number;
  readonly K: number;
  /** state[i * C + c], cell-major over the board's on-board cells. */
  state: Float32Array;
  private next: Float32Array;
  steps = 0;
  private readonly frames: Frames;
  private readonly frameOf: Int32Array;
  /** w1's state columns as rows [H, 7 * C] in (local tap, local channel) order. */
  private readonly w1s: Float32Array;
  /** w1's const columns transposed: [7, nIn, H]. */
  private readonly w1cT: Float32Array;
  /** Per cell: b1 + the consts' part of w1 · taps. [n, H] */
  private bias: Float32Array;
  private taps: Tap[] = [];
  // scratch, four cells at a time
  private readonly X: Float32Array;
  private readonly hA: Float32Array;
  private readonly hB: Float32Array;
  private readonly dOut: Float32Array;

  constructor(readonly weights: StrandWeights, readonly board: Board, readonly table: RuleTable) {
    const W = weights;
    this.C = W.channels;
    this.H = W.hidden;
    this.K = 7 * this.C;
    const Cx = this.C + W.nIn;
    this.frames = new Frames(W);
    this.frameOf = new Int32Array(board.n);
    if (W.arch === 'frame') for (let i = 0; i < board.n; i++) this.frameOf[i] = board.geo[board.pos[i]] % 12;
    this.w1s = new Float32Array(this.H * this.K);
    this.w1cT = new Float32Array(7 * W.nIn * this.H);
    for (let h = 0; h < this.H; h++) {
      for (let j = 0; j < 7; j++) {
        for (let c = 0; c < this.C; c++) this.w1s[h * this.K + j * this.C + c] = W.w1[(h * 7 + j) * Cx + c];
        for (let c = 0; c < W.nIn; c++) this.w1cT[(j * W.nIn + c) * this.H + h] = W.w1[(h * 7 + j) * Cx + this.C + c];
      }
    }
    this.state = new Float32Array(board.n * this.C);
    this.next = new Float32Array(board.n * this.C);
    this.bias = new Float32Array(board.n * this.H);
    this.X = new Float32Array(4 * this.K);
    this.hA = new Float32Array(4 * this.H);
    this.hB = new Float32Array(4 * this.H);
    this.dOut = new Float32Array(4 * this.C);
    this.setTaps([]);
    this.reset();
  }

  get tapList(): readonly Tap[] {
    return this.taps;
  }

  /** The fresh state: zeros, channel 0 (the mask) 1. */
  reset(): void {
    this.state.fill(0);
    for (let i = 0; i < this.board.n; i++) this.state[i * this.C] = 1;
    this.steps = 0;
  }

  /** Hold these taps from now on (the state is kept: call reset() to start over). */
  setTaps(taps: Tap[]): void {
    if (this.weights.ruleEverywhere && taps.length > 1) {
      throw new Error('this model carries its rule on every cell: one tap per board');
    }
    this.taps = taps.slice();
    this.buildBias();
  }

  /** The consts of every cell, sparse (training's inputs: train2.Planes, train.consts_of). */
  private constsOf(): SparseConsts[] {
    const W = this.weights;
    const B = this.board;
    const out: SparseConsts[] = Array.from({ length: B.n }, () => []);
    let off = 0;
    const first = this.taps[0];
    const tapAt = new Map<number, Tap[]>();
    for (const t of this.taps) {
      const i = B.cellOf[t.row * B.w + t.col];
      if (i < 0) throw new Error(`tap off the board at (${t.row}, ${t.col})`);
      tapAt.set(i, [...(tapAt.get(i) ?? []), t]);
    }
    for (const [name, n] of W.consts) {
      if (name === 'mask') {
        for (let i = 0; i < B.n; i++) out[i].push([off, 1]);
      } else if (name === 'static') {
        const lut = W.staticLut!;
        for (let i = 0; i < B.n; i++) {
          const g = B.geo[B.pos[i]];
          for (let k = 0; k < lut.planes; k++) {
            const v = lut.data[g * lut.planes + k];
            if (v !== 0) out[i].push([off + k, v]);
          }
        }
      } else if (name === 'code') {
        if (first) {
          const code = this.table.code(first.rule);
          for (let i = 0; i < B.n; i++) for (let k = 0; k < CODE_BITS; k++) if (code[k]) out[i].push([off + k, 1]);
        }
      } else if (name === 'chords') {
        if (first) {
          const bits = this.table.render(first.rule, B);
          for (let i = 0; i < B.n; i++) for (let k = 0; k < 15; k++) if ((bits[B.pos[i]] >> k) & 1) out[i].push([off + k, 1]);
        }
      } else if (name === 'tapCode') {
        for (const [i, ts] of tapAt) {
          for (const t of ts) {
            const code = this.table.code(t.rule);
            for (let k = 0; k < CODE_BITS; k++) if (code[k]) out[i].push([off + k, 1]);
          }
        }
      } else if (name === 'tapDirs') {
        for (const [i, ts] of tapAt) for (const t of ts) out[i].push([off + t.d0, 1], [off + t.d1, 1]);
      } else {
        throw new Error(`strand weights: unknown const block "${name}"`);
      }
      off += n;
    }
    return out;
  }

  private buildBias(): void {
    const { H } = this;
    const nIn = this.weights.nIn;
    const B = this.board;
    const F = this.frames;
    const cs = this.constsOf();
    for (let i = 0; i < B.n; i++) {
      const f = this.frameOf[i];
      const o = i * H;
      for (let h = 0; h < H; h++) this.bias[o + h] = this.weights.b1[h];
      const acc = new Float64Array(H);
      for (let j = 0; j < 7; j++) {
        const dir = F.tapDir[f * 7 + j];
        const src = dir < 0 ? i : B.nbr[i * 6 + dir];
        if (src < 0) continue;
        for (const [ch, v] of cs[src]) {
          const w = (j * nIn + F.constLocal[f * nIn + ch]) * H;
          for (let h = 0; h < H; h++) acc[h] += v * this.w1cT[w + h];
        }
      }
      for (let h = 0; h < H; h++) this.bias[o + h] = this.weights.b1[h] + acc[h];
    }
  }

  /** One synchronous step of every cell. */
  step(): void {
    this.stepRange(0, this.board.n);
    const t = this.state;
    this.state = this.next;
    this.next = t;
    this.steps++;
  }

  /** The new state of cells [lo, hi) into `next` (from `state`). */
  stepRange(lo: number, hi: number): void {
    const { C, H, K, X, hA, hB, dOut } = this;
    const W = this.weights;
    const B = this.board;
    const F = this.frames;
    const src = this.state;
    const dst = this.next;
    const lo2 = W.clamp ? W.clamp[0] : -Infinity;
    const hi2 = W.clamp ? W.clamp[1] : Infinity;
    for (let i0 = lo; i0 < hi; i0 += 4) {
      const nb = Math.min(4, hi - i0);
      // gather the four cells' taps, each in its own frame
      for (let b = 0; b < 4; b++) {
        const xo = b * K;
        if (b >= nb) { X.fill(0, xo, xo + K); continue; }
        const i = i0 + b;
        const f = this.frameOf[i];
        const sc = f * C;
        for (let j = 0; j < 7; j++) {
          const dir = F.tapDir[f * 7 + j];
          const s = dir < 0 ? i : B.nbr[i * 6 + dir];
          const o = xo + j * C;
          if (s < 0) { X.fill(0, o, o + C); continue; }
          const so = s * C;
          for (let c = 0; c < C; c++) X[o + c] = src[so + F.stateChan[sc + c]];
        }
      }
      gemm4(this.w1s, H, K, X, K, hA, H);
      for (let b = 0; b < nb; b++) {
        const bo = (i0 + b) * H;
        for (let h = 0; h < H; h++) {
          const v = hA[b * H + h] + this.bias[bo + h];
          hA[b * H + h] = v > 0 ? v : 0;
        }
      }
      let cur = hA;
      let other = hB;
      for (const m of W.mids) {
        gemm4(m.w, H, H, cur, H, other, H);
        for (let b = 0; b < nb; b++) {
          for (let h = 0; h < H; h++) {
            const v = other[b * H + h] + m.b[h];
            other[b * H + h] = v > 0 ? v : 0;
          }
        }
        const t = cur;
        cur = other;
        other = t;
      }
      gemm4(W.w2, C, H, cur, H, dOut, C);
      for (let b = 0; b < nb; b++) {
        const i = i0 + b;
        const f = this.frameOf[i];
        const so = i * C;
        dst[so] = 1;
        for (let c = 1; c < C; c++) {
          const l = F.outLocal[f * C + c];
          let v = src[so + c] + (dOut[b * C + l] + W.b2[l]);
          if (v < lo2) v = lo2;
          else if (v > hi2) v = hi2;
          dst[so + c] = v;
        }
      }
    }
  }

  /** Whether the model draws edge d of cell i (state channel 1 + d > 0.5, as train2.evaluate reads it). */
  edgeOn(i: number, d: number): boolean {
    return this.state[i * this.C + 1 + d] > 0.5;
  }
}

/** Training's taps: a chord of the cell under the rule, either way round. */
export function chordsAt(table: RuleTable, rule: Rule, board: Board, row: number, col: number): [number, number][] {
  const bits = table.render(rule, board)[row * board.w + col];
  const out: [number, number][] = [];
  for (let p = 0; p < 15; p++) if ((bits >> p) & 1) out.push([PAIRS[p][0], PAIRS[p][1]]);
  return out;
}
