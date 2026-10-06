// Spectacle's strand rules on its hex fields, in TypeScript (pure, no DOM): the rule table, a rule's chords on
// a board, and the walker that follows a strand — nca/strand/rules.py and walker.py, which are checked against
// Spectacle's own rendering and walkStrand (`python -m nca.strand.rules --split --parity`); tests/strand.test.ts
// checks this file against them.
//
// Conventions (docs/strand-data.md, scripts/strand-export.ts): a board is an H×W array, cell (row, col), axial
// q = col + q0, r = row + r0. Direction d in 0..5 is (dq, dr) = (1,0) (1,-1) (0,-1) (-1,0) (-1,1) (0,1), i.e.
// (drow, dcol) = (0,1) (-1,1) (-1,0) (0,-1) (1,-1) (1,0); opposite = (d + 3) % 6. A cell's geo is
// type * 12 + rot * 2 + (mirror < 0), -1 off the board; its local edge k faces board direction
// (mirror * k + rot) mod 6. Chord bit p joins the direction pair PAIRS[p] = (0,1), (0,2), …, (4,5).
//
// A RULE is (s, digits): s one of the 7 kernel subsets, digits[t] the position of leaf type t's matching among
// its non-crossing ones, in Spectacle's order. Written as Spectacle's describeRule does, `128·010100000`.

export const DROW = [0, -1, -1, 0, 1, 1] as const;
export const DCOL = [1, 1, 0, -1, -1, 0] as const;
export const PAIRS: ReadonlyArray<readonly [number, number]> = (() => {
  const out: [number, number][] = [];
  for (let a = 0; a < 6; a++) for (let b = a + 1; b < 6; b++) out.push([a, b]);
  return out;
})();
/** PAIR_INDEX[a * 6 + b] = the chord bit of the pair (a, b), either order; -1 for a = b. */
export const PAIR_INDEX: Int8Array = (() => {
  const out = new Int8Array(36).fill(-1);
  PAIRS.forEach(([a, b], p) => { out[a * 6 + b] = p; out[b * 6 + a] = p; });
  return out;
})();

export const N_TYPES = 9;
export const N_DIGITS = 5;
export const CODE_BITS = 8 + N_TYPES * N_DIGITS; // 53

export interface SubsetData {
  key: string;
  edges: number[];
  count: number;
  /** options[t] = Spectacle's matching indices of type t's non-crossing matchings (digit -> index). */
  options: number[][];
  /** localPairs[t][digit] = the matching's chords as local edge pairs. */
  localPairs: number[][][][];
  /** The split: the k rules with the smallest (hash, index) are held out; thr* is the k-th. v1: the subset's
   * held-out rules are the legacy ones instead (legacyHeldout). */
  split: { k: number; thrHash: number; thrIndex: number; v1: boolean };
}

export interface BoardData {
  level: number;
  root: string;
  h: number;
  w: number;
  mirror: number;
  tiles: number;
  geo: number[];
}

export interface StrandData {
  version: number;
  source: string;
  leafOrder: string[];
  majors: number[];
  typeMajors: number[][];
  subsets: SubsetData[];
  legacyHeldout: [number, number][];
  boards: Record<string, BoardData>;
}

export interface Rule {
  s: number;
  digits: number[];
}

const SPLIT_SALT = 'strand-split-v2';

/** FNV-1a, then murmur3's fmix32 (scripts/strand-export.ts's splitHash). */
export function splitHash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

export type Split = 'train' | 'heldout';

export class RuleTable {
  readonly subsets: SubsetData[];
  readonly keys: string[];
  readonly majors: number[];
  /** nOpt[s][t]: how many matchings type t has in subset s (1 for a type with none: digit 0, no chords). */
  readonly nOpt: number[][];
  readonly radix: number[][];
  readonly leafOrder: string[];
  private readonly legacy = new Set<string>();

  constructor(readonly data: StrandData) {
    this.subsets = data.subsets;
    this.keys = data.subsets.map((s) => s.key);
    this.majors = data.majors;
    this.leafOrder = data.leafOrder;
    this.nOpt = data.subsets.map((s) => s.options.map((o) => o.length));
    this.radix = this.nOpt.map((n) => n.map((_, t) => n.slice(t + 1).reduce((a, b) => a * b, 1)));
    for (const [s, i] of data.legacyHeldout) this.legacy.add(`${s}:${i}`);
  }

  get nSub(): number {
    return this.subsets.length;
  }

  index(rule: Rule): number {
    return rule.digits.reduce((a, d, t) => a + d * this.radix[rule.s][t], 0);
  }

  digitsOf(s: number, index: number): number[] {
    return this.radix[s].map((r, t) => Math.floor(index / r) % this.nOpt[s][t]);
  }

  /** Spectacle's ruleKey, `hex|<subset>|<matching indices joined by .>`. */
  key(rule: Rule): string {
    const sub = this.subsets[rule.s];
    return `hex|${sub.key}|${rule.digits.map((d, t) => sub.options[t][d]).join('.')}`;
  }

  /** Spectacle's describeRule form (without its spaces): `128·010100000`. */
  describe(rule: Rule): string {
    return `${this.keys[rule.s]}·${rule.digits.join('')}`;
  }

  /**
   * A rule from its describeRule form, `128·010100000` (or `128 · 010100000`, `128/010100000`, `128 010100000`),
   * checked against the table: the subset must be one of the kernel's seven and every digit in range for its
   * type. Missing trailing digits are 0, as in Spectacle. Returns the rule or why not.
   */
  parse(text: string): Rule | string {
    const m = /^\s*(\d+)\s*[·•.:|/,\s]\s*(\d{1,9})\s*$/.exec(text);
    if (!m) return 'write it as <subset>·<digits>, e.g. 128·010100000';
    const key = [...new Set(m[1].split('').map(Number))].sort((a, b) => a - b).join('');
    const s = this.keys.indexOf(key);
    if (s < 0) return `subset ${m[1]} is not one of the kernel's: ${this.keys.join(', ')}`;
    const digits = m[2].padEnd(N_TYPES, '0').split('').map(Number);
    for (let t = 0; t < N_TYPES; t++) {
      if (digits[t] >= this.nOpt[s][t]) {
        const n = this.nOpt[s][t];
        return `${this.leafOrder[t]} (digit ${t + 1}) has ${n} matching${n > 1 ? 's' : ''} in subset ${key}: 0${n > 1 ? `-${n - 1}` : ''}`;
      }
    }
    return { s, digits };
  }

  /** uint8 [53]: 8 bits "class m carries a line" (m in majors), then per type a one-hot of its digit. */
  code(rule: Rule): Uint8Array {
    const out = new Uint8Array(CODE_BITS);
    const edges = this.subsets[rule.s].edges;
    this.majors.forEach((m, j) => { out[j] = edges.includes(m) ? 1 : 0; });
    rule.digits.forEach((d, t) => { out[this.majors.length + t * N_DIGITS + d] = 1; });
    return out;
  }

  /** Held out of training (the v2 split; nca/strand/rules.py's held_out). */
  heldOut(rule: Rule): boolean {
    const sub = this.subsets[rule.s];
    const idx = this.index(rule);
    if (sub.split.v1) return this.legacy.has(`${rule.s}:${idx}`);
    const h = splitHash(`${SPLIT_SALT}|${this.key(rule)}`);
    return h < sub.split.thrHash || (h === sub.split.thrHash && idx <= sub.split.thrIndex);
  }

  /** A random rule of `split`, as training draws one: the subset uniform over the seven, the rule uniform in it. */
  sample(rand: () => number, split: Split): Rule {
    const s = Math.floor(rand() * this.nSub);
    for (;;) {
      const digits = this.nOpt[s].map((n) => Math.floor(rand() * n));
      const rule = { s, digits };
      if (this.heldOut(rule) === (split === 'heldout')) return rule;
    }
  }

  /** Every cell's chords under the rule: bit p = PAIRS[p]; 0 off the board. */
  render(rule: Rule, board: Board): Int16Array {
    const out = new Int16Array(board.h * board.w);
    const lp = this.subsets[rule.s].localPairs;
    for (let p = 0; p < out.length; p++) {
      const g = board.geo[p];
      if (g < 0) continue;
      const t = Math.floor(g / 12);
      const rot = Math.floor(g / 2) % 6;
      const m = g % 2 ? -1 : 1;
      let bits = 0;
      for (const [k0, k1] of lp[t][rule.digits[t]]) {
        const a = (((m * k0 + rot) % 6) + 6) % 6;
        const b = (((m * k1 + rot) % 6) + 6) % 6;
        bits |= 1 << PAIR_INDEX[a * 6 + b];
      }
      out[p] = bits;
    }
    return out;
  }

  /** Int8 [6 * H * W]: exits[d * H * W + p] = the edge a line entering cell p across edge d leaves by, -1 none. */
  exits(rule: Rule, board: Board): Int8Array {
    return exitsOf(this.render(rule, board), board.h * board.w);
  }
}

export function exitsOf(bits: Int16Array, n: number): Int8Array {
  const ex = new Int8Array(6 * n).fill(-1);
  for (let p = 0; p < n; p++) {
    const b = bits[p];
    if (!b) continue;
    for (let k = 0; k < 15; k++) {
      if (!((b >> k) & 1)) continue;
      const [a, c] = PAIRS[k];
      ex[a * n + p] = c;
      ex[c * n + p] = a;
    }
  }
  return ex;
}

/** One of the page's boards: the H×W array, its on-board cells and their neighbours. */
export class Board {
  readonly h: number;
  readonly w: number;
  readonly geo: Int16Array;
  /** cellOf[row * w + col] = the cell's index among the on-board cells, -1 off the board. */
  readonly cellOf: Int32Array;
  /** pos[i] = row * w + col of cell i. */
  readonly pos: Int32Array;
  /** nbr[i * 6 + d] = the cell across direction d, -1 off the board. */
  readonly nbr: Int32Array;
  readonly n: number;

  constructor(readonly data: BoardData) {
    this.h = data.h;
    this.w = data.w;
    this.geo = Int16Array.from(data.geo);
    this.cellOf = new Int32Array(this.h * this.w).fill(-1);
    const pos: number[] = [];
    for (let p = 0; p < this.geo.length; p++) if (this.geo[p] >= 0) { this.cellOf[p] = pos.length; pos.push(p); }
    this.pos = Int32Array.from(pos);
    this.n = pos.length;
    this.nbr = new Int32Array(this.n * 6).fill(-1);
    for (let i = 0; i < this.n; i++) {
      const row = Math.floor(this.pos[i] / this.w);
      const col = this.pos[i] % this.w;
      for (let d = 0; d < 6; d++) {
        const r = row + DROW[d];
        const c = col + DCOL[d];
        if (r >= 0 && r < this.h && c >= 0 && c < this.w) this.nbr[i * 6 + d] = this.cellOf[r * this.w + c];
      }
    }
  }

  on(row: number, col: number): boolean {
    return row >= 0 && row < this.h && col >= 0 && col < this.w && this.geo[row * this.w + col] >= 0;
  }

  type(p: number): number { return Math.floor(this.geo[p] / 12); }
  rot(p: number): number { return Math.floor(this.geo[p] / 2) % 6; }
  /** The board direction of the cell's local edge k. */
  edgeDir(p: number, k: number): number {
    const m = this.geo[p] % 2 ? -1 : 1;
    return (((m * k + this.rot(p)) % 6) + 6) % 6;
  }
}

/** One strand in walking order, from the tap's d0 end to its d1 end (walker.Strand). index: signed steps from the
 * tap (a loop: 0..n-1). */
export interface Strand {
  rows: number[];
  cols: number[];
  ins: number[];
  outs: number[];
  index: number[];
  closed: boolean;
}

function oneWay(ex: Int8Array, board: Board, row: number, col: number, din: number, dout: number, limit: number):
  { steps: [number, number, number, number][]; closed: boolean } {
  const n = board.h * board.w;
  const steps: [number, number, number, number][] = [[row, col, din, dout]];
  const seen = new Set<number>([(row * board.w + col) * 6 + din, (row * board.w + col) * 6 + dout]);
  let r = row;
  let c = col;
  let d = dout;
  while (steps.length < limit) {
    const nr = r + DROW[d];
    const nc = c + DCOL[d];
    if (!board.on(nr, nc)) return { steps, closed: false };
    const ein = (d + 3) % 6;
    const p = nr * board.w + nc;
    const eout = ex[ein * n + p];
    if (eout < 0) return { steps, closed: false };
    if (nr === row && nc === col && (ein === din || ein === dout)) return { steps, closed: true };
    if (seen.has(p * 6 + ein)) throw new Error('walk re-entered a chord other than its first (the bits branch)');
    seen.add(p * 6 + ein);
    seen.add(p * 6 + eout);
    steps.push([nr, nc, ein, eout]);
    r = nr;
    c = nc;
    d = eout;
  }
  throw new Error(`walk longer than ${limit} steps`);
}

/** The strand through chord (d0, d1) of cell (row, col), both ways (walker.walk). */
export function walk(ex: Int8Array, board: Board, row: number, col: number, d0: number, d1: number,
  limit = 1 << 22): Strand {
  const n = board.h * board.w;
  if (ex[d0 * n + row * board.w + col] !== d1) throw new Error(`no chord (${d0}, ${d1}) at (${row}, ${col})`);
  const ahead = oneWay(ex, board, row, col, d0, d1, limit);
  let behind: [number, number, number, number][] = [];
  let closed = ahead.closed;
  if (!closed) {
    const back = oneWay(ex, board, row, col, d1, d0, limit);
    if (back.closed) throw new Error('closed one way, open the other');
    behind = back.steps.slice(1).reverse().map(([r, c, din, dout]) => [r, c, dout, din]);
  }
  const seq = behind.concat(ahead.steps);
  return {
    rows: seq.map((x) => x[0]), cols: seq.map((x) => x[1]), ins: seq.map((x) => x[2]), outs: seq.map((x) => x[3]),
    index: seq.map((_, k) => k - behind.length), closed,
  };
}
