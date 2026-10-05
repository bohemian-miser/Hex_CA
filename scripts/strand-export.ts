/**
 * Strand data for the M1 neural CA (docs/spectacle-nca-plan.md §4 M1a/M1b, §7; docs/strand-data.md).
 *
 * Builds Spectacle's hex fields (levels 2, 3 and crops of level 4), maps every tile onto
 * hex_ca's axial lattice, renders each clean hex rule without edge class 0 as 15 chord
 * bits per cell, and walks strands with Spectacle's own `walkStrand` (the oracle). One
 * compressed .npz per (split, level, rule); `meta.json` describes rules, boards and the
 * conventions. Spectacle's code is imported read-only by path (SPECTACLE_DIR or
 * --spectacle, default ../Spectacle), so this file typechecks and CI runs without it.
 *
 *   npx tsx scripts/strand-export.ts [--out data/strand] [--taps 32] [--crops 16] [--seed 1]
 *       [--eval-taps 64] [--eval-crops 16] [--crop-radius 8-14] [--rules 0-99] [--spectacle DIR]
 *
 * Conventions (identical to nca/hexgrid.py and src/hex.ts):
 *   axial (q, r); array index row = r - r0, col = q - q0 (r0, q0 stored per board, i.e. the
 *   "offset" of hexgrid.py is -r0 / -q0); direction d in 0..5 is src/hex.ts DIRS[d] as (dq, dr):
 *   0 (1,0)  1 (1,-1)  2 (0,-1)  3 (-1,0)  4 (-1,1)  5 (0,1); opposite = (d + 3) % 6.
 *   Chord bit p in 0..14 is the unordered direction pair PAIRS[p], (0,1),(0,2),..,(4,5).
 *   Spectacle world -> axial: with (x, y) a tile centre minus the field's tile-0 centre,
 *   q = (x / 1.5 + 2y / sqrt 3) / 2, r = (x / 1.5 - 2y / sqrt 3) / 2, so the world direction at
 *   angle 30° + 60°·d (the hexagons are flat-top, edge length 1, centre spacing sqrt 3) is d.
 *   Each board then takes the lattice symmetry with the smallest array box (`board_orient`,
 *   see `orientMap`); every plane, direction and step of that board is in its frame.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { deflateRawSync } from 'node:zlib';

// ---------------------------------------------------------------------------------------------
// The slice of Spectacle's API used here (imported dynamically, so typed by hand).

interface Pt {
  x: number;
  y: number;
}
type Segment = readonly [Pt, Pt];
interface Field {
  spec: { family: string; level: number; rootTile: string };
  family: string;
  count: number;
  leafTypes: readonly string[];
  types: Uint8Array;
  xforms: Float64Array;
  centers: Float64Array;
  nbrStart: Int32Array;
  nbrs: Int32Array;
}
interface PlayerRule {
  family: string;
  subset: readonly number[];
  matching: readonly number[];
}
interface ChordTable {
  byType: readonly (readonly Segment[])[];
}
interface WalkStep {
  tile: number;
  chord: number;
  a: Pt;
  b: Pt;
}
interface FullWalk {
  steps: readonly WalkStep[];
  closed: boolean;
  stoppedAt: 'closed' | 'dead' | 'junction' | 'limit';
}
interface Spectacle {
  buildField(spec: { family: string; level: number; rootTile: string }): Field;
  chordTableFor(field: Field, rule: PlayerRule): ChordTable;
  walkStrand(field: Field, table: ChordTable, i: number, c: number, exitEnd: 0 | 1): FullWalk;
  ruleKey(rule: PlayerRule): string;
  describeRule(rule: PlayerRule): string;
  validEdgeSubsets(family: string): readonly { edges: readonly number[] }[];
  nonCrossingForTile(family: string, type: string, selected: ReadonlySet<number>): readonly number[];
  edgeLabels(family: string, type: string): readonly string[];
  HEX_LEAF_ORDER: readonly string[];
  HEX_PTS: readonly Pt[];
}

async function loadSpectacle(dir: string): Promise<Spectacle> {
  const mod = async (p: string): Promise<Record<string, unknown>> =>
    (await import(pathToFileURL(join(dir, p)).href)) as Record<string, unknown>;
  const [field, strand, rule, tiles] = await Promise.all([
    mod('shared/game/field.ts'),
    mod('shared/game/strand.ts'),
    mod('shared/game/rule.ts'),
    mod('shared/tiles/index.ts'),
  ]);
  return { ...tiles, ...rule, ...strand, ...field } as unknown as Spectacle;
}

// ---------------------------------------------------------------------------------------------
// Lattice conventions.

const DIRS: readonly (readonly [number, number])[] = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
];
const PAIRS: readonly (readonly [number, number])[] = (() => {
  const out: [number, number][] = [];
  for (let a = 0; a < 6; a++) for (let b = a + 1; b < 6; b++) out.push([a, b]);
  return out;
})();
const pairIndex = (a: number, b: number): number => {
  const [lo, hi] = a < b ? [a, b] : [b, a];
  return PAIRS.findIndex(([x, y]) => x === lo && y === hi);
};

const SQRT3 = Math.sqrt(3);
const EPS = 1e-6;

/** World offset -> fractional axial offset (see the header). */
function toAxial(x: number, y: number): [number, number] {
  const s = x / 1.5;
  const t = (2 * y) / SQRT3;
  return [(s + t) / 2, (s - t) / 2];
}

function roundChecked(v: number, what: string): number {
  const k = Math.round(v);
  if (Math.abs(v - k) > EPS) throw new Error(`${what}: off lattice by ${Math.abs(v - k)}`);
  return k;
}

/** Grid direction of a world offset that is a lattice neighbour step (or half of one, `scale` 2). */
function dirOf(dx: number, dy: number, scale: number, what: string): number {
  const [fq, fr] = toAxial(dx * scale, dy * scale);
  const dq = roundChecked(fq, what);
  const dr = roundChecked(fr, what);
  const d = DIRS.findIndex(([a, b]) => a === dq && b === dr);
  if (d < 0) throw new Error(`${what}: (${dq}, ${dr}) is not a neighbour step`);
  return d;
}

// ---------------------------------------------------------------------------------------------
// A field on the lattice.

interface Lattice {
  field: Field;
  q: Int32Array;
  r: Int32Array;
  /** perm[i * 6 + k] = grid direction of tile i's local edge k. */
  perm: Int8Array;
  /** perm[k] = (sign * k + rot) mod 6 for every tile; sign is the patch's mirror sign. */
  rot: Int8Array;
  sign: 1 | -1;
  cellOf: Map<string, number>;
  /** Facing-edge class mismatches (should be 0). */
  classMismatch: number;
}

const ckey = (q: number, r: number): string => `${q},${r}`;

function latticeOf(S: Spectacle, field: Field): Lattice {
  const n = field.count;
  const q = new Int32Array(n);
  const r = new Int32Array(n);
  const perm = new Int8Array(n * 6);
  const rot = new Int8Array(n);
  const cellOf = new Map<string, number>();
  const x0 = field.centers[0];
  const y0 = field.centers[1];
  const P = S.HEX_PTS;
  const mids = P.map((p, k) => ({ x: (p.x + P[(k + 1) % 6].x) / 2, y: (p.y + P[(k + 1) % 6].y) / 2 }));
  let sign: 1 | -1 | 0 = 0;
  for (let i = 0; i < n; i++) {
    const cx = field.centers[2 * i];
    const cy = field.centers[2 * i + 1];
    const [fq, fr] = toAxial(cx - x0, cy - y0);
    q[i] = roundChecked(fq, `tile ${i} centre`);
    r[i] = roundChecked(fr, `tile ${i} centre`);
    const key = ckey(q[i], r[i]);
    if (cellOf.has(key)) throw new Error(`tiles ${cellOf.get(key)} and ${i} share cell ${key}`);
    cellOf.set(key, i);
    const o = i * 6;
    const M = field.xforms;
    for (let k = 0; k < 6; k++) {
      const m = mids[k];
      const wx = M[o] * m.x + M[o + 1] * m.y + M[o + 2];
      const wy = M[o + 3] * m.x + M[o + 4] * m.y + M[o + 5];
      perm[o + k] = dirOf(wx - cx, wy - cy, 2, `tile ${i} edge ${k}`);
    }
    const det = Math.sign(M[o] * M[o + 4] - M[o + 1] * M[o + 3]) as 1 | -1;
    // Local edges run counter-clockwise (HEX_PTS); a mirror reverses them.
    const s = perm[o + 1] === (perm[o] + 1) % 6 ? 1 : -1;
    if (s !== det) throw new Error(`tile ${i}: edge order ${s} disagrees with det ${det}`);
    if (sign === 0) sign = s;
    else if (sign !== s) throw new Error(`tile ${i}: mixed mirror signs in one patch`);
    rot[i] = perm[o];
    for (let k = 0; k < 6; k++) {
      if (perm[o + k] !== (((s * k + rot[i]) % 6) + 6) % 6) throw new Error(`tile ${i}: perm not a rotation`);
    }
  }
  // Lattice neighbours present == Spectacle's vertex-sharing neighbours, and facing edges agree.
  let classMismatch = 0;
  const major = (t: number, k: number): number =>
    Number(/^-?(\d+)\./.exec(S.edgeLabels(field.family, field.leafTypes[field.types[t]])[k])![1]);
  for (let i = 0; i < n; i++) {
    const want = new Set<number>();
    for (let k = 0; k < 6; k++) {
      const d = perm[i * 6 + k];
      const j = cellOf.get(ckey(q[i] + DIRS[d][0], r[i] + DIRS[d][1]));
      if (j === undefined) continue;
      want.add(j);
      const kj = [0, 1, 2, 3, 4, 5].find((e) => perm[j * 6 + e] === (d + 3) % 6)!;
      if (major(i, k) !== major(j, kj)) classMismatch++;
    }
    const got = new Set(field.nbrs.subarray(field.nbrStart[i], field.nbrStart[i + 1]));
    if (got.size !== want.size || [...got].some((j) => !want.has(j))) {
      throw new Error(`tile ${i}: Spectacle neighbours ${[...got]} != lattice neighbours ${[...want]}`);
    }
  }
  return { field, q, r, perm, rot, sign: sign as 1 | -1, cellOf, classMismatch };
}

/** A Field restricted to some of its tiles (walkStrand only reads these members). */
function subField(field: Field, keep: readonly number[]): { field: Field; old: Int32Array } {
  const idx = new Map<number, number>();
  keep.forEach((t, k) => idx.set(t, k));
  const n = keep.length;
  const types = new Uint8Array(n);
  const xforms = new Float64Array(n * 6);
  const centers = new Float64Array(n * 2);
  const nbrStart = new Int32Array(n + 1);
  const nbrs: number[] = [];
  keep.forEach((t, k) => {
    types[k] = field.types[t];
    xforms.set(field.xforms.subarray(t * 6, t * 6 + 6), k * 6);
    centers.set(field.centers.subarray(t * 2, t * 2 + 2), k * 2);
    for (const j of field.nbrs.subarray(field.nbrStart[t], field.nbrStart[t + 1])) {
      const m = idx.get(j);
      if (m !== undefined) nbrs.push(m);
    }
    nbrStart[k + 1] = nbrs.length;
  });
  return {
    field: { ...field, count: n, types, xforms, centers, nbrStart, nbrs: Int32Array.from(nbrs) },
    old: Int32Array.from(keep),
  };
}

// ---------------------------------------------------------------------------------------------
// Rules.

interface RuleRec {
  id: number;
  rule: PlayerRule;
  key: string;
  describe: string;
  subset: string;
  split: 'train' | 'heldout';
  /** Per leaf type (HEX_LEAF_ORDER): chords as [local edge of end 0, local edge of end 1]. */
  localPairs: number[][][];
}

const SPLIT_SALT = 'strand-split-v1';
const HELDOUT_SHARE = 0.2;

/** FNV-1a, then murmur3's fmix32 (bare FNV-1a keeps keys that differ in the last digit adjacent). */
function splitHash(s: string): number {
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

function enumerateRules(S: Spectacle): RuleRec[] {
  const order = S.HEX_LEAF_ORDER;
  const P = S.HEX_PTS;
  const mids = P.map((p, k) => ({ x: (p.x + P[(k + 1) % 6].x) / 2, y: (p.y + P[(k + 1) % 6].y) / 2 }));
  const edgeAt = (p: Pt): number => {
    const k = mids.findIndex((m) => Math.abs(m.x - p.x) < EPS && Math.abs(m.y - p.y) < EPS);
    if (k < 0) throw new Error(`chord end (${p.x}, ${p.y}) is not an edge midpoint`);
    return k;
  };
  const probe = S.buildField({ family: 'hex', level: 0, rootTile: 'Delta' });
  const out: RuleRec[] = [];
  const subsets = S.validEdgeSubsets('hex').filter((v) => v.edges.length > 0 && !v.edges.includes(0));
  for (const { edges } of subsets) {
    const selected = new Set(edges);
    const allowed = order.map((t) => {
      const a = S.nonCrossingForTile('hex', t, selected);
      return a.length ? a : [0];
    });
    const recs: RuleRec[] = [];
    const walk = (i: number, acc: number[]): void => {
      if (i === order.length) {
        const rule: PlayerRule = { family: 'hex', subset: [...edges], matching: [...acc] };
        const table = S.chordTableFor(probe, rule);
        const localPairs = table.byType.map((segs) => segs.map(([a, b]) => [edgeAt(a), edgeAt(b)]));
        recs.push({
          id: -1,
          rule,
          key: S.ruleKey(rule),
          describe: S.describeRule(rule),
          subset: edges.join(''),
          split: 'train',
          localPairs,
        });
        return;
      }
      for (const m of allowed[i]) walk(i + 1, [...acc, m]);
    };
    walk(0, []);
    // Deterministic split: within each subset, the round(20 %) rules with the smallest
    // splitHash(`${SPLIT_SALT}|${ruleKey}`) are held out.
    const k = Math.round(HELDOUT_SHARE * recs.length);
    const hash = (r: RuleRec): number => splitHash(`${SPLIT_SALT}|${r.key}`);
    const byHash = [...recs].sort((a, b) => hash(a) - hash(b));
    for (const r of byHash.slice(0, k)) r.split = 'heldout';
    out.push(...recs);
  }
  out.forEach((r, i) => (r.id = i));
  return out;
}

// ---------------------------------------------------------------------------------------------
// Boards: a full patch, or a crop of one (a union of 1-3 hex discs, cut by the patch).

interface Board {
  level: number;
  root: number;
  kind: 'full' | 'crop';
  lat: Lattice;
  /** The walkable field (a sub-field for a crop) and its tiles' indices in lat.field. */
  field: Field;
  old: Int32Array;
  /** Board-frame axial per board tile: the lattice's (q, r) under the board's orientation. */
  qa: Int32Array;
  ra: Int32Array;
  /** Base direction -> board direction under the orientation. */
  g: Int8Array;
  /** Orientation m * 6 + k: swap (q, r) if m, then k times (q, r) -> (-r, q + r). */
  orient: number;
  /** Board-frame mirror sign: grid direction of local edge k = (mirror * k + tile_rot) mod 6. */
  mirror: 1 | -1;
  q0: number;
  r0: number;
  h: number;
  w: number;
  crop?: { discs: [number, number, number][] };
}

function orientMap(orient: number): (q: number, r: number) => [number, number] {
  const m = Math.floor(orient / 6);
  const k = orient % 6;
  return (q, r) => {
    let [a, b] = m ? [r, q] : [q, r];
    for (let i = 0; i < k; i++) [a, b] = [-b, a + b];
    return [a, b];
  };
}

/**
 * A board on a set of tiles (null = the whole field). Of the lattice's 12 symmetries it takes
 * the one with the smallest array box (ties: the lowest orient): the CA works in the array, so
 * the frame is free, and a level-3 patch needs 30 x 37 cells this way instead of 37 x 38.
 */
function boardOf(lat: Lattice, level: number, root: number, keep: readonly number[] | null, crop?: Board['crop']): Board {
  const { field, old } = keep
    ? subField(lat.field, keep)
    : { field: lat.field, old: Int32Array.from({ length: lat.field.count }, (_, i) => i) };
  let best: { orient: number; area: number; q0: number; r0: number; h: number; w: number } | null = null;
  for (let orient = 0; orient < 12; orient++) {
    const T = orientMap(orient);
    let qmin = Infinity;
    let qmax = -Infinity;
    let rmin = Infinity;
    let rmax = -Infinity;
    for (const t of old) {
      const [q, r] = T(lat.q[t], lat.r[t]);
      qmin = Math.min(qmin, q);
      qmax = Math.max(qmax, q);
      rmin = Math.min(rmin, r);
      rmax = Math.max(rmax, r);
    }
    const h = rmax - rmin + 1;
    const w = qmax - qmin + 1;
    if (!best || h * w < best.area) best = { orient, area: h * w, q0: qmin, r0: rmin, h, w };
  }
  const { orient, q0, r0, h, w } = best!;
  const T = orientMap(orient);
  const qa = new Int32Array(old.length);
  const ra = new Int32Array(old.length);
  old.forEach((t, i) => ([qa[i], ra[i]] = T(lat.q[t], lat.r[t])));
  const g = Int8Array.from(DIRS.map(([dq, dr]) => {
    const [a, b] = T(dq, dr);
    return DIRS.findIndex(([x, y]) => x === a && y === b);
  }));
  const mirror = (lat.sign * (g[1] === (g[0] + 1) % 6 ? 1 : -1)) as 1 | -1;
  return { level, root, kind: keep ? 'crop' : 'full', lat, field, old, qa, ra, g, orient, mirror, q0, r0, h, w, crop };
}

const hexDist = (dq: number, dr: number): number => Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));

function cropBoard(lat: Lattice, level: number, root: number, rng: () => number, rlo: number, rhi: number): Board {
  for (;;) {
    const n = lat.field.count;
    const nd = 1 + Math.floor(rng() * 3);
    const discs: [number, number, number][] = [];
    const c0 = Math.floor(rng() * n);
    const R0 = rlo + Math.floor(rng() * (rhi - rlo + 1));
    discs.push([lat.q[c0], lat.r[c0], R0]);
    for (let k = 1; k < nd; k++) {
      // Further discs centred within the first one: concave unions, same overall size.
      const R = Math.max(3, Math.round(R0 * (0.4 + 0.4 * rng())));
      let c = c0;
      for (let tries = 0; tries < 100; tries++) {
        const t = Math.floor(rng() * n);
        if (hexDist(lat.q[t] - lat.q[c0], lat.r[t] - lat.r[c0]) <= R0) {
          c = t;
          break;
        }
      }
      discs.push([lat.q[c], lat.r[c], R]);
    }
    const keep: number[] = [];
    for (let t = 0; t < n; t++) {
      if (discs.some(([cq, cr, R]) => hexDist(lat.q[t] - cq, lat.r[t] - cr) <= R)) keep.push(t);
    }
    if (keep.length >= 3 * rlo * (rlo + 1) * 0.5) return boardOf(lat, level, root, keep, { discs });
  }
}

// ---------------------------------------------------------------------------------------------
// Walking, per (board, rule).

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Accum {
  boards: Board[];
  chords: Uint8Array[];
  chordStrand: Int32Array[];
  strandLen: number[];
  strandClosed: number[];
  strandBoard: number[];
  tap: { board: number[]; row: number[]; col: number[]; d0: number[]; d1: number[]; closed: number[]; strand: number[]; ptr: number[] };
  step: { row: number[]; col: number[]; din: number[]; dout: number[]; index: number[] };
  stops: Record<string, number>;
}

function newAccum(): Accum {
  return {
    boards: [],
    chords: [],
    chordStrand: [],
    strandLen: [],
    strandClosed: [],
    strandBoard: [],
    tap: { board: [], row: [], col: [], d0: [], d1: [], closed: [], strand: [], ptr: [0] },
    step: { row: [], col: [], din: [], dout: [], index: [] },
    stops: {},
  };
}

function addBoard(S: Spectacle, acc: Accum, b: Board, rule: PlayerRule, taps: number, rng: () => number): void {
  const bi = acc.boards.length;
  acc.boards.push(b);
  const { field } = b;
  const table = S.chordTableFor(field, rule);
  const HW = b.h * b.w;
  const chords = new Uint8Array(15 * HW);
  const chordStrand = new Int32Array(15 * HW).fill(-1);
  const rowOf = (t: number): number => b.ra[t] - b.r0;
  const colOf = (t: number): number => b.qa[t] - b.q0;
  const cell = (t: number): number => rowOf(t) * b.w + colOf(t);
  /** Board direction of a chord end (world point) on tile t: geometry only, not the chord table. */
  const dirAt = (t: number, p: Pt): number =>
    b.g[dirOf(p.x - field.centers[2 * t], p.y - field.centers[2 * t + 1], 2, `chord end on tile ${t}`)];
  const pairOfChord = (t: number, c: number): number => {
    const seg = table.byType[field.types[t]][c];
    const [p0, p1] = seg;
    const M = field.xforms;
    const o = t * 6;
    const w = (p: Pt): Pt => ({ x: M[o] * p.x + M[o + 1] * p.y + M[o + 2], y: M[o + 3] * p.x + M[o + 4] * p.y + M[o + 5] });
    return pairIndex(dirAt(t, w(p0)), dirAt(t, w(p1)));
  };
  // Chord planes, rendered from the chord table and each tile's rotation.
  const all: [number, number][] = [];
  for (let t = 0; t < field.count; t++) {
    const segs = table.byType[field.types[t]];
    const used = new Set<number>();
    for (let c = 0; c < segs.length; c++) {
      const p = pairOfChord(t, c);
      for (const d of PAIRS[p]) {
        if (used.has(d)) throw new Error(`tile ${t}: two chords use direction ${d}`);
        used.add(d);
      }
      chords[p * HW + cell(t)] = 1;
      all.push([t, c]);
    }
  }
  // Whole-board decomposition into strands (M1b's "no tap" target), by walkStrand.
  const sid = new Map<number, number>();
  const toSteps = (fw: FullWalk, bw: FullWalk | null): { t: number; c: number }[] => {
    const back = bw ? bw.steps.slice(1).reverse() : [];
    return [...back, ...fw.steps].map((s) => ({ t: s.tile, c: s.chord }));
  };
  const note = (w: FullWalk): void => {
    acc.stops[w.stoppedAt] = (acc.stops[w.stoppedAt] ?? 0) + 1;
    if (w.stoppedAt === 'junction' || w.stoppedAt === 'limit') throw new Error(`walk stopped at ${w.stoppedAt}`);
  };
  for (const [t, c] of all) {
    if (sid.has(t * 4 + c)) continue;
    const fw = S.walkStrand(field, table, t, c, 1);
    note(fw);
    const bw = fw.closed ? null : S.walkStrand(field, table, t, c, 0);
    if (bw) {
      note(bw);
      if (bw.closed) throw new Error('backward walk closed, forward did not');
    }
    const id = acc.strandLen.length;
    const steps = toSteps(fw, bw);
    for (const s of steps) {
      sid.set(s.t * 4 + s.c, id);
      chordStrand[pairOfChord(s.t, s.c) * HW + cell(s.t)] = id;
    }
    acc.strandLen.push(steps.length);
    acc.strandClosed.push(fw.closed ? 1 : 0);
    acc.strandBoard.push(bi);
  }
  // Taps: uniform over the board's chords, without replacement; each walked both ways.
  const pick = [...all];
  const n = Math.min(taps, pick.length);
  for (let k = 0; k < n; k++) {
    const j = k + Math.floor(rng() * (pick.length - k));
    [pick[k], pick[j]] = [pick[j], pick[k]];
    const [t, c] = pick[k];
    const fw = S.walkStrand(field, table, t, c, 1);
    const bw = fw.closed ? null : S.walkStrand(field, table, t, c, 0);
    // Forward orientation throughout: backward steps reversed, entry and exit swapped.
    const seq: { s: WalkStep; flip: boolean; index: number }[] = [];
    if (bw) bw.steps.slice(1).forEach((s, i) => seq.unshift({ s, flip: true, index: -(i + 1) }));
    fw.steps.forEach((s, i) => seq.push({ s, flip: false, index: i }));
    const tap0 = fw.steps[0];
    acc.tap.board.push(bi);
    acc.tap.row.push(rowOf(t));
    acc.tap.col.push(colOf(t));
    acc.tap.d0.push(dirAt(t, tap0.a));
    acc.tap.d1.push(dirAt(t, tap0.b));
    acc.tap.closed.push(fw.closed ? 1 : 0);
    acc.tap.strand.push(sid.get(t * 4 + c)!);
    for (const { s, flip, index } of seq) {
      const din = dirAt(s.tile, flip ? s.b : s.a);
      const dout = dirAt(s.tile, flip ? s.a : s.b);
      acc.step.row.push(rowOf(s.tile));
      acc.step.col.push(colOf(s.tile));
      acc.step.din.push(din);
      acc.step.dout.push(dout);
      acc.step.index.push(index);
    }
    acc.tap.ptr.push(acc.step.row.length);
  }
  acc.chords.push(chords);
  acc.chordStrand.push(chordStrand);
}

// ---------------------------------------------------------------------------------------------
// .npy / .npz writing (zip with deflate; numpy.load reads it).

type Arr = Uint8Array | Int8Array | Int16Array | Int32Array | Float32Array;
const DESCR = new Map<unknown, string>([
  [Uint8Array, '|u1'],
  [Int8Array, '|i1'],
  [Int16Array, '<i2'],
  [Int32Array, '<i4'],
  [Float32Array, '<f4'],
]);

function npy(a: Arr, shape: readonly number[]): Buffer {
  if (shape.reduce((x, y) => x * y, 1) !== a.length) throw new Error(`shape ${shape} != length ${a.length}`);
  const sh = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '${DESCR.get(a.constructor)}', 'fortran_order': False, 'shape': ${sh}, }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += ' '.repeat(pad % 64) + '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1;
  head[7] = 0;
  head.writeUInt16LE(header.length, 8);
  return Buffer.concat([head, Buffer.from(header, 'latin1'), Buffer.from(a.buffer, a.byteOffset, a.byteLength)]);
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function npz(entries: Record<string, [Arr, readonly number[]]>): Buffer {
  const parts: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const [name, [a, shape]] of Object.entries(entries)) {
    const raw = npy(a, shape);
    const data = deflateRawSync(raw, { level: 6 });
    const fname = Buffer.from(`${name}.npy`, 'utf8');
    const crc = crc32(raw);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(8, 8);
    local.writeUInt16LE(0, 10);
    local.writeUInt16LE(0x21, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(fname.length, 26);
    local.writeUInt16LE(0, 28);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(0, 8);
    cd.writeUInt16LE(8, 10);
    cd.writeUInt16LE(0, 12);
    cd.writeUInt16LE(0x21, 14);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(data.length, 20);
    cd.writeUInt32LE(raw.length, 24);
    cd.writeUInt16LE(fname.length, 28);
    cd.writeUInt32LE(offset, 42);
    parts.push(local, fname, data);
    central.push(cd, fname);
    offset += local.length + fname.length + data.length;
  }
  const cdBuf = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(Object.keys(entries).length, 8);
  end.writeUInt16LE(Object.keys(entries).length, 10);
  end.writeUInt32LE(cdBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...parts, cdBuf, end]);
}

/** Pack an accumulated (split, level, rule) into one .npz; arrays padded to the largest board. */
function packFile(acc: Accum, rule: RuleRec, level: number): Buffer {
  const B = acc.boards.length;
  const H = Math.max(...acc.boards.map((b) => b.h));
  const W = Math.max(...acc.boards.map((b) => b.w));
  const mask = new Uint8Array(B * H * W);
  const chords = new Uint8Array(B * 15 * H * W);
  const chordStrand = new Int32Array(B * 15 * H * W).fill(-1);
  const tileIndex = new Int32Array(B * H * W).fill(-1);
  const tileType = new Int8Array(B * H * W).fill(-1);
  const tileRot = new Int8Array(B * H * W).fill(-1);
  acc.boards.forEach((b, bi) => {
    const HW = b.h * b.w;
    b.old.forEach((t, i) => {
      const o = bi * H * W + (b.ra[i] - b.r0) * W + (b.qa[i] - b.q0);
      mask[o] = 1;
      tileIndex[o] = t;
      tileType[o] = b.lat.field.types[t];
      tileRot[o] = b.g[b.lat.rot[t]];
    });
    for (let p = 0; p < 15; p++) {
      for (let row = 0; row < b.h; row++) {
        for (let col = 0; col < b.w; col++) {
          const src = p * HW + row * b.w + col;
          const dst = ((bi * 15 + p) * H + row) * W + col;
          chords[dst] = acc.chords[bi][src];
          chordStrand[dst] = acc.chordStrand[bi][src];
        }
      }
    }
  });
  const T = acc.tap.board.length;
  const N = acc.step.row.length;
  const i32 = (a: number[]): Int32Array => Int32Array.from(a);
  const i16 = (a: number[]): Int16Array => Int16Array.from(a);
  const i8 = (a: number[]): Int8Array => Int8Array.from(a);
  const u8 = (a: number[]): Uint8Array => Uint8Array.from(a);
  const S = acc.strandLen.length;
  return npz({
    rule_id: [Int32Array.of(rule.id), [1]],
    level: [Int32Array.of(level), [1]],
    board_root: [i8(acc.boards.map((b) => b.root)), [B]],
    board_kind: [u8(acc.boards.map((b) => (b.kind === 'full' ? 0 : 1))), [B]],
    board_orient: [i8(acc.boards.map((b) => b.orient)), [B]],
    board_mirror: [i8(acc.boards.map((b) => b.mirror)), [B]],
    board_q0: [i32(acc.boards.map((b) => b.q0)), [B]],
    board_r0: [i32(acc.boards.map((b) => b.r0)), [B]],
    board_h: [i32(acc.boards.map((b) => b.h)), [B]],
    board_w: [i32(acc.boards.map((b) => b.w)), [B]],
    board_tiles: [i32(acc.boards.map((b) => b.old.length)), [B]],
    mask: [mask, [B, H, W]],
    chords: [chords, [B, 15, H, W]],
    tile_index: [tileIndex, [B, H, W]],
    tile_type: [tileType, [B, H, W]],
    tile_rot: [tileRot, [B, H, W]],
    chord_strand: [chordStrand, [B, 15, H, W]],
    strand_len: [i32(acc.strandLen), [S]],
    strand_closed: [u8(acc.strandClosed), [S]],
    strand_board: [i32(acc.strandBoard), [S]],
    tap_board: [i32(acc.tap.board), [T]],
    tap_row: [i16(acc.tap.row), [T]],
    tap_col: [i16(acc.tap.col), [T]],
    tap_d0: [i8(acc.tap.d0), [T]],
    tap_d1: [i8(acc.tap.d1), [T]],
    tap_closed: [u8(acc.tap.closed), [T]],
    tap_strand: [i32(acc.tap.strand), [T]],
    tap_ptr: [i32(acc.tap.ptr), [T + 1]],
    step_row: [i16(acc.step.row), [N]],
    step_col: [i16(acc.step.col), [N]],
    step_in: [i8(acc.step.din), [N]],
    step_out: [i8(acc.step.dout), [N]],
    step_index: [i32(acc.step.index), [N]],
  });
}

// ---------------------------------------------------------------------------------------------
// Main.

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) throw new Error(`unexpected argument ${a}`);
    const eq = a.indexOf('=');
    if (eq > 0) out[a.slice(2, eq)] = a.slice(eq + 1);
    else out[a.slice(2)] = argv[++i];
  }
  return out;
}

function range(s: string): [number, number] {
  const [a, b] = s.split('-').map(Number);
  return [a, b ?? a];
}

async function main(): Promise<void> {
  const here = dirname(fileURLToPath(import.meta.url));
  const root = resolve(here, '..');
  const args = parseArgs(process.argv.slice(2));
  const spectacleDir = resolve(root, args.spectacle ?? process.env.SPECTACLE_DIR ?? '../Spectacle');
  const outDir = resolve(root, args.out ?? 'data/strand');
  const taps = Number(args.taps ?? 32);
  const crops = Number(args.crops ?? 16);
  const evalTaps = Number(args['eval-taps'] ?? 64);
  const evalCrops = Number(args['eval-crops'] ?? 16);
  const seed = Number(args.seed ?? 1);
  const [rlo, rhi] = range(args['crop-radius'] ?? '8-14');
  const [idLo, idHi] = range(args.rules ?? '0-99');
  const t0 = performance.now();

  const S = await loadSpectacle(spectacleDir);
  const rules = enumerateRules(S);
  const leaf = S.HEX_LEAF_ORDER;

  // Fields and lattices (all nine roots per level).
  const lattices = new Map<string, Lattice>();
  const latticeStats: Record<string, unknown>[] = [];
  for (const level of [2, 3, 4]) {
    for (let ri = 0; ri < leaf.length; ri++) {
      const lat = latticeOf(S, S.buildField({ family: 'hex', level, rootTile: leaf[ri] }));
      lattices.set(`${level}/${ri}`, lat);
      const b = boardOf(lat, level, ri, null);
      latticeStats.push({
        level,
        root: leaf[ri],
        tiles: lat.field.count,
        box: [b.h, b.w],
        orient: b.orient,
        mirror: b.mirror,
        typesPresent: [...new Set(lat.field.types)].sort().map((t) => leaf[t]),
        classMismatch: lat.classMismatch,
      });
    }
  }
  const tLattice = performance.now();

  // Boards. train/ = the TRAIN rules: levels 2 and 3 are the nine full root patches, level 4 is
  // crops (seeded by (seed, rule, level)). eval/ = the HELD-OUT rules only, fixed seed 20261006:
  // levels 2 and 3 as above with more taps, level 4 = the full Delta patch + crops.
  const EVAL_SEED = 20261006;
  type Job = { split: 'train' | 'eval'; level: number; rule: RuleRec };
  const jobs: Job[] = [];
  for (const rule of rules) {
    if (rule.id < idLo || rule.id > idHi) continue;
    for (const level of [2, 3, 4]) {
      if (rule.split === 'train') jobs.push({ split: 'train', level, rule });
      else jobs.push({ split: 'eval', level, rule });
    }
  }
  const files: Record<string, unknown>[] = [];
  const stops: Record<string, number> = {};
  let bytes = 0;
  for (const job of jobs) {
    const { split, level, rule } = job;
    const isEval = split === 'eval';
    const rng = mulberry32(((isEval ? EVAL_SEED : seed) * 1000003 + rule.id * 101 + level) >>> 0);
    const nTaps = isEval ? evalTaps : taps;
    const acc = newAccum();
    if (level < 4) {
      for (let ri = 0; ri < leaf.length; ri++) {
        addBoard(S, acc, boardOf(lattices.get(`${level}/${ri}`)!, level, ri, null), rule.rule, nTaps, rng);
      }
    } else {
      if (isEval) addBoard(S, acc, boardOf(lattices.get(`4/0`)!, 4, 0, null), rule.rule, nTaps, rng);
      const n = isEval ? evalCrops : crops;
      for (let k = 0; k < n; k++) {
        const ri = Math.floor(rng() * leaf.length);
        addBoard(S, acc, cropBoard(lattices.get(`4/${ri}`)!, 4, ri, rng, rlo, rhi), rule.rule, nTaps, rng);
      }
    }
    for (const [k, v] of Object.entries(acc.stops)) stops[k] = (stops[k] ?? 0) + v;
    const rel = `${split}/L${level}/r${String(rule.id).padStart(3, '0')}.npz`;
    const buf = packFile(acc, rule, level);
    mkdirSync(dirname(join(outDir, rel)), { recursive: true });
    writeFileSync(join(outDir, rel), buf);
    bytes += buf.length;
    files.push({
      path: rel,
      split,
      level,
      rule: rule.id,
      boards: acc.boards.length,
      taps: acc.tap.board.length,
      steps: acc.step.row.length,
      strands: acc.strandLen.length,
      bytes: buf.length,
    });
    process.stdout.write(`\r${files.length}/${jobs.length} ${rel}   `);
  }
  process.stdout.write('\n');
  const tEnd = performance.now();

  const meta = {
    version: 1,
    generated: new Date().toISOString(),
    spectacle: spectacleDir,
    args: { taps, crops, evalTaps, evalCrops, seed, evalSeed: EVAL_SEED, cropRadius: [rlo, rhi], rules: [idLo, idHi] },
    conventions: {
      dirs_dq_dr: DIRS,
      pairs: PAIRS,
      index: 'row = r - board_r0, col = q - board_q0; (q, r) = orient(base axial), base axial of tile 0 of the field = (0, 0)',
      orient: 'board_orient = m * 6 + k: swap (q, r) if m, then k times (q, r) -> (-r, q + r)',
      world_to_axial: 'q = (x / 1.5 + 2y / sqrt3) / 2, r = (x / 1.5 - 2y / sqrt3) / 2, (x, y) = centre - centre of tile 0',
      tile_rot: 'board direction of local edge k = (board_mirror * k + tile_rot) mod 6',
      leaf_order: leaf,
      split: `per subset, the round(${HELDOUT_SHARE} * n) rules with the smallest fmix32(FNV-1a('${SPLIT_SALT}|' + ruleKey)) are held out`,
    },
    rules: rules.map((r) => ({
      id: r.id,
      key: r.key,
      describe: r.describe,
      subset: r.subset,
      matching: r.rule.matching,
      split: r.split,
      local_pairs: r.localPairs,
    })),
    lattices: latticeStats,
    walkStops: stops,
    timing_s: { lattices: (tLattice - t0) / 1000, files: (tEnd - tLattice) / 1000, total: (tEnd - t0) / 1000 },
    bytes,
    files,
  };
  writeFileSync(join(outDir, 'meta.json'), JSON.stringify(meta, null, 1));
  console.log(
    `${files.length} files, ${(bytes / 1e6).toFixed(1)} MB, ${((tEnd - t0) / 1000).toFixed(1)} s ` +
      `(lattices ${((tLattice - t0) / 1000).toFixed(1)} s); walk stops ${JSON.stringify(stops)}`,
  );
  for (const l of latticeStats) {
    console.log(`L${l.level} ${l.root}: ${l.tiles} tiles, box ${l.box} (orient ${l.orient}), mirror ${l.mirror}, class mismatches ${l.classMismatch}`);
  }
  const split = (s: string): number => rules.filter((r) => r.split === s).length;
  console.log(`rules: ${rules.length} (train ${split('train')}, held-out ${split('heldout')})`);
}

main().catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
