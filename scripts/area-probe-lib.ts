// The flood probe's boards and scoring (scripts/area-probe.ts, B1 of docs/spectacle-ca-hybrid.md §3.6), shared by
// its CPU run (Node) and its GL run (scripts/area-probe-page.ts, in Chromium). No I/O.
//
// A board: 1-3 strands of one random rule (one owner), each a closed loop or a rim-to-rim claim from allStrands,
// their tiles the walls, in the area layer's frame (src/game/area.ts, §3.1). A board is exact when the flood's fill
// (channel 1 > 0.5) equals one of targets()'s acceptable answers on every board cell, walls included
// (nca/evaluate.py's rule).

import type { AreaFrame } from '../src/game/area.js';
import { targets } from '../src/nca.js';
import { allStrands, Board, DCOL, DROW, RuleTable, type Rule, type Strand } from '../src/strand.js';

export type Kind = 'loop' | 'claim';
export const SIZE_BUCKETS = [0, 1, 4, 16, 64] as const; // the primary target's filled cells: 0, 1-3, 4-15, 16-63, 64+
export const TIE_BINS = [0, 0.25, 0.5, 0.75]; // second-largest / largest rim region, boards with 2+ rim regions
export const bucketOf = (n: number): number => { let b = 0; while (b + 1 < SIZE_BUCKETS.length && n >= SIZE_BUCKETS[b + 1]) b++; return b; };
export const bucketName = (b: number): string => b + 1 < SIZE_BUCKETS.length
  ? (SIZE_BUCKETS[b + 1] - 1 === SIZE_BUCKETS[b] ? `${SIZE_BUCKETS[b]}` : `${SIZE_BUCKETS[b]}-${SIZE_BUCKETS[b + 1] - 1}`)
  : `${SIZE_BUCKETS[b]}+`;
export const kindsOf = (k: Kind[]): string => (k.every((x) => x === 'loop') ? 'loops' : k.every((x) => x === 'claim') ? 'claims' : 'mixed');

export interface Region { size: number; rim: boolean }
export interface Sample {
  rule: string;
  kinds: Kind[];
  walls: Uint8Array; // S² slots
  wallCells: number;
  targets: Uint8Array[];
  fillCells: number; // the primary target's
  rimRegions: number;
  tie: number; // second-largest / largest rim region (0 with fewer than two)
  regions: Region[];
  lab: Int32Array; // per board cell, its region (-1 a wall)
}

/** A strand's ends: true where the line would carry on off the board (false: the next tile has no chord). */
function endsOf(s: Strand, board: Board): [boolean, boolean] {
  const n = s.rows.length;
  const off = (r: number, c: number, d: number) => !board.on(r + DROW[d], c + DCOL[d]);
  return [off(s.rows[0], s.cols[0], s.ins[0]), off(s.rows[n - 1], s.cols[n - 1], s.outs[n - 1])];
}

/** The 6-connected regions of the free board cells, as targets() sees them. */
function regions(board: Board, wall: Uint8Array): { regs: Region[]; lab: Int32Array } {
  const lab = new Int32Array(board.n).fill(-1);
  const regs: Region[] = [];
  const q = new Int32Array(board.n);
  for (let s = 0; s < board.n; s++) {
    if (wall[s] || lab[s] >= 0) continue;
    let h = 0;
    let t = 0;
    let size = 0;
    let rim = false;
    lab[s] = regs.length;
    q[t++] = s;
    while (h < t) {
      const i = q[h++];
      size++;
      for (let d = 0; d < 6; d++) {
        const j = board.nbr[i * 6 + d];
        if (j < 0) { rim = true; continue; }
        if (wall[j] || lab[j] >= 0) continue;
        lab[j] = regs.length;
        q[t++] = j;
      }
    }
    regs.push({ size, rim });
  }
  return { regs, lab };
}

function sampleBoard(table: RuleTable, board: Board, frame: AreaFrame, rand: () => number,
  cache: Map<string, { loops: Strand[]; claims: Strand[] }>): Sample {
  for (;;) {
    const s = Math.floor(rand() * table.nSub);
    const rule: Rule = { s, digits: table.nOpt[s].map((n) => Math.floor(rand() * n)) };
    const key = table.describe(rule);
    let kinds = cache.get(key);
    if (!kinds) {
      const loops: Strand[] = [];
      const claims: Strand[] = [];
      for (const st of allStrands(table.exits(rule, board), board)) {
        if (st.closed) loops.push(st);
        else {
          const [a, b] = endsOf(st, board);
          if (a && b) claims.push(st);
        }
      }
      kinds = { loops, claims };
      if (cache.size > 2000) cache.clear();
      cache.set(key, kinds);
    }
    if (!kinds.loops.length && !kinds.claims.length) continue;
    const k = 1 + Math.floor(rand() * 3);
    const pool = { loop: kinds.loops.slice(), claim: kinds.claims.slice() };
    const picked: { kind: Kind; st: Strand }[] = [];
    for (let i = 0; i < k; i++) {
      const avail = (['loop', 'claim'] as Kind[]).filter((x) => pool[x].length);
      if (!avail.length) break;
      const kind = avail[Math.floor(rand() * avail.length)];
      const j = Math.floor(rand() * pool[kind].length);
      picked.push({ kind, st: pool[kind][j] });
      pool[kind].splice(j, 1);
    }
    const wallCell = new Uint8Array(board.n);
    for (const { st } of picked) {
      for (let t = 0; t < st.rows.length; t++) wallCell[board.cellOf[st.rows[t] * board.w + st.cols[t]]] = 1;
    }
    const walls = new Uint8Array(frame.S * frame.S);
    let wallCells = 0;
    for (let i = 0; i < board.n; i++) if (wallCell[i]) { walls[frame.slot[i]] = 1; wallCells++; }
    const tg = targets(walls, frame.R, frame.mask);
    let fillCells = 0;
    for (const v of tg[0]) fillCells += v;
    const { regs, lab } = regions(board, wallCell);
    const rims = regs.filter((r) => r.rim).map((r) => r.size).sort((a, b) => b - a);
    return {
      rule: key, kinds: picked.map((p) => p.kind), walls, wallCells, targets: tg, fillCells, rimRegions: rims.length,
      tie: rims.length >= 2 ? rims[1] / rims[0] : 0, regions: regs, lab,
    };
  }
}

/** n boards whose primary target fills at least `minFill` cells, deterministic given `rand`. */
export function sampleBoards(table: RuleTable, board: Board, frame: AreaFrame, n: number, rand: () => number, minFill = 0): Sample[] {
  const cache = new Map<string, { loops: Strand[]; claims: Strand[] }>();
  const out: Sample[] = [];
  while (out.length < n) {
    const smp = sampleBoard(table, board, frame, rand, cache);
    if (smp.fillCells >= minFill) out.push(smp);
  }
  return out;
}

/** What the probe needs of a flood: load a wall picture (from the fresh state), step, read channel 1 (S² slots). */
export interface Runner {
  load(walls: Uint8Array): void;
  /** Steps since the last load. */
  readonly steps: number;
  advance(n: number): void;
  ch1(): Float32Array;
}

export interface Tally { exact: number; n: number }
const tally = (): Tally => ({ exact: 0, n: 0 });
const add = (t: Tally, ok: boolean) => { t.n++; if (ok) t.exact++; };
export const fmt = (t: Tally): string => {
  if (!t.n) return '   —      ';
  const p = t.exact / t.n;
  const se = Math.sqrt((p * (1 - p)) / t.n);
  return `${p.toFixed(3)}±${(2 * se).toFixed(3)} (${t.n})`;
};

export interface Miss { rule: string; kinds: Kind[]; fillCells: number; wrong: number; tie: number; wallFilled: number; regions: number; off: string[] }
export interface Result {
  level: string; weights: string; steps: number; mult: number; R: number;
  all: Tally; nontrivial: Tally; byBucket: Tally[]; byKind: Record<string, Tally>; byTie: Tally[];
  wrongCells: number; boardCells: number; msPerStep: number; misses: Miss[];
}

/** The probe of one weights set on one level's boards, read out at mult·R steps for each mult. */
export function probe(level: string, weights: string, board: Board, frame: AreaFrame, samples: Sample[], mults: number[],
  runner: Runner): Result[] {
  const res: Result[] = mults.map((mult) => ({
    level, weights, mult, steps: mult * frame.R, R: frame.R, all: tally(), nontrivial: tally(),
    byBucket: SIZE_BUCKETS.map(tally), byKind: { loops: tally(), claims: tally(), mixed: tally() }, byTie: TIE_BINS.map(tally),
    wrongCells: 0, boardCells: 0, msPerStep: 0, misses: [],
  }));
  const { slot } = frame;
  let stepMs = 0;
  for (const smp of samples) {
    runner.load(smp.walls);
    for (let m = 0; m < mults.length; m++) {
      const t0 = performance.now();
      runner.advance(res[m].steps - runner.steps);
      const ch1 = runner.ch1(); // (timed with the steps: on the GPU, the read is where they finish)
      stepMs += performance.now() - t0;
      let wrong = Infinity;
      for (const t of smp.targets) {
        let wr = 0;
        for (let i = 0; i < board.n; i++) if ((ch1[slot[i]] > 0.5 ? 1 : 0) !== t[slot[i]]) wr++;
        wrong = Math.min(wrong, wr);
      }
      const ok = wrong === 0;
      const r = res[m];
      add(r.all, ok);
      if (smp.fillCells > 0) add(r.nontrivial, ok);
      add(r.byBucket[bucketOf(smp.fillCells)], ok);
      add(r.byKind[kindsOf(smp.kinds)], ok);
      if (smp.rimRegions >= 2) add(r.byTie[Math.min(TIE_BINS.length - 1, Math.floor(smp.tie * 4))], ok);
      r.wrongCells += wrong;
      r.boardCells += board.n;
      if (!ok) {
        // Region by region: size, rim or enclosed, what the primary target wants, the share the flood filled.
        const t0 = smp.targets[0];
        const got = smp.regions.map(() => 0);
        const want = smp.regions.map(() => 0);
        let wallFilled = 0;
        for (let i = 0; i < board.n; i++) {
          const f = ch1[slot[i]] > 0.5;
          const g = smp.lab[i];
          if (g >= 0) { if (f) got[g]++; want[g] = t0[slot[i]]; } else if (f) wallFilled++;
        }
        const off = smp.regions.map((g, k) => ({ g, k, share: got[k] / g.size }))
          .filter(({ k, share }) => Math.abs(share - want[k]) > 1e-9)
          .map(({ g, k, share }) => `${g.rim ? 'rim' : 'enclosed'} ${g.size} want ${want[k]} got ${share.toFixed(2)}`);
        r.misses.push({ rule: smp.rule, kinds: smp.kinds, fillCells: smp.fillCells, wrong, tie: +smp.tie.toFixed(3), wallFilled,
          regions: smp.regions.length, off });
      }
    }
  }
  const totalSteps = samples.length * mults[mults.length - 1] * frame.R;
  for (const r of res) r.msPerStep = stepMs / totalSteps;
  return res;
}

const median = (xs: number[]): number => {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

/** The level's header line: the boards' make-up. */
export function describeSamples(level: string, board: Board, frame: AreaFrame, samples: Sample[]): string {
  return `=== ${level}: ${board.n} cells, frame S ${frame.S} (R ${frame.R}), ${samples.length} boards; ` +
    `walls p50 ${median(samples.map((s) => s.wallCells))} cells; filled (primary) p50 ${median(samples.map((s) => s.fillCells))}, ` +
    `max ${Math.max(...samples.map((s) => s.fillCells))}; kinds ${['loops', 'claims', 'mixed'].map((k) => `${k} ${samples.filter((s) => kindsOf(s.kinds) === k).length}`).join(', ')}; ` +
    `filled-cell buckets ${SIZE_BUCKETS.map((_, b) => `${bucketName(b)}: ${samples.filter((s) => bucketOf(s.fillCells) === b).length}`).join(', ')}; ` +
    `2+ rim regions ${samples.filter((s) => s.rimRegions >= 2).length}`;
}

/** A result as printed lines. */
export function reportLines(r: Result, verbose: boolean): string[] {
  const out = [
    `${r.weights.padEnd(12)} ${String(r.steps).padStart(4)} steps (${r.mult}R): exact ${fmt(r.all)}; ` +
      `non-trivial ${fmt(r.nontrivial)}; cell acc ${(1 - r.wrongCells / r.boardCells).toFixed(5)}; ${r.msPerStep.toFixed(2)} ms/step`,
    `    filled cells ${SIZE_BUCKETS.map((_, b) => `${bucketName(b)}: ${fmt(r.byBucket[b])}`).join('  ')}`,
    `    ${Object.entries(r.byKind).map(([k, t]) => `${k}: ${fmt(t)}`).join('  ')}`,
    `    2+ rim regions, 2nd/largest ${TIE_BINS.map((b, i) => `[${b},${i + 1 < TIE_BINS.length ? TIE_BINS[i + 1] : 1}]: ${fmt(r.byTie[i])}`).join('  ')}`,
  ];
  if (verbose) {
    for (const x of r.misses) {
      out.push(`    miss ${x.rule} ${x.kinds.join('+')}: ${x.wrong} cells wrong, ${x.fillCells} to fill, tie ${x.tie}, ${x.regions} regions; ` +
        `${x.off.join('; ')}${x.wallFilled ? `; ${x.wallFilled} walls filled` : ''}`);
    }
  }
  return out;
}

/** The seed of a level's board stream (so the CPU and GL runs, and every weights set, see the same boards). */
export const levelSeed = (seed: number, level: string, minFill: number): number =>
  seed * 1000 + Number(level.slice(1)) + (minFill > 0 ? 500 * minFill : 0);
