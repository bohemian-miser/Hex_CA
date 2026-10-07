// B1 of docs/spectacle-ca-hybrid.md (§3.6): how well the trained flood fills strand-shaped walls — the fine-tune
// go/no-go. Per level, n boards; each is 1-3 strands of one random rule (one owner), every strand a closed loop or a
// rim-to-rim claim from allStrands, its tiles the walls. The flood runs from the fresh state in the area layer's
// own frame (src/game/area.ts, §3.1) and is read out at mult·R steps; a board is exact when its fill (channel 1 >
// 0.5) equals any of targets()'s acceptable answers on every board cell, walls included (nca/evaluate.py's rule).
//
//   npx tsx scripts/area-probe.ts [--levels l2,l3,l4] [--n 200] [--n-l4 24] [--mults 8,16] [--seed 1]
//                                 [--weights name=path ...] [--json out.json] [--dry] [--nontrivial] [--min-fill N]
//                                 [--verbose]
//
// Default weights: web/nca-weights.json (fb-r6816nt). Every weights set sees the same boards.

import { readFileSync, writeFileSync } from 'node:fs';
import { areaFrame } from '../src/game/area.js';
import { rng } from '../src/lines.js';
import { HexNCA, loadWeights, targets, type NCAWeights } from '../src/nca.js';
import { allStrands, Board, DCOL, DROW, RuleTable, type Rule, type Strand, type StrandData } from '../src/strand.js';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
function args(name: string): string[] {
  const out: string[] = [];
  process.argv.forEach((a, i) => { if (a === `--${name}` && i + 1 < process.argv.length) out.push(process.argv[i + 1]); });
  return out;
}

const levels = arg('levels', 'l2,l3,l4').split(',');
const nDefault = Number(arg('n', '200'));
const mults = arg('mults', '8,16').split(',').map(Number).sort((a, b) => a - b);
const seed = Number(arg('seed', '1'));
const jsonOut = arg('json', '');
const dry = process.argv.includes('--dry'); // sample the boards and print their make-up, run nothing
// Only boards whose primary target fills at least this many cells (--nontrivial: 1).
const minFill = process.argv.includes('--nontrivial') ? Math.max(1, Number(arg('min-fill', '1'))) : Number(arg('min-fill', '0'));
const verbose = process.argv.includes('--verbose'); // print every miss, region by region
const weightArgs = args('weights').length ? args('weights') : ['fb-r6816nt=web/nca-weights.json'];
const weightSets: { name: string; w: NCAWeights }[] = weightArgs.map((s) => {
  const [name, path] = s.includes('=') ? s.split('=') : [s, s];
  return { name, w: loadWeights(JSON.parse(readFileSync(path, 'utf8'))) };
});

const data = JSON.parse(readFileSync('web/strand-data.json', 'utf8')) as StrandData;
const table = new RuleTable(data);

type Kind = 'loop' | 'claim';
const SIZE_BUCKETS = [0, 1, 4, 16, 64] as const; // primary target's filled cells: 0, 1-3, 4-15, 16-63, 64+
const bucketOf = (n: number): number => { let b = 0; while (b + 1 < SIZE_BUCKETS.length && n >= SIZE_BUCKETS[b + 1]) b++; return b; };
const bucketName = (b: number): string => b + 1 < SIZE_BUCKETS.length
  ? (SIZE_BUCKETS[b + 1] - 1 === SIZE_BUCKETS[b] ? `${SIZE_BUCKETS[b]}` : `${SIZE_BUCKETS[b]}-${SIZE_BUCKETS[b + 1] - 1}`)
  : `${SIZE_BUCKETS[b]}+`;
const TIE_BINS = [0, 0.25, 0.5, 0.75]; // second-largest / largest rim region, boards with 2+ rim regions

/** A strand's ends: 'rim' when the line would carry on off the board, 'tail' when the next tile has no chord. */
function endsOf(s: Strand, board: Board): [boolean, boolean] {
  const n = s.rows.length;
  const off = (r: number, c: number, d: number) => !board.on(r + DROW[d], c + DCOL[d]);
  return [off(s.rows[0], s.cols[0], s.ins[0]), off(s.rows[n - 1], s.cols[n - 1], s.outs[n - 1])];
}

interface Sample {
  rule: string;
  kinds: Kind[];
  strandLen: number[];
  walls: Uint8Array; // S² slots
  wallCells: number;
  targets: Uint8Array[];
  fillCells: number; // primary target
  rimRegions: number;
  tie: number; // second-largest / largest rim region (0 with fewer than two)
  regions: { size: number; rim: boolean; lab: Int32Array }[];
}

/** Regions of the free board cells (6-connected), as targets() sees them: [cells, touches the rim] each. */
function regions(board: Board, wall: Uint8Array): { size: number; rim: boolean; lab: Int32Array }[] {
  const lab = new Int32Array(board.n).fill(-1);
  const out: { size: number; rim: boolean; lab: Int32Array }[] = [];
  const q = new Int32Array(board.n);
  for (let s = 0; s < board.n; s++) {
    if (wall[s] || lab[s] >= 0) continue;
    let h = 0, t = 0, size = 0, rim = false;
    lab[s] = out.length;
    q[t++] = s;
    while (h < t) {
      const i = q[h++];
      size++;
      for (let d = 0; d < 6; d++) {
        const j = board.nbr[i * 6 + d];
        if (j < 0) { rim = true; continue; }
        if (wall[j] || lab[j] >= 0) continue;
        lab[j] = out.length;
        q[t++] = j;
      }
    }
    out.push({ size, rim, lab });
  }
  return out;
}

function sampleBoard(board: Board, frame: ReturnType<typeof areaFrame>, rand: () => number,
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
    const regs = regions(board, wallCell);
    const rims = regs.filter((r) => r.rim).map((r) => r.size).sort((a, b) => b - a);
    return {
      rule: key, kinds: picked.map((p) => p.kind), strandLen: picked.map((p) => p.st.rows.length), walls, wallCells,
      targets: tg, fillCells, rimRegions: rims.length, tie: rims.length >= 2 ? rims[1] / rims[0] : 0, regions: regs,
    };
  }
}

interface Tally { exact: number; n: number }
const tally = (): Tally => ({ exact: 0, n: 0 });
const add = (t: Tally, ok: boolean) => { t.n++; if (ok) t.exact++; };
const fmt = (t: Tally): string => {
  if (!t.n) return '   —      ';
  const p = t.exact / t.n;
  const se = Math.sqrt((p * (1 - p)) / t.n);
  return `${p.toFixed(3)}±${(2 * se).toFixed(3)} (${t.n})`;
};

interface Result {
  level: string; weights: string; steps: number; mult: number; R: number;
  all: Tally; nontrivial: Tally; byBucket: Tally[]; byKind: Record<string, Tally>; byTie: Tally[];
  wrongCells: number; boardCells: number; msPerStep: number;
  misses: { rule: string; kinds: Kind[]; fillCells: number; wrong: number; tie: number; wallFilled: number; regions: number; off: string[] }[];
}

const results: Result[] = [];
for (const level of levels) {
  const board = new Board(data.boards[level]);
  const frame = areaFrame(board);
  const n = Number(arg(`n-${level}`, level === 'l4' ? '24' : String(nDefault)));
  const rand = rng(seed * 1000 + Number(level.slice(1)) + (minFill > 0 ? 500 * minFill : 0));
  const cache = new Map<string, { loops: Strand[]; claims: Strand[] }>();
  const samples: Sample[] = [];
  while (samples.length < n) {
    const smp = sampleBoard(board, frame, rand, cache);
    if (smp.fillCells >= minFill) samples.push(smp);
  }
  const kindsTxt = (k: Kind[]) => (k.every((x) => x === 'loop') ? 'loops' : k.every((x) => x === 'claim') ? 'claims' : 'mixed');
  console.log(`\n=== ${level}: ${board.n} cells, frame S ${frame.S} (R ${frame.R}), ${n} boards; ` +
    `walls p50 ${median(samples.map((s) => s.wallCells))} cells; filled (primary) p50 ${median(samples.map((s) => s.fillCells))}, ` +
    `max ${Math.max(...samples.map((s) => s.fillCells))}; kinds ${['loops', 'claims', 'mixed'].map((k) => `${k} ${samples.filter((s) => kindsTxt(s.kinds) === k).length}`).join(', ')}; ` +
    `filled-cell buckets ${SIZE_BUCKETS.map((_, b) => `${bucketName(b)}: ${samples.filter((s) => bucketOf(s.fillCells) === b).length}`).join(', ')}; ` +
    `2+ rim regions ${samples.filter((s) => s.rimRegions >= 2).length}`);
  if (dry) continue;
  for (const { name, w } of weightSets) {
    const nca = new HexNCA(w, frame.R, frame.mask);
    const res: Result[] = mults.map((mult) => ({
      level, weights: name, mult, steps: mult * frame.R, R: frame.R, all: tally(), nontrivial: tally(),
      byBucket: SIZE_BUCKETS.map(tally), byKind: { loops: tally(), claims: tally(), mixed: tally() }, byTie: TIE_BINS.map(tally),
      wrongCells: 0, boardCells: 0, msPerStep: 0, misses: [],
    }));
    let stepMs = 0;
    for (const smp of samples) {
      for (const s of nca.cells) nca.setWall(s, smp.walls[s] as 0 | 1);
      nca.reset();
      for (let m = 0; m < mults.length; m++) {
        const t0 = performance.now();
        nca.step(res[m].steps - nca.steps);
        stepMs += performance.now() - t0;
        const ch1 = nca.channel(1);
        let wrong = Infinity;
        for (const t of smp.targets) {
          let wr = 0;
          for (const s of nca.cells) if ((ch1[s] > 0.5 ? 1 : 0) !== t[s]) wr++;
          wrong = Math.min(wrong, wr);
        }
        const ok = wrong === 0;
        const r = res[m];
        add(r.all, ok);
        if (smp.fillCells > 0) add(r.nontrivial, ok);
        add(r.byBucket[bucketOf(smp.fillCells)], ok);
        add(r.byKind[kindsTxt(smp.kinds)], ok);
        if (smp.rimRegions >= 2) add(r.byTie[Math.min(TIE_BINS.length - 1, Math.floor(smp.tie * 4))], ok);
        r.wrongCells += wrong;
        r.boardCells += nca.cells.length;
        if (!ok) {
          // Region by region: size, rim or enclosed, what the primary target wants, the share the flood filled.
          const t0 = smp.targets[0];
          const lab = smp.regions.length ? smp.regions[0].lab : new Int32Array(0);
          const got = smp.regions.map(() => 0);
          let wallFilled = 0;
          for (let i = 0; i < board.n; i++) {
            const f = ch1[frame.slot[i]] > 0.5;
            if (lab[i] >= 0) { if (f) got[lab[i]]++; } else if (f) wallFilled++;
          }
          const regs = smp.regions.map((g, k) => ({ size: g.size, rim: g.rim, want: 0, got: +(got[k] / g.size).toFixed(2), k }));
          for (const g of regs) { let at = -1; for (let i = 0; i < board.n; i++) if (lab[i] === g.k) { at = i; break; } g.want = t0[frame.slot[at]]; }
          const off = regs.filter((g) => Math.abs(g.got - g.want) > 0.001).map((g) => `${g.rim ? 'rim' : 'enc'} ${g.size} want ${g.want} got ${g.got}`);
          r.misses.push({ rule: smp.rule, kinds: smp.kinds, fillCells: smp.fillCells, wrong, tie: +smp.tie.toFixed(3), wallFilled, regions: regs.length, off });
        }
      }
    }
    const totalSteps = samples.length * mults[mults.length - 1] * frame.R;
    for (const r of res) r.msPerStep = stepMs / totalSteps;
    for (const r of res) {
      results.push(r);
      console.log(`${name.padEnd(12)} ${String(r.steps).padStart(4)} steps (${r.mult}R): exact ${fmt(r.all)}; ` +
        `non-trivial ${fmt(r.nontrivial)}; cell acc ${(1 - r.wrongCells / r.boardCells).toFixed(5)}; ${r.msPerStep.toFixed(2)} ms/step`);
      console.log(`    filled cells ${SIZE_BUCKETS.map((_, b) => `${bucketName(b)}: ${fmt(r.byBucket[b])}`).join('  ')}`);
      console.log(`    ${Object.entries(r.byKind).map(([k, t]) => `${k}: ${fmt(t)}`).join('  ')}`);
      if (verbose) for (const x of r.misses) console.log(`    miss ${x.rule} ${x.kinds.join('+')}: ${x.wrong} cells wrong of ${x.fillCells} filled, tie ${x.tie}, ${x.regions} regions; ${x.off.join('; ')}${x.wallFilled ? `; ${x.wallFilled} walls filled` : ''}`);
      console.log(`    2+ rim regions, 2nd/largest ${TIE_BINS.map((b, i) => `[${b},${i + 1 < TIE_BINS.length ? TIE_BINS[i + 1] : 1}]: ${fmt(r.byTie[i])}`).join('  ')}`);
    }
  }
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ seed, mults, results }, null, 1));

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}
