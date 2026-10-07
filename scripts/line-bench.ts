// Line CA (src/game/line-ca.ts) speed per step on the strand boards: four players tapping at random (the fuzz's
// game), the active-set step against the full sweep, and the same with the probe off.
//
//   npx tsx scripts/line-bench.ts [steps=2000]

import { readFileSync } from 'node:fs';
import { LineCA, type LineKnobs } from '../src/game/line-ca.js';
import { rng } from '../src/lines.js';
import { Board, RuleTable, type StrandData } from '../src/strand.js';

const data = JSON.parse(readFileSync(new URL('../web/strand-data.json', import.meta.url), 'utf8')) as StrandData;
const table = new RuleTable(data);
const STEPS = Number(process.argv[2]) || 2000;

function game(board: Board, knobs: Partial<LineKnobs>, full: boolean, seed: number): { ms: number; lineTiles: number } {
  const N = board.h * board.w;
  const rand = rng(seed);
  const lc = new LineCA(board, table, { ...knobs, maxTips: 6 });
  lc.full = full;
  const rules: number[] = [];
  const exits: Int8Array[] = [];
  for (let o = 1; o <= 4; o++) {
    const rule = table.sample(rand, 'train');
    rules.push(lc.addRule(rule, o));
    exits.push(table.exits(rule, board));
  }
  let ms = 0;
  let tiles = 0;
  for (let s = 0; s < STEPS; s++) {
    for (let k = 0; k < 4; k++) {
      if (rand() > 0.5) continue;
      const pl = Math.floor(rand() * 4);
      const cell = Math.floor(rand() * board.n);
      const p = board.pos[cell];
      const ds: number[] = [];
      for (let d = 0; d < 6; d++) if (exits[pl][d * N + p] >= 0) ds.push(d);
      if (!ds.length) continue;
      const d0 = ds[Math.floor(rand() * ds.length)];
      lc.tap({ cell, d0, d1: exits[pl][d0 * N + p], rule: rules[pl] });
    }
    const t0 = performance.now();
    lc.step();
    ms += performance.now() - t0;
    if (s % 100 === 99) for (let o = 1; o <= 4; o++) tiles += lc.lineTiles(o);
  }
  return { ms: ms / STEPS, lineTiles: tiles / (STEPS / 100) };
}

console.log(`line CA, ${STEPS} steps, 4 players tapping at random (ms per step, mean)`);
console.log('level  cells  line tiles  active  full sweep  active, probe off');
for (const [name, board] of Object.entries(data.boards).map(([k, b]) => [k, new Board(b)] as const)) {
  game(board, {}, false, 1); // warm up
  const a = game(board, {}, false, 2);
  const f = game(board, {}, true, 2);
  const np = game(board, { probe: false }, false, 2);
  console.log(`${name.padEnd(6)} ${String(board.n).padStart(5)}  ${a.lineTiles.toFixed(0).padStart(10)}  ${a.ms.toFixed(3).padStart(6)}  ${f.ms.toFixed(3).padStart(10)}  ${np.ms.toFixed(3).padStart(17)}`);
}

// §7's probe risk: one tap of a subset-128 rule (the infinite-line rules: the longest strands there are), grown to
// rest with the probe on and off; ms per step while anything changes.
console.log('\none tap of the longest subset-128 strand, grown to rest (ms per step while changing)');
console.log('level  chords  steps  probe on  probe off');
const s128 = table.keys.indexOf('128');
for (const [name, board] of Object.entries(data.boards).map(([k, b]) => [k, new Board(b)] as const)) {
  const N = board.h * board.w;
  let best = { len: 0, rule: { s: s128, digits: [] as number[] }, cell: 0, d0: 0, d1: 0 };
  for (let i = 0; i < table.nOpt[s128].reduce((a, b) => a * b, 1); i++) {
    const rule = { s: s128, digits: table.digitsOf(s128, i) };
    const ex = table.exits(rule, board);
    const seen = new Uint8Array(N * 6);
    for (let p = 0; p < N; p++) {
      for (let d = 0; d < 6; d++) {
        const e = ex[d * N + p];
        if (e < 0 || seen[p * 6 + d]) continue;
        // a strand's length by walking it once
        const lc0 = { len: 0 };
        let r = Math.floor(p / board.w), c = p % board.w, din = d, dout = e;
        const start = p * 6 + d;
        for (;;) {
          const q = r * board.w + c;
          seen[q * 6 + din] = seen[q * 6 + dout] = 1;
          lc0.len++;
          const nr = r + [0, -1, -1, 0, 1, 1][dout], nc = c + [1, 1, 0, -1, -1, 0][dout];
          if (!board.on(nr, nc)) break;
          const ein = (dout + 3) % 6;
          const eout = ex[ein * N + nr * board.w + nc];
          if (eout < 0 || (nr * board.w + nc) * 6 + ein === start || seen[(nr * board.w + nc) * 6 + ein]) break;
          r = nr; c = nc; din = ein; dout = eout;
        }
        if (lc0.len > best.len) best = { len: lc0.len, rule, cell: board.cellOf[p], d0: d, d1: e };
      }
    }
  }
  const timed = (probe: boolean) => {
    const lc = new LineCA(board, table, { probe });
    const A = lc.addRule(best.rule, 1);
    lc.tap({ cell: best.cell, d0: best.d0, d1: best.d1, rule: A });
    let ms = 0, steps = 0, quiet = 0;
    while (quiet < 3) {
      const t0 = performance.now();
      lc.step();
      ms += performance.now() - t0;
      steps++;
      quiet = lc.ca.changed ? 0 : quiet + 1;
    }
    return { ms: ms / steps, steps, chords: lc.components().reduce((a, c) => a + c.chords, 0) };
  };
  const on = timed(true);
  const off = timed(false);
  console.log(`${name.padEnd(6)} ${String(on.chords).padStart(6)}  ${String(on.steps).padStart(5)}  ${on.ms.toFixed(3).padStart(8)}  ${off.ms.toFixed(3).padStart(9)}`);
}
