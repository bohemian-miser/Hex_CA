// ms per step of the strand NCA (src/strand-nca.ts) on the play page's boards, in Node:
//   npx tsx scripts/strand-bench.ts [weights.json ...]      (default web/strand-weights.json)
import { readFileSync } from 'node:fs';
import { Board, RuleTable, type Rule, type StrandData } from '../src/strand.js';
import { StrandNCA, chordsAt, loadStrandWeights } from '../src/strand-nca.js';

const data = JSON.parse(readFileSync('web/strand-data.json', 'utf8')) as StrandData;
const table = new RuleTable(data);
const files = process.argv.slice(2).length ? process.argv.slice(2) : ['web/strand-weights.json'];
const rule = table.parse('15·000000000') as Rule;

for (const file of files) {
  const w = loadStrandWeights(JSON.parse(readFileSync(file, 'utf8')));
  const macs = 7 * w.channels * w.hidden + (w.depth - 1) * w.hidden * w.hidden + w.channels * w.hidden;
  console.log(`${file}: ${w.kind} ${w.arch} ${w.inputs}, C ${w.channels} H ${w.hidden} depth ${w.depth} ` +
    `(${(macs / 1000).toFixed(0)}k multiply-adds per cell-step)`);
  for (const name of ['l2', 'l3', 'l4']) {
    const b = new Board(data.boards[name]);
    const m = new StrandNCA(w, b, table);
    let tap = null;
    for (let i = 0; i < b.n && !tap; i++) {
      const row = Math.floor(b.pos[i] / b.w);
      const col = b.pos[i] % b.w;
      const ch = chordsAt(table, rule, b, row, col);
      if (ch.length) tap = { rule, row, col, d0: ch[0][0], d1: ch[0][1] };
    }
    m.setTaps(tap ? [tap] : []);
    const n = name === 'l4' ? 3 : name === 'l3' ? 10 : 40;
    for (let t = 0; t < 2; t++) m.step();
    const t0 = performance.now();
    for (let t = 0; t < n; t++) m.step();
    const ms = (performance.now() - t0) / n;
    console.log(`  ${name}: ${b.n} cells, ${ms.toFixed(1)} ms/step (${((b.n * macs) / ms / 1e6).toFixed(2)} GMAC/s)`);
  }
}
