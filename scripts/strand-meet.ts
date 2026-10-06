// What a strand NCA does when two patterns meet on one board (the strand page's shared board: both taps are
// inputs of one state), with the page's runner (src/strand-nca.ts):
//   npx tsx scripts/strand-meet.ts WEIGHTS.json [board l2|l3|l4] [pool 24] [pairs 12] [seed 1] [split train|heldout]
// Phase 1: a pool of taps (strands of 8..50 chords) that the model draws EXACTLY alone in 110 steps.
// Phase 2: pairs from the pool whose true strands share tiles (each tap in at most two pairs): (a) both taps from
// step 0; (b) A alone until complete, then B added with the state kept -- a line already there, another pattern
// runs into it. Per pair: the share of each strand's chords drawn, strays, where A lost chords, and for each way
// out of B's tap that reaches A whether B stops within a chord of the first tile it shares with A.
import { readFileSync } from 'node:fs';
import { Board, RuleTable, walk, type Strand, type StrandData } from '../src/strand.js';
import { StrandNCA, chordsAt, loadStrandWeights, type Tap } from '../src/strand-nca.js';
import { rng } from '../src/lines.js';
const data = JSON.parse(readFileSync(new URL('../web/strand-data.json', import.meta.url), 'utf8')) as StrandData;
const table = new RuleTable(data);
const [wfile, boardName = 'l3', poolN = '24', pairN = '12', seedS = '1', splitArg = 'train'] = process.argv.slice(2);
const w = loadStrandWeights(JSON.parse(readFileSync(wfile, 'utf8')));
const b = new Board(data.boards[boardName]);
const rand = rng(Number(seedS));
const G = 110;
interface T { tap: Tap; st: Strand; cells: Set<number>; edges: Set<number>; name: string }
const cellOf = (st: Strand, k: number) => b.cellOf[st.rows[k] * b.w + st.cols[k]];
function mk(tap: Tap): T {
  const st = walk(table.exits(tap.rule, b), b, tap.row, tap.col, tap.d0, tap.d1);
  const cells = new Set<number>();
  const edges = new Set<number>();
  st.rows.forEach((_, k) => { const i = cellOf(st, k); cells.add(i); edges.add(i * 6 + st.ins[k]); edges.add(i * 6 + st.outs[k]); });
  return { tap, st, cells, edges, name: `${table.describe(tap.rule)}(${st.rows.length}${st.closed ? 'o' : '-'})` };
}
function drawn(m: StrandNCA): Set<number> {
  const s = new Set<number>();
  for (let i = 0; i < b.n; i++) for (let d = 0; d < 6; d++) if (m.edgeOn(i, d)) s.add(i * 6 + d);
  return s;
}
const chordOn = (t: T, P: Set<number>, k: number) => P.has(cellOf(t.st, k) * 6 + t.st.ins[k]) && P.has(cellOf(t.st, k) * 6 + t.st.outs[k]);
const frac = (t: T, P: Set<number>) => t.st.rows.filter((_, k) => chordOn(t, P, k)).length / t.st.rows.length;
const exactAlone = (t: T, P: Set<number>) => P.size === t.edges.size && [...t.edges].every((e) => P.has(e));
function run(taps: Tap[], steps: number, m?: StrandNCA): StrandNCA {
  const mm = m ?? new StrandNCA(w, b, table);
  mm.setTaps(taps);
  for (let s = 0; s < steps; s++) mm.step();
  return mm;
}
// phase 1
const pool: T[] = [];
let cand = 0;
const t0 = Date.now();
while (pool.length < Number(poolN) && cand < 2000) {
  const r = table.sample(rand, splitArg as 'train' | 'heldout');
  const p = b.pos[Math.floor(rand() * b.n)];
  const row = Math.floor(p / b.w);
  const col = p % b.w;
  const ch = chordsAt(table, r, b, row, col);
  if (!ch.length) continue;
  const [d0, d1] = ch[Math.floor(rand() * ch.length)];
  const t = mk({ rule: r, row, col, d0, d1 });
  const n = t.st.rows.length;
  if (n < 8 || n > 50) continue;
  cand++;
  if (exactAlone(t, drawn(run([t.tap], G)))) pool.push(t);
}
console.log(`${wfile} ${boardName}: pool ${pool.length} exact-alone taps of ${cand} tried (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
// phase 2
const pairs: [T, T][] = [];
for (let i = 0; i < pool.length && pairs.length < Number(pairN); i++) {
  for (let j = 0; j < pool.length && pairs.length < Number(pairN); j++) {
    if (i === j) continue;
    const A = pool[i];
    const B = pool[j];
    if (pairs.some(([x, y]) => (x === A && y === B) || (x === B && y === A))) continue;
    if (pairs.filter(([x, y]) => x === A || y === A).length >= 2 || pairs.filter(([x, y]) => x === B || y === B).length >= 2) continue;
    if (table.describe(A.tap.rule) === table.describe(B.tap.rule)) continue;
    const iA = b.cellOf[A.tap.row * b.w + A.tap.col];
    const iB = b.cellOf[B.tap.row * b.w + B.tap.col];
    if (iA === iB || A.cells.has(iB) || B.cells.has(iA)) continue;
    if (![...A.cells].some((c) => B.cells.has(c))) continue;
    pairs.push([A, B]);
  }
}
const S = { n: 0, togA: 0, togB: 0, togBoth: 0, seqA0: 0, seqA: 0, seqB: 0, seqBoth: 0, aLostShared: 0, aLostElse: 0, aChords: 0, bStop: 0, bOn: 0, bNoMeet: 0, togStray: 0, seqStray: 0, aGone: 0, aIntact: 0 };
for (const [A, B] of pairs) {
  const union = new Set([...A.edges, ...B.edges]);
  const shared = [...A.cells].filter((c) => B.cells.has(c));
  const m1 = run([A.tap, B.tap], G);
  const P1 = drawn(m1);
  const tA = frac(A, P1);
  const tB = frac(B, P1);
  const st1 = [...P1].filter((e) => !union.has(e)).length;
  const m2 = run([A.tap], G);
  const a0 = frac(A, drawn(m2));
  run([A.tap, B.tap], G, m2);
  const P2 = drawn(m2);
  const qA = frac(A, P2);
  const qB = frac(B, P2);
  const st2 = [...P2].filter((e) => !union.has(e)).length;
  let lostS = 0;
  let lostE = 0;
  A.st.rows.forEach((_, k) => { if (!chordOn(A, P2, k)) { if (B.cells.has(cellOf(A.st, k))) lostS++; else lostE++; } });
  // B, each way from its tap: stops within one chord of its first cell shared with A, or runs on past it
  const p0 = B.st.index.indexOf(0);
  const ways: string[] = [];
  for (const dir of [1, -1]) {
    let meet = -1;
    let gap = -1;
    for (let k = p0, s = 0; s < B.st.rows.length; k += dir, s++) {
      if (B.st.closed) k = (k + B.st.rows.length) % B.st.rows.length;
      else if (k < 0 || k >= B.st.rows.length) break;
      if (meet < 0 && A.cells.has(cellOf(B.st, k))) meet = s;
      if (gap < 0 && !chordOn(B, P2, k)) gap = s;
    }
    if (meet < 0) { S.bNoMeet++; continue; }
    if (gap >= 0 && gap <= meet + 1) { S.bStop++; ways.push(`stops (meets A ${meet} out, gap ${gap})`); }
    else { S.bOn++; ways.push(`runs on (meets A ${meet} out${gap >= 0 ? `, gap ${gap}` : ''})`); }
  }
  S.n++; S.togA += tA; S.togB += tB; S.togBoth += +(tA === 1 && tB === 1 && st1 === 0); S.seqA0 += a0; S.seqA += qA; S.seqB += qB;
  S.seqBoth += +(qA === 1 && qB === 1 && st2 === 0); S.aLostShared += lostS; S.aLostElse += lostE; S.aChords += A.st.rows.length;
  S.togStray += st1; S.seqStray += st2; S.aGone += +(qA < 0.5); S.aIntact += +(qA === 1);
  console.log(`${A.name} x ${B.name}, ${shared.length} shared cells | together: A ${tA.toFixed(2)} B ${tB.toFixed(2)} +${st1} stray | A first (${a0.toFixed(2)}), then B: A ${qA.toFixed(2)} (lost ${lostS} chords at shared cells, ${lostE} elsewhere) B ${qB.toFixed(2)} +${st2} stray | B: ${ways.join('; ')}`);
}
const n = S.n || 1;
const f = (x: number) => (x / n).toFixed(2);
console.log(`\nSUMMARY ${wfile} on ${boardName}, ${S.n} pairs, each tap exact alone, ${G} steps:`);
console.log(`  together from step 0: A ${f(S.togA)} B ${f(S.togB)} of their chords drawn, both exact ${S.togBoth}/${S.n}, ${f(S.togStray)} stray edges`);
console.log(`  A first, then B: A ${f(S.seqA0)} -> ${f(S.seqA)} (lost ${S.aLostShared} chords at shared cells + ${S.aLostElse} elsewhere of ${S.aChords}; A intact in ${S.aIntact}, mostly gone in ${S.aGone}), B ${f(S.seqB)}, both exact ${S.seqBoth}/${S.n}, ${f(S.seqStray)} stray`);
console.log(`  B's ways that reach A: stop there ${S.bStop}, run on ${S.bOn} (${S.bNoMeet} ways never reach A)`);
