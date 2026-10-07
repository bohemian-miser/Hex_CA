import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { PARITY_KNOBS, compare, replay, replayAll, type Fixture } from '../scripts/game-parity.js';
import { rng } from '../src/lines.js';
import { Board, PAIRS, RuleTable, walk, type StrandData } from '../src/strand.js';

// The hybrid line CA's parity (docs/spectacle-ca-hybrid.md §2.10, §5):
//   (a) one tap = the walker at double time: chord k from the tap (a loop: the shorter way round) is drawn at
//       t + 2k on the CA's phase (go steps are even, so a tap on an odd step grows its first chord one step
//       after it), and at rest the line is the whole strand, every tip gone;
//   (c) Spectacle's engine on collide.json (scripts/strand-export.ts --collide): LineCA agrees at rest (accepted
//       taps and every player's chords) on exactly the episodes nca/strand/ca_parity.py found agreeing, and
//       every other one has a class there that isn't "unexplained" (sim.py's waves or head timing, or the CA's
//       own timing). After a change to LineCA, rerun `python -m nca.strand.ca_parity --classes
//       tests/fixtures/ca-collide-classes.json` and look at what moved.
// (b), against sim.py's CA mode, is `python -m nca.strand.ca_parity` (Python).

const readJson = (path: string): unknown => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const data = readJson('../web/strand-data.json') as StrandData;
const table = new RuleTable(data);

/** 300 taps per level, in chunks of 50 (a long synchronous test starves vitest's worker RPC on the Pi). */
const CHUNKS = ['l2', 'l3', 'l4'].flatMap((name, l) => Array.from({ length: 6 }, (_, k) => [name, k, 21 + 10 * l + k] as const));

describe('(a) one tap = the walker at double time', () => {
  const seen = { taps: 0, loops: 0, odd: 0 };
  it.each(CHUNKS)('%s: taps %i (50 of random rules)', (name, _k, seed) => {
    const board = new Board(data.boards[name]);
    const rand = rng(seed);
    for (let n = 0; n < 50;) {
      const rule = table.sample(rand, rand() < 0.8 ? 'train' : 'heldout');
      const bits = table.render(rule, board);
      const cells: number[] = [];
      for (let p = 0; p < bits.length; p++) if (bits[p]) cells.push(p);
      if (!cells.length) continue;
      const p = cells[Math.floor(rand() * cells.length)];
      const pairs = PAIRS.filter((_, k) => (bits[p] >> k) & 1);
      let [d0, d1] = pairs[Math.floor(rand() * pairs.length)];
      if (rand() < 0.5) [d0, d1] = [d1, d0];
      const row = Math.floor(p / board.w);
      const col = p % board.w;
      const t0 = Math.floor(rand() * 4);
      const strand = walk(table.exits(rule, board), board, row, col, d0, d1);
      const len = strand.rows.length;
      const want = new Map<string, number>();
      let settle = t0;
      for (let k = 0; k < len; k++) {
        const i = strand.index[k];
        const dist = strand.closed ? Math.min(i, len - i) : Math.abs(i);
        const on = dist === 0 ? t0 : t0 + 2 * dist - (t0 % 2);
        settle = Math.max(settle, on);
        const a = Math.min(strand.ins[k], strand.outs[k]);
        const b = Math.max(strand.ins[k], strand.outs[k]);
        want.set(`0,${strand.rows[k]},${strand.cols[k]},${a},${b}`, on);
      }
      const ep = { board: name, players: [[rule.s, ...rule.digits]], taps: [{ t: t0, player: 0, row, col, d0, d1 }] };
      const got = replay(table, board, ep, PARITY_KNOBS, settle + 8);
      const label = `${table.describe(rule)} tap t${t0} (${row}, ${col}) ${d0}-${d1}, ${len} chords${strand.closed ? ', a loop' : ''}`;
      expect(got.taps[0].ok, label).toBe(true);
      const have = new Map(got.chords.map((c) => [c.slice(0, 5).join(','), c]));
      expect(have.size, label).toBe(want.size);
      for (const [k, on] of want) {
        const c = have.get(k);
        expect(c, `${label}: chord ${k}`).toBeDefined();
        expect([c![5], c![6]], `${label}: chord ${k} on, off`).toEqual([on, -1]);
      }
      expect(got.chords.length, `${label}: drawn once each`).toBe(want.size);
      expect(got.settle, label).toBe(settle);
      seen.taps++;
      seen.loops += strand.closed ? 1 : 0;
      seen.odd += t0 % 2;
      n++;
    }
  });

  it('the sample has loops and taps on both phases', () => {
    expect(seen.taps).toBe(900);
    expect(seen.loops).toBeGreaterThan(20);
    expect(seen.odd).toBeGreaterThan(300);
  });
});

describe("(c) Spectacle's engine at rest (collide.json)", () => {
  const fx = readJson('./fixtures/collide.json') as Fixture;
  const { classes } = readJson('./fixtures/ca-collide-classes.json') as { classes: string[] };

  it('agrees exactly where ca_parity found it agreeing; every other episode has an explained class', () => {
    expect(classes.length).toBe(fx.episodes.length);
    const ca = replayAll(table, fx);
    const c = compare(fx, ca);
    const differ = new Set(c.differ);
    const moved = fx.episodes.map((_, i) => i).filter((i) => differ.has(i) !== (classes[i] !== 'agree'));
    expect(moved.map((i) => `${i}: ${classes[i]}`)).toEqual([]);
    expect(classes.filter((x) => x.includes('unexplained'))).toEqual([]);
    expect(c.episodes).toBe(400);
  });
});
