import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { allStrands, Board, RuleTable, splitHash, walk, type Rule, type StrandData } from '../src/strand.js';
import { CODE35, StrandNCA, chordsAt, code35, decodeF32, isStrandWeights, loadStrandWeights, type Tap } from '../src/strand-nca.js';
import { rng } from '../src/lines.js';

const readJson = (path: string): unknown => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const data = readJson('../web/strand-data.json') as StrandData;
const table = new RuleTable(data);
const boards = Object.fromEntries(Object.entries(data.boards).map(([k, b]) => [k, new Board(b)]));

interface RulesFixture {
  samples: { subset: number; index: number; digits: number[]; key: string; hash: number }[];
  held: [number, number, boolean][];
  board: string;
  renders: {
    rule: number[]; describe: string; key: string; code: number[]; bits: number[];
    walks: { tap: number[]; rows: number[]; cols: number[]; ins: number[]; outs: number[]; index: number[]; closed: boolean }[];
  }[];
}
interface ParityCase {
  name: string;
  weights: unknown;
  board: string;
  taps: { rule: number[]; tap: number[]; at?: number }[];
  steps: number;
  shape: [number, number, number];
  step1?: string;
  state: string;
  drawn: number;
}
const rulesFix = readJson('./fixtures/strand-rules.json') as RulesFixture;
const parity = readJson('./fixtures/strand-parity.json') as { cases: ParityCase[] };
const ruleOf = (r: number[]): Rule => ({ s: r[0], digits: r.slice(1) });

describe('strand rules (src/strand.ts against nca/strand/rules.py)', () => {
  it('keys, indices and split hashes match the exporter', () => {
    expect(rulesFix.samples.length).toBeGreaterThan(20);
    for (const sm of rulesFix.samples) {
      const rule = { s: sm.subset, digits: sm.digits };
      expect(table.key(rule)).toBe(sm.key);
      expect(table.index(rule)).toBe(sm.index);
      expect(table.digitsOf(sm.subset, sm.index)).toEqual(sm.digits);
      expect(splitHash(`strand-split-v2|${sm.key}`)).toBe(sm.hash);
    }
  });

  it('held out exactly the rules rules.py holds out', () => {
    let held = 0;
    for (const [s, index, want] of rulesFix.held) {
      expect(table.heldOut({ s, digits: table.digitsOf(s, index) }), `${s}:${index}`).toBe(want);
      held += +want;
    }
    expect(held).toBeGreaterThan(50);
  });

  it('renders, codes, describes and walks as rules.py and walker.py', () => {
    const board = boards[rulesFix.board];
    for (const r of rulesFix.renders) {
      const rule = ruleOf(r.rule);
      expect(table.describe(rule)).toBe(r.describe);
      expect(table.key(rule)).toBe(r.key);
      expect(Array.from(table.code(rule))).toEqual(r.code);
      expect(Array.from(table.render(rule, board))).toEqual(r.bits);
      const ex = table.exits(rule, board);
      for (const w of r.walks) {
        const st = walk(ex, board, w.tap[0], w.tap[1], w.tap[2], w.tap[3]);
        expect(st.closed).toBe(w.closed);
        expect(st.rows).toEqual(w.rows);
        expect(st.cols).toEqual(w.cols);
        expect(st.ins).toEqual(w.ins);
        expect(st.outs).toEqual(w.outs);
        expect(st.index).toEqual(w.index);
      }
    }
  });

  it('parses describeRule forms and checks them against the table', () => {
    const fass = table.parse('128·010100000');
    expect(fass).toEqual({ s: 1, digits: [0, 1, 0, 1, 0, 0, 0, 0, 0] });
    expect(table.parse('128 · 010100000')).toEqual(fass);
    expect(table.parse('821/0101')).toEqual(fass);
    expect(table.heldOut(fass as Rule)).toBe(true); // one of the 20 legacy held-out rules
    expect(table.parse('15·000000000')).toEqual({ s: 0, digits: [0, 0, 0, 0, 0, 0, 0, 0, 0] });
    expect(typeof table.parse('12·000000000')).toBe('string'); // not a kernel subset
    expect(typeof table.parse('15·000020000')).toBe('string'); // Pi has 2 matchings in 15
    expect(typeof table.parse('hello')).toBe('string');
  });

  it('samples train and held-out rules from their own side of the split', () => {
    const rand = rng(5);
    const seen = new Set<number>();
    for (let k = 0; k < 200; k++) {
      const h = table.sample(rand, 'heldout');
      const t = table.sample(rand, 'train');
      expect(table.heldOut(h)).toBe(true);
      expect(table.heldOut(t)).toBe(false);
      seen.add(h.s);
    }
    expect(seen.size).toBe(7);
  });

  it('allStrands covers every chord of a rule exactly once, agreeing with walk() from any of its chords', () => {
    for (const key of ['l2', 'l3']) {
      const board = boards[key];
      for (const r of ['15·000000000', '128·010100000']) {
        const rule = table.parse(r) as Rule;
        const bits = table.render(rule, board);
        const ex = table.exits(rule, board);
        let total = 0;
        for (let p = 0; p < bits.length; p++) for (let k = 0; k < 15; k++) if ((bits[p] >> k) & 1) total++;
        const strands = allStrands(ex, board);
        const seen = new Set<string>();
        let covered = 0;
        for (const st of strands) {
          covered += st.rows.length;
          expect(st.rows.length).toBeGreaterThan(0);
          const again = walk(ex, board, st.rows[0], st.cols[0], st.ins[0], st.outs[0]);
          expect(again.closed).toBe(st.closed);
          expect(again.rows).toEqual(st.rows);
          expect(again.cols).toEqual(st.cols);
          expect(again.ins).toEqual(st.ins);
          expect(again.outs).toEqual(st.outs);
          for (let k = 0; k < st.rows.length; k++) {
            const a = Math.min(st.ins[k], st.outs[k]);
            const b = Math.max(st.ins[k], st.outs[k]);
            const chordKey = `${st.rows[k]},${st.cols[k]},${a},${b}`;
            expect(seen.has(chordKey)).toBe(false); // every chord walked exactly once
            seen.add(chordKey);
          }
        }
        expect(covered).toBe(total);
      }
    }
  });

  it('the boards are the Delta patches of levels 2, 3 and 4', () => {
    expect(boards.l2.n).toBe(63);
    expect(boards.l3.n).toBe(496);
    expect(boards.l4.n).toBe(3905);
    for (const b of Object.values(boards)) {
      for (let i = 0; i < b.n; i++) {
        for (let d = 0; d < 6; d++) {
          const j = b.nbr[i * 6 + d];
          if (j >= 0) expect(b.nbr[j * 6 + ((d + 3) % 6)]).toBe(i);
        }
      }
    }
  });
});

/** max |TS - torch| over the case's channels and on-board cells, and the torch state off the board (must be 0). */
function compare(m: StrandNCA, b: Board, want: Float32Array, shape: [number, number, number]): number {
  const [Ck, H, W] = shape;
  let worst = 0;
  for (let c = 0; c < Ck; c++) {
    for (let p = 0; p < H * W; p++) {
      const i = b.cellOf[p];
      const v = want[c * H * W + p];
      const got = i < 0 ? 0 : m.state[i * m.C + c];
      worst = Math.max(worst, Math.abs(got - v));
    }
  }
  return worst;
}

describe('strand NCA (src/strand-nca.ts against PyTorch)', () => {
  for (const cs of parity.cases) {
    it(`${cs.name}: ${cs.steps} steps on ${cs.board}, ${cs.taps.length} tap(s), to 1e-4`, () => {
      const wj = typeof cs.weights === 'string' ? readJson(`../${cs.weights}`) : cs.weights;
      expect(isStrandWeights(wj)).toBe(true);
      const w = loadStrandWeights(wj);
      const b = boards[cs.board];
      const m = new StrandNCA(w, b, table);
      const taps: Tap[] = cs.taps.map((t) => ({ rule: ruleOf(t.rule), row: t.tap[0], col: t.tap[1], d0: t.tap[2], d1: t.tap[3] }));
      for (const t of taps) expect(chordsAt(table, t.rule, b, t.row, t.col).some(([a, c]) => (a === t.d0 && c === t.d1) || (a === t.d1 && c === t.d0))).toBe(true);
      // held weights hold every tap from step 0; event weights take each at its step (`at`)
      const fire = (step: number) => { cs.taps.forEach((t, k) => { if ((t.at ?? 0) === step) m.tap(taps[k]); }); };
      if (m.eventTaps) fire(0);
      else m.setTaps(taps);
      m.step();
      if (cs.step1) expect(compare(m, b, decodeF32(cs.step1), [w.channels, b.h, b.w])).toBeLessThan(1e-4);
      for (let t = 1; t < cs.steps; t++) {
        if (m.eventTaps) fire(t);
        m.step();
      }
      expect(compare(m, b, decodeF32(cs.state), cs.shape)).toBeLessThan(1e-4);
      let drawn = 0;
      for (let i = 0; i < b.n; i++) for (let d = 0; d < 6; d++) drawn += +m.edgeOn(i, d);
      expect(drawn).toBe(cs.drawn);
    });
  }

  it('reset() starts over, and the same taps give the same state', () => {
    const w = loadStrandWeights(readJson('../web/strand-weights.json'));
    const b = boards.l2;
    const m = new StrandNCA(w, b, table);
    const rule = table.parse('15·000000000') as Rule;
    const [d0, d1] = chordsAt(table, rule, b, Math.floor(b.pos[10] / b.w), b.pos[10] % b.w)[0] ?? [];
    expect(d0).toBeDefined();
    m.setTaps([{ rule, row: Math.floor(b.pos[10] / b.w), col: b.pos[10] % b.w, d0, d1 }]);
    for (let t = 0; t < 5; t++) m.step();
    const a = m.state.slice();
    m.reset();
    for (let t = 0; t < 5; t++) m.step();
    expect(Array.from(m.state)).toEqual(Array.from(a));
  });

  it('tap modes: older files hold their taps; event weights refuse setTaps, tap() is an event, reset() forgets', () => {
    const held = loadStrandWeights(readJson('../web/strand-weights.json'));
    expect(held.tap.mode).toBe('held');
    expect(held.speed).toBe(1);
    const b = boards.l2;
    const rule = table.parse('15·000000000') as Rule;
    const p = b.pos[10];
    const [d0, d1] = chordsAt(table, rule, b, Math.floor(p / b.w), p % b.w)[0];
    const t: Tap = { rule, row: Math.floor(p / b.w), col: p % b.w, d0, d1 };
    const a = new StrandNCA(held, b, table);
    const c = new StrandNCA(held, b, table);
    a.setTaps([t]);
    c.tap(t); // held weights: tap() = one more held tap
    for (let k = 0; k < 4; k++) { a.step(); c.step(); }
    expect(Array.from(c.state)).toEqual(Array.from(a.state));
    const imp = parity.cases.find((x) => x.name.startsWith('impulse e'))!;
    const w = loadStrandWeights(imp.weights);
    expect(w.tap).toEqual({ mode: 'impulse', steps: 1, codeChannels: null });
    const m = new StrandNCA(w, b, table);
    expect(m.eventTaps).toBe(true);
    expect(() => m.setTaps([t])).toThrow();
    m.tap(t);
    for (let k = 0; k < 3; k++) m.step();
    const once = m.state.slice();
    m.reset();
    expect(m.tapList.length).toBe(0);
    m.tap(t);
    for (let k = 0; k < 3; k++) m.step();
    expect(Array.from(m.state)).toEqual(Array.from(once)); // the same tap, the same state: nothing left over
    const fx = loadStrandWeights(parity.cases.find((x) => x.name.startsWith('fixed'))!.weights);
    const f = new StrandNCA(fx, b, table);
    f.tap(t);
    const i = b.cellOf[t.row * b.w + t.col];
    expect(f.state[i * f.C + 1 + d0]).toBe(1);
    expect(f.state[i * f.C + 1 + d1]).toBe(1);
    const code = code35(table, rule);
    expect(code.length).toBe(CODE35);
    fx.tap.codeChannels!.forEach((ch, k) => expect(f.state[i * f.C + ch]).toBe(code[k]));
    expect(Array.from(code.slice(0, 8)).map((v) => (v > 0 ? 1 : 0))).toEqual(Array.from(table.code(rule).slice(0, 8)));
  });

  it('a rule-everywhere model refuses a second tap', () => {
    const v1 = parity.cases.find((c) => c.name.startsWith('v1'))!;
    const w = loadStrandWeights(v1.weights);
    expect(w.ruleEverywhere).toBe(true);
    const m = new StrandNCA(w, boards.l2, table);
    const t: Tap = { rule: ruleOf(v1.taps[0].rule), row: v1.taps[0].tap[0], col: v1.taps[0].tap[1], d0: v1.taps[0].tap[2], d1: v1.taps[0].tap[3] };
    expect(() => m.setTaps([t, t])).toThrow();
  });
});
