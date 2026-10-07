// The line CA (src/game/line-ca.ts), docs/spectacle-ca-hybrid.md §5: every unit case on hand-built boards, step by
// step; single taps against the walker at double time on the real boards; and seeded fuzz games of 2-4 rules on
// l2/l3 with every §5 invariant asserted after every step, against the engine's reference sweep in lockstep.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CH, DEFAULT_LINE_KNOBS, GEOS, LineCA, OPP, type LineKnobs } from '../src/game/line-ca.js';
import { rng } from '../src/lines.js';
import { Board, RuleTable, walk, type BoardData, type Rule, type StrandData } from '../src/strand.js';
import { FUZZ_SEEDS, seeds } from './helpers.js';

const data = JSON.parse(readFileSync(new URL('../web/strand-data.json', import.meta.url), 'utf8')) as StrandData;
const table = new RuleTable(data);
const realBoards = { l2: new Board(data.boards.l2), l3: new Board(data.boards.l3) };

// ---------------------------------------------------------------- helpers

const DROW = [0, -1, -1, 0, 1, 1];
const DCOL = [1, 1, 0, -1, -1, 0];
const strip = new Board({ level: 0, root: 'geo', h: 1, w: GEOS, mirror: 1, tiles: GEOS, geo: Array.from({ length: GEOS }, (_, g) => g) });
const exitCache = new Map<string, Int8Array>();
/** partner under `rule` at geo g, edge d (the table line-ca builds). */
function partnerAt(rule: Rule, g: number, d: number): number {
  const key = table.key(rule);
  let ex = exitCache.get(key);
  if (!ex) exitCache.set(key, (ex = table.exits(rule, strip)));
  return ex[d * GEOS + g];
}

/** A requirement on one cell's geo: under `rule`, these chords exist, and these edges carry none. */
interface Need { rule: Rule; chords?: [number, number][]; none?: number[] }
function satisfies(g: number, needs: Need[]): boolean {
  return needs.every(({ rule, chords = [], none = [] }) =>
    chords.every(([a, b]) => partnerAt(rule, g, a) === b) && none.every((d) => partnerAt(rule, g, d) < 0));
}

/** The first geo meeting every need, or -1. */
function geoFor(needs: Need[]): number {
  for (let g = 0; g < GEOS; g++) if (satisfies(g, needs)) return g;
  return -1;
}

/** Random rules, the subset uniform over the seven (a tail or a 120° chord needs edges that carry no line). */
function* candidateRules(seed: number): Generator<Rule> {
  const rand = rng(seed);
  for (;;) {
    const s = Math.floor(rand() * table.nSub);
    yield { s, digits: table.nOpt[s].map((n) => Math.floor(rand() * n)) };
  }
}

/**
 * A hand-built board: cells (row, col) with requirements, everything else off the board. Finds rules (one per name)
 * for which every cell's needs are met by some geo, and the geos.
 */
function build(h: number, w: number, cells: { at: [number, number]; needs: { rule: string; chords?: [number, number][]; none?: number[] }[] }[],
  names: string[], seed = 1): { board: Board; rules: Record<string, Rule>; cell: (row: number, col: number) => number } {
  const gen = candidateRules(seed);
  for (let attempt = 0; attempt < 20000; attempt++) {
    const rules: Record<string, Rule> = {};
    for (const n of names) rules[n] = gen.next().value as Rule;
    if (new Set(names.map((n) => table.key(rules[n]))).size !== names.length) continue;
    const geo = new Array<number>(h * w).fill(-1);
    let ok = true;
    for (const { at: [r, c], needs } of cells) {
      const g = geoFor(needs.map((x) => ({ rule: rules[x.rule], chords: x.chords, none: x.none })));
      if (g < 0) { ok = false; break; }
      geo[r * w + c] = g;
    }
    if (!ok) continue;
    const bd: BoardData = { level: 0, root: 'test', h, w, mirror: 1, tiles: cells.length, geo };
    const board = new Board(bd);
    return { board, rules, cell: (row, col) => board.cellOf[row * w + col] };
  }
  throw new Error('no rules found for the board');
}

const bits = (...ds: number[]): number => ds.reduce((a, d) => a | (1 << d), 0);

/** Step `n` times, calling `each` after every step. */
function run(lc: LineCA, n: number, each?: () => void): void {
  for (let i = 0; i < n; i++) {
    lc.step();
    each?.();
  }
}

/** Step until nothing has changed for `period + 1` steps (a waiting tip changes nothing on off-phase steps). */
function settle(lc: LineCA, max = 100000): number {
  let quiet = 0;
  let k = 0;
  while (quiet <= lc.knobs.period && k < max) {
    lc.step();
    k++;
    quiet = lc.ca.changed === 0 ? quiet + 1 : 0;
  }
  return k;
}

/** Every channel equal, the sentinel included. */
function sameState(a: LineCA, b: LineCA): string | null {
  for (let c = 0; c < a.ca.ch.length; c++) {
    const x = a.ca.ch[c], y = b.ca.ch[c];
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return `channel ${a.ca.rule.channels[c].name} at ${i}: ${x[i]} vs ${y[i]}`;
  }
  return null;
}

// ---------------------------------------------------------------- invariants (§2.2, §5)

/** Every chord end connected to `start` (cell·6 + edge) along drawn chords of its rule. */
function endsOf(lc: LineCA, start: number): number[] {
  const { D, rule } = lc.ch;
  const seen = new Set<number>([start]);
  const stack = [start];
  while (stack.length) {
    const k = stack.pop()!;
    const c = Math.floor(k / 6), d = k % 6;
    const p = c * 6 + lc.pt(rule[c], c, d);
    if (!seen.has(p)) { seen.add(p); stack.push(p); }
    const m = lc.board.nbr[k];
    if (m >= 0 && rule[m] === rule[c] && (D[m] >> OPP[d]) & 1) {
      const q = m * 6 + OPP[d];
      if (!seen.has(q)) { seen.add(q); stack.push(q); }
    }
  }
  return [...seen];
}

/** One line (the test's own partition of the drawn chords, by search): its chord ends, smallest first. */
interface LineSet { keys: number[]; root: number; rule: number; loose: number[] }

function lineSets(lc: LineCA): LineSet[] {
  const { D, rule } = lc.ch;
  const seen = new Set<number>();
  const out: LineSet[] = [];
  for (let c = 0; c < lc.n; c++) {
    for (let d = 0; d < 6; d++) {
      const k0 = c * 6 + d;
      if (!((D[c] >> d) & 1) || seen.has(k0)) continue;
      const keys = endsOf(lc, k0).sort((a, b) => a - b);
      for (const k of keys) seen.add(k);
      const loose = keys.filter((k) => {
        const m = lc.board.nbr[k];
        return !(m >= 0 && rule[m] === rule[c] && (D[m] >> OPP[k % 6]) & 1);
      });
      out.push({ keys, root: keys[0], rule: rule[c], loose });
    }
  }
  return out;
}

const isOn = (a: Int32Array, k: number): boolean => ((a[Math.floor(k / 6)] >> (k % 6)) & 1) === 1;

/**
 * components() against the test's own partition: the same lines in the same order (by smallest chord end), each
 * with its owner, rule, cells, ends, closed and chord count.
 */
function checkComponents(lc: LineCA, sets: LineSet[], label: string): void {
  const comps = lc.components();
  if (comps.length !== sets.length) expect.fail(`${label}: ${comps.length} components, ${sets.length} lines`);
  comps.forEach((comp, i) => {
    const s = sets[i];
    const cells = [...new Set(s.keys.map((k) => Math.floor(k / 6)))].sort((a, b) => a - b);
    const want = { owner: lc.ownerOf[s.rule], rule: s.rule, cells, ends: s.loose.length, closed: s.loose.length === 0, chords: s.keys.length / 2 };
    const got = { owner: comp.owner, rule: comp.rule, cells: Array.from(comp.cells), ends: comp.ends, closed: comp.closed, chords: comp.chords };
    if (JSON.stringify(got) !== JSON.stringify(want)) expect.fail(`${label}: component ${i} ${JSON.stringify(got)} ≠ ${JSON.stringify(want)}`);
  });
}

interface Snapshot { D: Int32Array; rule: Int32Array; T: Int32Array; W: Int32Array; fuel: Int32Array; lineOf: Map<number, number>; sets: LineSet[] }

/** Checks every §5 invariant of one game, step by step: fed the state before each step and after it. */
class Invariants {
  private prevW: Int32Array;
  /** Per cell·6 + d: steps a tip has waited with an empty tile ahead. */
  private waiting: Int32Array;
  /** Doomed chord ends (lines that were hit, and whatever joined them) → their group. */
  private doomGroup = new Map<number, number>();
  /** at: the hit; chords: chord instances ever in the group (a chord the fleeing tip re-draws counts again). */
  private groups: { at: number; chords: number }[] = [];
  /** A closed line (smallest end : size) → the step it was first seen closed. */
  private closedSince = new Map<string, number>();
  constructor(readonly lc: LineCA, readonly label: string) {
    this.prevW = new Int32Array(lc.n + 1);
    this.waiting = new Int32Array(lc.n * 6);
  }

  before(): Snapshot {
    const { ch } = this.lc;
    const sets = lineSets(this.lc);
    const lineOf = new Map<number, number>();
    sets.forEach((s, i) => { for (const k of s.keys) lineOf.set(k, i); });
    return { D: ch.D.slice(), rule: ch.rule.slice(), T: ch.T.slice(), W: ch.W.slice(), fuel: ch.fuel.slice(), lineOf, sets };
  }

  after(pre: Snapshot, twin: LineCA | null): void {
    const { lc, label } = this;
    const { rule, owner, D, T, W, H, closed, fuel } = lc.ch;
    const { nbr } = lc.board;
    const n = lc.n;
    const gen = lc.ca.generation;
    const fuelMax = lc.knobs.fuel;
    const where = (c: number) => `${label} gen ${gen} cell ${c}`;
    // step == stepFull: the engine's reference sweep in lockstep
    if (twin) {
      const diff = sameState(lc, twin);
      if (diff) expect.fail(`${label} gen ${gen}: step ≠ stepFull: ${diff}`);
    }
    // §2.2, cell by cell
    for (let c = 0; c < n; c++) {
      const r = rule[c];
      if (D[c] !== 0 && r === 0) expect.fail(`${where(c)}: chords without a rule`);
      if (r !== 0 && D[c] === 0 && W[c] === 0) expect.fail(`${where(c)}: a rule without chords or waves`);
      if (r === 0 && W[c] !== 0 && H[c] !== 1) expect.fail(`${where(c)}: rule-0 waves off a crash`);
      if ((T[c] & ~D[c]) !== 0) expect.fail(`${where(c)}: T ⊄ D`);
      if ((closed[c] & ~D[c]) !== 0) expect.fail(`${where(c)}: closed ⊄ D`);
      if (owner[c] !== lc.ownerOf[r]) expect.fail(`${where(c)}: owner ${owner[c]} ≠ ownerOf[${r}]`);
      for (let d = 0; d < 6; d++) {
        if (!((D[c] >> d) & 1)) continue;
        const p = lc.pt(r, c, d);
        if (p < 0 || !((D[c] >> p) & 1)) expect.fail(`${where(c)}: D[${d}] without D[pt]`);
      }
      // no W lasts two steps
      if ((W[c] & this.prevW[c]) !== 0) expect.fail(`${where(c)}: W ${W[c]} two steps`);
      this.prevW[c] = W[c];
      // no tip waits more than `period` steps with an empty tile ahead
      for (let d = 0; d < 6; d++) {
        const k = c * 6 + d;
        const m = nbr[k];
        const empty = (T[c] >> d) & 1 && m >= 0 && rule[m] === 0 && W[m] === 0 && lc.pt(r, m, OPP[d]) >= 0
          && !(fuelMax > 0 && fuel[c] === 0);
        this.waiting[k] = empty ? this.waiting[k] + 1 : 0;
        if (this.waiting[k] > lc.knobs.period) expect.fail(`${where(c)}: tip ${d} waited ${this.waiting[k]} steps`);
      }
    }
    for (let o = 1; o <= 15; o++) {
      const k = lc.countTips(o);
      if (k > lc.knobs.maxTips) expect.fail(`${label} gen ${gen}: owner ${o} has ${k} tips`);
    }
    const sets = lineSets(lc);
    checkComponents(lc, sets, `${label} gen ${gen}`);

    // 0. a tap that joined a dying line (between steps) dies with it: the §2.10 race between a wave and a join
    for (const s of pre.sets) if (s.keys.some((k) => this.doomGroup.has(k))) this.addDoom(s.keys, gen);
    // 1. doom the lines hit this step, from the state before it: a victim tile's every line, a hitter's line, a
    //    crash's aimers
    const doomLine = (k: number) => {
      const i = pre.lineOf.get(k);
      if (i !== undefined) this.addDoom(pre.sets[i].keys, gen);
    };
    for (let c = 0; c < n; c++) {
      if (!H[c]) continue;
      const r = pre.rule[c];
      for (let d = 0; d < 6; d++) {
        const m = nbr[c * 6 + d];
        if (m < 0) continue;
        const aimedHere = (pre.T[m] >> OPP[d]) & 1 && lc.pt(pre.rule[m], c, d) >= 0 && !(fuelMax > 0 && pre.fuel[m] === 0);
        if (aimedHere && pre.rule[m] !== r) {
          if (r !== 0) for (let e = 0; e < 6; e++) if ((pre.D[c] >> e) & 1) doomLine(c * 6 + e);
          doomLine(m * 6 + OPP[d]);
        }
        if ((pre.T[c] >> d) & 1 && pre.rule[m] !== 0 && pre.rule[m] !== r && lc.pt(r, m, OPP[d]) >= 0) doomLine(c * 6 + d);
      }
    }
    // 2. collateral 0: every chord that went was on a doomed line
    for (let c = 0; c < n; c++) {
      const lost = pre.D[c] & ~(rule[c] === pre.rule[c] ? D[c] : 0);
      for (let d = 0; d < 6; d++) {
        if ((lost >> d) & 1 && !this.doomGroup.has(c * 6 + d)) expect.fail(`${where(c)}: chord end ${d} lost without a hit (collateral)`);
      }
    }
    // 3. a fresh chord with an end against a doomed chord joined the dying line as it was drawn: drawn by a doomed
    //    tip (whose own chord may have gone in the same step), or drawn into a dying line's loose end just as the
    //    wave got there (the §2.10 race) — but not one drawn right behind a wave that just left through that end
    //    (pre-step W there): the wave is moving away from it
    const postLine = new Map<number, number>();
    sets.forEach((s, i) => { for (const k of s.keys) postLine.set(k, i); });
    for (let c = 0; c < n; c++) {
      const fresh = D[c] & ~(rule[c] === pre.rule[c] ? pre.D[c] : 0);
      for (let d = 0; d < 6; d++) {
        if (!((fresh >> d) & 1)) continue;
        const m = nbr[c * 6 + d];
        if (m >= 0 && pre.rule[m] === rule[c] && this.doomGroup.has(m * 6 + OPP[d]) && !((pre.W[c] >> d) & 1)) {
          this.addDoom(sets[postLine.get(c * 6 + d)!].keys, gen);
        }
      }
    }
    // 4. forget gone chords; 5. doom spreads over joins (a line touching a doomed one is one line with it)
    for (const k of [...this.doomGroup.keys()]) if (!isOn(D, k)) this.doomGroup.delete(k);
    for (const s of sets) if (s.keys.some((k) => this.doomGroup.has(k))) this.addDoom(s.keys, gen);
    // 6. every doomed chord goes within L + 2 steps of its group's hit (L: chords ever in the group, re-drawn ones
    //    again: on a loop strand the far tip can run round the gap and re-draw behind the wave until it is caught)
    for (const [k, gi] of this.doomGroup) {
      const g = this.groups[gi];
      const L = g.chords;
      if (gen - g.at > L + 2) expect.fail(`${label} gen ${gen}: chord end ${k} still there ${gen - g.at} steps after its line was hit (L ${L})`);
    }

    // closed ⇒ a closed or dying line; a closed line is flagged everywhere within 1.5 L + 2 of closing;
    // every loose end of a line is a tip, a tail, the next chord of a wave, or spent fuel
    const seen = new Set<string>();
    for (const s of sets) {
      const flagged = s.keys.filter((k) => isOn(closed, k)).length;
      const dying = s.keys.some((k) => this.doomGroup.has(k));
      const isClosed = s.loose.length === 0;
      if (!lc.knobs.probe && flagged > 0) expect.fail(`${label} gen ${gen}: closed without the probe`);
      if (flagged > 0 && !isClosed && !dying) expect.fail(`${label} gen ${gen}: closed flags on an open line at ${s.root}`);
      if (isClosed && lc.knobs.probe) {
        const id = `${s.root}:${s.keys.length}`;
        seen.add(id);
        const since = this.closedSince.get(id) ?? gen;
        this.closedSince.set(id, since);
        const L = s.keys.length / 2;
        if (gen - since > 1.5 * L + 2 && flagged !== s.keys.length && !dying) {
          expect.fail(`${label} gen ${gen}: a loop of ${L} closed at ${since} is not flagged everywhere`);
        }
      }
      // a one-way tap's back end is a loose end of its own; under fuel a line stops anywhere (and a retap refuels
      // the whole tile, so where fuel ran out cannot be read back)
      if (lc.knobs.oneWay || fuelMax > 0 || dying) continue;
      for (const k of s.loose) {
        const c = Math.floor(k / 6), d = k % 6;
        const m = nbr[k];
        const tail = m < 0 || lc.pt(rule[c], m, OPP[d]) < 0;
        const wave = m >= 0 && (W[m] >> OPP[d]) & 1;
        if (!tail && !((T[c] >> d) & 1) && !wave) expect.fail(`${where(c)}: loose end ${d} that is no tip, tail or wave`);
      }
    }
    for (const id of [...this.closedSince.keys()]) if (!seen.has(id)) this.closedSince.delete(id);
  }

  private addDoom(keys: number[], gen: number): void {
    const gs = new Set<number>();
    for (const k of keys) {
      const g = this.doomGroup.get(k);
      if (g !== undefined) gs.add(g);
    }
    let target: number;
    if (gs.size === 0) {
      target = this.groups.length;
      this.groups.push({ at: gen, chords: 0 });
    } else {
      const list = [...gs].sort((a, b) => this.groups[a].at - this.groups[b].at);
      target = list[0];
      for (const g of list.slice(1)) this.groups[target].chords += this.groups[g].chords;
      for (const [k, g] of this.doomGroup) if (gs.has(g)) this.doomGroup.set(k, target);
    }
    let fresh = 0;
    for (const k of keys) {
      if (!this.doomGroup.has(k)) fresh++;
      this.doomGroup.set(k, target);
    }
    this.groups[target].chords += fresh / 2;
  }
}

/**
 * Lines are stretches of strands: every line is one contiguous run of the strand `walk` finds through any of its
 * chords (cyclically on a closed strand), and closed exactly when it is the whole closed strand.
 */
function checkStretches(lc: LineCA, label: string): void {
  const { board } = lc;
  const N = board.h * board.w;
  for (const s of lineSets(lc)) {
    const c0 = Math.floor(s.root / 6), d0 = s.root % 6;
    const ex = table.exits(lc.rules[s.rule]!, board);
    const p = board.pos[c0];
    const st = walk(ex, board, Math.floor(p / board.w), p % board.w, d0, ex[d0 * N + p]);
    const L = st.rows.length;
    const idx = new Map<number, number>();
    st.rows.forEach((r, k) => {
      const c = board.cellOf[r * board.w + st.cols[k]];
      idx.set(c * 6 + st.ins[k], k);
      idx.set(c * 6 + st.outs[k], k);
    });
    const on = new Uint8Array(L);
    for (const k of s.keys) {
      const i = idx.get(k);
      if (i === undefined) expect.fail(`${label}: chord end ${k} is not on the strand of its line`);
      on[i] = 1;
    }
    let count = 0, runs = 0;
    for (let k = 0; k < L; k++) {
      count += on[k];
      const prev = k > 0 ? on[k - 1] : st.closed ? on[L - 1] : 0;
      if (on[k] && !prev) runs++;
    }
    if (count === L && st.closed) runs = 1;
    if (count * 2 !== s.keys.length) expect.fail(`${label}: ${s.keys.length} chord ends for ${count} chords`);
    if (runs !== 1) expect.fail(`${label}: a line in ${runs} pieces of its strand`);
    if ((s.loose.length === 0) !== (st.closed && count === L)) expect.fail(`${label}: closed ≠ the whole closed strand`);
  }
}

// ---------------------------------------------------------------- unit cases (§5)

/** A straight row of `w` cells, each with the straight chord (3, 0) under every named rule. */
function row(w: number, names = ['A'], seed = 1, extra: Record<number, { rule: string; chords?: [number, number][]; none?: number[] }[]> = {}) {
  const cells = Array.from({ length: w }, (_, c) => ({
    at: [0, c] as [number, number],
    needs: extra[c] ?? names.map((rule) => ({ rule, chords: [[3, 0]] as [number, number][] })),
  }));
  return build(1, w, cells, names, seed);
}

describe('line CA: growth', () => {
  it('a tap grows both ways, chord k at tap + period·k, and stops at the rim', () => {
    for (const period of [2, 3]) {
      const { board, rules } = row(9);
      const lc = new LineCA(board, table, { period });
      const A = lc.addRule(rules.A, 1);
      expect(lc.tap({ cell: 4, d0: 3, d1: 0, rule: A })).toBeNull();
      expect(lc.ch.D[4]).toBe(bits(0, 3));
      expect(lc.ch.T[4]).toBe(bits(0, 3));
      expect(lc.countTips(1)).toBe(2);
      const drawnAt = new Array<number>(9).fill(-1);
      drawnAt[4] = 0;
      for (let g = 1; g <= 5 * period + 3; g++) {
        lc.step();
        for (let c = 0; c < 9; c++) if (lc.ch.D[c] && drawnAt[c] < 0) drawnAt[c] = g;
      }
      expect(drawnAt).toEqual([4, 3, 2, 1, 0, 1, 2, 3, 4].map((k) => k * period));
      expect(Array.from(lc.ch.D.subarray(0, 9))).toEqual(new Array(9).fill(bits(0, 3)));
      expect(Array.from(lc.ch.T.subarray(0, 9))).toEqual(new Array(9).fill(0));
      expect(lc.countTips(1)).toBe(0);
      expect(lc.lineTiles(1)).toBe(9);
      const comps = lc.components();
      expect(comps.length).toBe(1);
      expect(comps[0]).toMatchObject({ owner: 1, rule: A, ends: 2, closed: false, chords: 9 });
    }
  });

  it('a tip waits for its tile: one tip bit per line end, retiring once the tile ahead has drawn', () => {
    const { board, rules } = row(5);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    lc.step(); // gen 1: off phase; the rim tip (edge 3) retires
    expect(lc.ch.T[0]).toBe(bits(0));
    expect(lc.ch.D[1]).toBe(0);
    lc.step(); // gen 2: cell 1 draws, its tip out of 0
    expect(lc.ch.D[1]).toBe(bits(0, 3));
    expect(lc.ch.T[1]).toBe(bits(0));
    expect(lc.ch.T[0]).toBe(bits(0)); // still there: it sees the drawn tile next step
    expect(lc.countTips(1)).toBe(1); // ... but no longer counts
    lc.step();
    expect(lc.ch.T[0]).toBe(0);
  });

  it('a tail at a tile without the chord, and at the rim', () => {
    const { board, rules } = row(5, ['A'], 1, { 4: [{ rule: 'A', none: [3] }] });
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 1, d0: 3, d1: 0, rule: A });
    settle(lc);
    expect(Array.from(lc.ch.D.subarray(0, 5))).toEqual([bits(0, 3), bits(0, 3), bits(0, 3), bits(0, 3), 0]);
    expect(lc.ch.rule[4]).toBe(0);
    expect(lc.countTips(1)).toBe(0);
    expect(lc.components()[0]).toMatchObject({ ends: 2, closed: false, chords: 4 });
  });

  it('oneWay: a tap grows out of d1 only; retap grows the other end', () => {
    const { board, rules } = row(7);
    const lc = new LineCA(board, table, { oneWay: true });
    const A = lc.addRule(rules.A, 1);
    expect(lc.tap({ cell: 3, d0: 3, d1: 0, rule: A })).toBeNull();
    expect(lc.ch.T[3]).toBe(bits(0));
    expect(lc.countTips(1)).toBe(1);
    settle(lc);
    expect(Array.from(lc.ch.D.subarray(0, 7)).map((x) => +!!x)).toEqual([0, 0, 0, 1, 1, 1, 1]);
    expect(lc.retap(3, 3, 0, A)).toBe(true); // its d0 end is loose: Spectacle's turnRound
    expect(lc.ch.T[3]).toBe(bits(3));
    settle(lc);
    expect(Array.from(lc.ch.D.subarray(0, 7)).map((x) => +!!x)).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(lc.retap(3, 3, 0, A)).toBe(false); // nowhere left to go
  });

  it('two-way and unbounded, retap is a no-op (every stuck end is a tail)', () => {
    const { board, rules } = row(5);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 2, d0: 3, d1: 0, rule: A });
    settle(lc);
    for (let c = 0; c < 5; c++) expect(lc.retap(c, 3, 0, A)).toBe(false);
  });

  it('fuel: a tap lays fuelMax chords each way, and retap on the stuck end lays fuelMax more', () => {
    const { board, rules } = row(13);
    const lc = new LineCA(board, table, { fuel: 2 });
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 6, d0: 3, d1: 0, rule: A });
    expect(lc.ch.fuel[6]).toBe(2);
    settle(lc);
    const drawn = () => Array.from(lc.ch.D.subarray(0, 13)).map((x) => +!!x).join('');
    expect(drawn()).toBe('0000111110000');
    expect(lc.countTips(1)).toBe(0);
    expect(Array.from(lc.ch.T.subarray(0, 13)).every((t) => t === 0)).toBe(true);
    expect(lc.retap(8, 3, 0, A)).toBe(true); // the east end: refuelled
    expect(lc.ch.T[8]).toBe(bits(0));
    expect(lc.ch.fuel[8]).toBe(2);
    settle(lc);
    expect(drawn()).toBe('0000111111100');
    expect(lc.retap(4, 3, 0, A)).toBe(true);
    settle(lc);
    expect(drawn()).toBe('0011111111100');
  });

  it('per-owner go pulses: a tile draws only on its owner\'s go step', () => {
    const { board, rules } = row(11, ['A', 'B']);
    const lc = new LineCA(board, table, { maxTips: 4 });
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    // far apart, so they never meet in this test
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    lc.tap({ cell: 10, d0: 0, d1: 3, rule: B });
    const goA = Uint8Array.from([0, 1, 0]);
    const goB = Uint8Array.from([0, 0, 1]);
    lc.step(goA); // A draws cell 1; B waits
    expect(lc.ch.D[1]).toBe(bits(0, 3));
    expect(lc.ch.D[9]).toBe(0);
    lc.step(goA); // nobody goes two steps running: A waits
    expect(lc.ch.D[2]).toBe(0);
    lc.step(goB);
    expect(lc.ch.D[2]).toBe(0);
    expect(lc.ch.D[9]).toBe(bits(0, 3));
    lc.step(goA);
    expect(lc.ch.D[2]).toBe(bits(0, 3));
    expect(lc.ch.D[8]).toBe(0);
    lc.step(new Uint8Array(3));
    expect(lc.ch.D[3]).toBe(0);
    expect(lc.ch.D[8]).toBe(0);
  });

  it('period must be at least 2: at a chord a step a fleeing tip outruns its wave', () => {
    const { board } = row(3);
    expect(() => new LineCA(board, table, { period: 1 })).toThrow(/period/);
  });
});

describe('line CA: joins, overlap and loops', () => {
  // The draw-beside board: X = (1,1) holds two chords of A, (3,0) and (1,2); line 1 runs along row 1, line 2 from
  // (0,1) through X's edges 2 and 1 to (0,2).
  const beside = () => build(2, 3, [
    { at: [0, 1], needs: [{ rule: 'A', chords: [[5, 3]] }] },
    { at: [0, 2], needs: [{ rule: 'A', chords: [[4, 0]] }] },
    { at: [1, 0], needs: [{ rule: 'A', chords: [[0, 3]] }] },
    { at: [1, 1], needs: [{ rule: 'A', chords: [[3, 0], [1, 2]] }] },
    { at: [1, 2], needs: [{ rule: 'A', chords: [[3, 0]] }] },
  ], ['A']);

  it('draws beside its own chord on a tile, and the two lines stay two', () => {
    const { board, rules, cell } = beside();
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    expect(lc.tap({ cell: cell(1, 0), d0: 3, d1: 0, rule: A })).toBeNull();
    settle(lc);
    const X = cell(1, 1);
    expect(lc.ch.D[X]).toBe(bits(0, 3));
    expect(lc.tap({ cell: cell(0, 1), d0: 3, d1: 5, rule: A })).toBeNull();
    settle(lc);
    expect(lc.ch.D[X]).toBe(bits(0, 1, 2, 3));
    expect(lc.ch.D[cell(0, 2)]).toBe(bits(4, 0));
    const comps = lc.components();
    expect(comps.length).toBe(2);
    expect(comps.every((c) => c.ends === 2 && !c.closed && c.chords === 3)).toBe(true);
    // the tile's chords are both taken: a tap there is refused as drawn
    expect(lc.tap({ cell: X, d0: 1, d1: 2, rule: A })).toBe('drawn');
  });

  it('a tip reaching its own drawn chord retires: a join at a loose end, nothing announced', () => {
    const { board, rules } = row(9);
    const lc = new LineCA(board, table, { maxTips: 4 });
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 1, d0: 3, d1: 0, rule: A });
    lc.tap({ cell: 4, d0: 3, d1: 0, rule: A });
    lc.step();
    lc.step(); // cell 3 drawn by 4's west tip; cell 2 by 1's east tip
    expect(lc.ch.D[2]).toBe(bits(0, 3));
    expect(lc.ch.D[3]).toBe(bits(0, 3));
    lc.step(); // the tips at 2 (east) and 3 (west) face each other's drawn chords: both retire
    expect(lc.ch.T[2]).toBe(0);
    expect(lc.ch.T[3]).toBe(0);
    settle(lc);
    const comps = lc.components();
    expect(comps.length).toBe(1);
    expect(comps[0]).toMatchObject({ chords: 9, ends: 2, closed: false });
    expect(Array.from(lc.ch.closed.subarray(0, 9)).every((x) => x === 0)).toBe(true);
  });

  // The smallest loop: three tiles round a vertex, A = (0,0) (0,5), B = (0,1) (3,4), C = (1,0) (2,1).
  const triangle = () => build(2, 2, [
    { at: [0, 0], needs: [{ rule: 'A', chords: [[0, 5]] }] },
    { at: [0, 1], needs: [{ rule: 'A', chords: [[3, 4]] }] },
    { at: [1, 0], needs: [{ rule: 'A', chords: [[2, 1]] }] },
  ], ['A']);

  it('a loop closing on drawn chords: the probe comes back and closed floods round within 1.5 L', () => {
    const { board, rules, cell } = triangle();
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: cell(0, 0), d0: 0, d1: 5, rule: A });
    let closedAt = -1, flaggedAt = -1;
    for (let g = 1; g <= 20; g++) {
      lc.step();
      const comps = lc.components();
      if (closedAt < 0 && comps.length === 1 && comps[0].closed) closedAt = g;
      if (flaggedAt < 0 && [0, 1, 2].every((c) => lc.ch.closed[c] === lc.ch.D[c] && lc.ch.D[c] !== 0)) flaggedAt = g;
    }
    expect(closedAt).toBe(2); // both tips draw at gen 2: B and C hold each other's continuation
    expect(flaggedAt).toBeGreaterThan(closedAt);
    expect(flaggedAt - closedAt).toBeLessThanOrEqual(1.5 * 3 + 2);
    expect(lc.countTips(1)).toBe(0);
    for (let c = 0; c < 3; c++) for (let d = 0; d < 6; d++) expect(lc.ca.ch[CH.probe + d][c]).toBe(0); // probes gone
    expect(lc.components()[0]).toMatchObject({ closed: true, ends: 0, chords: 3 });
  });

  // A ring of six round a centre: the tips meet on the far tile's one chord at once (a loop closing on a fresh tile).
  const ring = (names = ['A'], seed = 1) => {
    const at: [number, number][] = [[1, 2], [0, 2], [0, 1], [1, 0], [2, 0], [2, 1]]; // direction k from the centre (1,1)
    return build(3, 3, at.map((p, k) => ({
      at: p, needs: names.map((rule) => ({ rule, chords: [[(k + 2) % 6, (k + 4) % 6]] as [number, number][] })),
    })), names, seed);
  };

  it('a loop closing on a fresh tile: both tips draw its chord, the probe comes back, closed floods', () => {
    const { board, rules, cell } = ring();
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: cell(1, 2), d0: 2, d1: 4, rule: A });
    const far = cell(1, 0);
    let fullAt = -1, flaggedAt = -1;
    for (let g = 1; g <= 30; g++) {
      lc.step();
      if (fullAt < 0 && lc.ch.D[far]) {
        fullAt = g;
        expect(lc.ch.T[far]).toBe(0); // no tip where two came in
      }
      const all = [0, 1, 2, 3, 4, 5].map((k) => [[1, 2], [0, 2], [0, 1], [1, 0], [2, 0], [2, 1]][k]).map(([r, c]) => cell(r, c));
      if (flaggedAt < 0 && all.every((c) => lc.ch.D[c] !== 0 && lc.ch.closed[c] === lc.ch.D[c])) flaggedAt = g;
    }
    expect(fullAt).toBe(6);
    expect(flaggedAt - fullAt).toBeLessThanOrEqual(1.5 * 6);
    expect(lc.components()).toHaveLength(1);
    expect(lc.components()[0]).toMatchObject({ closed: true, ends: 0, chords: 6 });
  });

  it('a long loop on the real board: closed everywhere within 1.5 L of closing', () => {
    // the longest closed strand of a few rules at l3
    const board = realBoards.l3;
    const N = board.h * board.w;
    const rand = rng(7);
    let best: { rule: Rule; L: number; row: number; col: number; d0: number; d1: number } | null = null;
    for (let i = 0; i < 40; i++) {
      const rule = table.sample(rand, 'train');
      const ex = table.exits(rule, board);
      for (let p = 0; p < N; p++) {
        for (let d = 0; d < 6; d++) {
          const e = ex[d * N + p];
          if (e < d) continue;
          const s = walk(ex, board, Math.floor(p / board.w), p % board.w, d, e);
          if (s.closed && (!best || s.rows.length > best.L)) best = { rule, L: s.rows.length, row: Math.floor(p / board.w), col: p % board.w, d0: d, d1: e };
        }
      }
    }
    expect(best!.L).toBeGreaterThan(30);
    const lc = new LineCA(board, table);
    const A = lc.addRule(best!.rule, 1);
    expect(lc.tap({ cell: board.cellOf[best!.row * board.w + best!.col], d0: best!.d0, d1: best!.d1, rule: A })).toBeNull();
    let closedAt = -1;
    let g = 0;
    while (g < 4 * best!.L) {
      lc.step();
      g++;
      if (closedAt < 0 && lc.components()[0].closed) closedAt = g;
      if (closedAt >= 0) {
        let all = true;
        for (let c = 0; c < lc.n; c++) if (lc.ch.closed[c] !== lc.ch.D[c]) all = false;
        if (all) break;
      }
    }
    expect(closedAt).toBe(2 * Math.ceil((best!.L - 1) / 2));
    expect(g - closedAt).toBeLessThanOrEqual(1.5 * best!.L);
  });
});

describe('line CA: hits and waves', () => {
  // A cross: rule B along row 3 (straight, (3,0)), rule A down column 4 (straight, (2,5)); X = (3,4) is on both.
  const cross = (w = 9) => {
    const cells: { at: [number, number]; needs: { rule: string; chords?: [number, number][] }[] }[] = [];
    for (let c = 0; c < w; c++) if (c !== 4) cells.push({ at: [3, c], needs: [{ rule: 'B', chords: [[3, 0]] }] });
    for (let r = 0; r < 7; r++) if (r !== 3) cells.push({ at: [r, 4], needs: [{ rule: 'A', chords: [[2, 5]] }] });
    cells.push({ at: [3, 4], needs: [{ rule: 'B', chords: [[3, 0]] }, { rule: 'A', chords: [[2, 5]] }] });
    return build(7, w, cells, ['A', 'B']);
  };

  it('a hit: the victim tile loses every chord, both lines go in waves a tile a step, H once each, the hot step', () => {
    const { board, rules, cell } = cross();
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: cell(3, 1), d0: 3, d1: 0, rule: B });
    settle(lc);
    if (lc.ca.generation % 2) lc.step();
    expect(lc.lineTiles(2)).toBe(9);
    expect(lc.tap({ cell: cell(0, 4), d0: 2, d1: 5, rule: A })).toBeNull();
    const X = cell(3, 4);
    const hits: [number, number][] = [];
    const goneAt = new Map<number, number>();
    const drawn = new Set<number>([...Array(lc.n).keys()].filter((c) => lc.ch.D[c]));
    for (let k = 1; k <= 14; k++) {
      lc.step();
      for (let c = 0; c < lc.n; c++) {
        if (lc.ch.H[c]) hits.push([k, c]);
        if (lc.ch.D[c]) drawn.add(c);
        else if (drawn.has(c) && !goneAt.has(c)) goneAt.set(c, k);
      }
      if (k === 5) {
        // the hot step: X still holds B's rule, no chords, waves out of both ends
        expect(lc.ch.rule[X]).toBe(B);
        expect(lc.ch.D[X]).toBe(0);
        expect(lc.ch.W[X]).toBe(bits(0, 3));
        expect(lc.tap({ cell: X, d0: 2, d1: 5, rule: A })).toBe('rival tile');
        expect(lc.tap({ cell: X, d0: 3, d1: 0, rule: B })).toBe('hot');
      }
      if (k === 6) expect(lc.ch.rule[X]).toBe(0);
    }
    // A drew (1,4) at 2 and (2,4) at 4, aimed at X at 4, hit at 5: X the victim, (2,4) the hitter
    expect(hits).toEqual([[5, cell(2, 4)], [5, X]]);
    expect(goneAt.get(X)).toBe(5);
    expect(goneAt.get(cell(2, 4))).toBe(5);
    expect(goneAt.get(cell(1, 4))).toBe(6);
    expect(goneAt.get(cell(0, 4))).toBe(7);
    for (let c = 0; c < 9; c++) expect(goneAt.get(cell(3, c)), `(3,${c})`).toBe(5 + Math.abs(c - 4));
    expect(lc.lineTiles(1) + lc.lineTiles(2)).toBe(0);
    for (let c = 0; c < lc.n; c++) expect(lc.ch.rule[c] | lc.ch.W[c] | lc.ch.T[c]).toBe(0);
    expect(lc.ch.D[cell(4, 4)]).toBe(0); // A's hitter never drew into or past X
  });

  it('a tip d tiles ahead of the hit flees and is caught at about 2d; its extra chords go too', () => {
    const { board, rules, cell } = cross(13);
    for (const lag of [4, 6, 8]) {
      const lc = new LineCA(board, table);
      const A = lc.addRule(rules.A, 1);
      const B = lc.addRule(rules.B, 2);
      lc.tap({ cell: cell(3, 3), d0: 3, d1: 0, rule: B }); // B's east tip passes X at 2
      run(lc, lag);
      lc.tap({ cell: cell(1, 4), d0: 2, d1: 5, rule: A }); // A hits X at lag + 3
      const h = lag + 3;
      let lastEast = -1, east0 = -1;
      for (let g = lag + 1; g <= h + 40; g++) {
        lc.step();
        if (g === h) {
          expect(lc.ch.H[cell(3, 4)]).toBe(1);
          for (let c = 12; c > 4; c--) if (lc.ch.D[cell(3, c)]) { east0 = c; break; }
        }
        if (g > h) for (let c = 5; c < 13; c++) if (lc.ch.D[cell(3, c)]) lastEast = g;
      }
      const d = east0 - 4; // the tip's tile, tiles east of X
      expect(d).toBeGreaterThan(0);
      expect(lastEast - h + 1).toBeGreaterThanOrEqual(2 * d - 1);
      expect(lastEast - h + 1).toBeLessThanOrEqual(2 * d + 1);
      expect(lc.lineTiles(2)).toBe(0);
    }
  });

  it('head-on rival tips: both lines die', () => {
    const { board, rules } = row(8, ['A', 'B']);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    lc.tap({ cell: 5, d0: 0, d1: 3, rule: B });
    const H: [number, number][] = [];
    for (let g = 1; g <= 12; g++) {
      lc.step();
      for (let c = 0; c < 8; c++) if (lc.ch.H[c]) H.push([g, c]);
    }
    // A: 1 at 2, 2 at 4; B: 4 at 2, 3 at 4 (and 6, 7 east): tips at 2 and 3 face each other at 4, hit at 5
    expect(H).toEqual([[5, 2], [5, 3]]);
    for (let c = 0; c < 8; c++) expect(lc.ch.D[c]).toBe(0);
  });

  it('a crash: tips of two rules into one empty tile both die; H on the tile, then once on each crashed tip', () => {
    const { board, rules } = row(9, ['A', 'B']);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    lc.tap({ cell: 6, d0: 0, d1: 3, rule: B });
    const H: [number, number][] = [];
    const W3: number[] = [];
    for (let g = 1; g <= 14; g++) {
      lc.step();
      for (let c = 0; c < 9; c++) if (lc.ch.H[c]) H.push([g, c]);
      W3.push(lc.ch.W[3]);
    }
    // both aim at 3 from 2 and 4 at gen 4 → crash at 5; the crashed tips (2, 4) see its waves at 6
    expect(H).toEqual([[5, 3], [6, 2], [6, 4]]);
    expect(W3.filter((w) => w !== 0)).toEqual([bits(0, 3)]); // one step of W, both ways
    for (let c = 0; c < 9; c++) expect(lc.ch.D[c] | lc.ch.rule[c] | lc.ch.W[c]).toBe(0);
  });

  it('a tip at a tail into a rival tile is no hit (fix 1)', () => {
    const { board, rules } = row(4, ['A', 'B'], 1, {
      1: [{ rule: 'A', chords: [[3, 0]] }, { rule: 'B', none: [0] }],
      2: [{ rule: 'A', none: [3] }, { rule: 'B', chords: [[3, 0]] }],
    });
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: 3, d0: 0, d1: 3, rule: B });
    settle(lc); // B on 2, 3
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    settle(lc); // A on 0, 1: its tip at 1 faces 2, which has no A chord through 3
    expect(lc.lineTiles(1)).toBe(2);
    expect(lc.lineTiles(2)).toBe(2);
    expect(lc.ch.H[1] | lc.ch.H[2]).toBe(0);
  });

  // A strand through X = (1,1) twice: (1,0) → X (3,0) → (1,2) (3,2) → (0,2) (5,4) → X (1,2) → (0,1) (5,3); a rival A
  // taps (2,2) (5,2) and hits (1,2) from below.
  const twice = () => build(3, 3, [
    { at: [1, 0], needs: [{ rule: 'B', chords: [[0, 3]] }] },
    { at: [1, 1], needs: [{ rule: 'B', chords: [[3, 0], [1, 2]] }] },
    { at: [1, 2], needs: [{ rule: 'B', chords: [[3, 2]] }, { rule: 'A', chords: [[5, 2]] }] },
    { at: [0, 2], needs: [{ rule: 'B', chords: [[5, 4]] }] },
    { at: [0, 1], needs: [{ rule: 'B', chords: [[5, 3]] }] },
    { at: [2, 2], needs: [{ rule: 'A', chords: [[5, 2]] }] },
  ], ['A', 'B']);

  it('a wave through a twice-visited tile takes one chord and leaves the other, then comes back for it', () => {
    const { board, rules, cell } = twice();
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: cell(1, 0), d0: 3, d1: 0, rule: B });
    settle(lc);
    const X = cell(1, 1);
    expect(lc.ch.D[X]).toBe(bits(0, 1, 2, 3));
    expect(lc.components()).toHaveLength(1);
    expect(lc.components()[0].chords).toBe(6);
    if (lc.ca.generation % 2) lc.step();
    lc.tap({ cell: cell(2, 2), d0: 5, d1: 2, rule: A }); // aimed at (1,2) at once
    lc.step(); // the hit
    expect(lc.ch.H[cell(1, 2)]).toBe(1);
    expect(lc.ch.D[cell(1, 2)]).toBe(0);
    lc.step(); // the waves reach X across 0 and (0,2) across 5
    expect(lc.ch.D[X]).toBe(bits(1, 2));
    expect(lc.ch.D[cell(0, 2)]).toBe(0);
    lc.step(); // the second wave reaches X across 1; the first reaches (1,0)
    expect(lc.ch.D[X]).toBe(0);
    expect(lc.ch.D[cell(1, 0)]).toBe(0);
    settle(lc);
    expect(lc.lineTiles(2)).toBe(0);
  });

  it('a wave stops at a loose end: it never jumps to another line of its rule on the same tile', () => {
    // the draw-beside board, with a rival C entering (1,2) from below
    const { board, rules, cell } = build(3, 3, [
      { at: [0, 1], needs: [{ rule: 'B', chords: [[5, 3]] }] },
      { at: [0, 2], needs: [{ rule: 'B', chords: [[4, 0]] }] },
      { at: [1, 0], needs: [{ rule: 'B', chords: [[0, 3]] }] },
      { at: [1, 1], needs: [{ rule: 'B', chords: [[3, 0], [1, 2]] }] },
      { at: [1, 2], needs: [{ rule: 'B', chords: [[3, 0]] }, { rule: 'A', chords: [[5, 2]] }] },
      { at: [2, 2], needs: [{ rule: 'A', chords: [[5, 2]] }] },
    ], ['A', 'B']);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: cell(1, 0), d0: 3, d1: 0, rule: B });
    settle(lc);
    lc.tap({ cell: cell(0, 1), d0: 3, d1: 5, rule: B });
    settle(lc);
    lc.tap({ cell: cell(2, 2), d0: 5, d1: 2, rule: A });
    settle(lc);
    const X = cell(1, 1);
    expect(lc.ch.D[X]).toBe(bits(1, 2)); // line 1 went; line 2 (X's (1,2) chord) is untouched
    expect(lc.ch.D[cell(0, 1)]).toBe(bits(5, 3));
    expect(lc.ch.D[cell(0, 2)]).toBe(bits(4, 0));
    expect(lc.ch.D[cell(1, 0)]).toBe(0);
    expect(lc.lineTiles(1)).toBe(0);
  });
});

describe('line CA: taps', () => {
  it('refuses as Spectacle does: no chord, rival tile, drawn, hot, heads, territory', () => {
    const { board, rules } = row(9, ['A', 'B'], 1, { 8: [{ rule: 'A', none: [3, 0] }, { rule: 'B', chords: [[3, 0]] }] });
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    expect(lc.tap({ cell: 8, d0: 3, d1: 0, rule: A })).toBe('no chord');
    expect(lc.tap({ cell: 0, d0: 3, d1: 0, rule: 99 })).toBe('no chord');
    expect(lc.tap({ cell: 0, d0: 3, d1: 0, rule: A }, () => true)).toBe('territory');
    expect(lc.ch.D[0]).toBe(0);
    expect(lc.tap({ cell: 0, d0: 3, d1: 0, rule: A })).toBeNull();
    expect(lc.tap({ cell: 0, d0: 0, d1: 3, rule: A })).toBe('drawn');
    expect(lc.tap({ cell: 0, d0: 3, d1: 0, rule: B })).toBe('rival tile');
    expect(lc.tap({ cell: 5, d0: 3, d1: 0, rule: A })).toBe('heads'); // A's live east tip + two more > 2
    lc.step();
    lc.step(); // A: 1 drawn, one live tip left (the west one was a tail)
    expect(lc.countTips(1)).toBe(1);
    expect(lc.tap({ cell: 5, d0: 3, d1: 0, rule: A })).toBe('heads');
    const loneTip = new LineCA(board, table, { oneWay: true });
    const A2 = loneTip.addRule(rules.A, 1);
    expect(loneTip.tap({ cell: 0, d0: 3, d1: 0, rule: A2 })).toBeNull();
    expect(loneTip.tap({ cell: 5, d0: 3, d1: 0, rule: A2 })).toBeNull(); // one-way: two lines of one tip each
    expect(loneTip.tap({ cell: 3, d0: 3, d1: 0, rule: A2 })).toBe('heads');
    settle(lc);
    expect(lc.lineTiles(1)).toBe(8);
  });

  it('refuses a tap on a tile just wiped (hot) and on a crash tile', () => {
    const { board, rules } = row(9, ['A', 'B']);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    const B = lc.addRule(rules.B, 2);
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    lc.tap({ cell: 6, d0: 0, d1: 3, rule: B });
    run(lc, 5); // crash at 3 on gen 5
    expect(lc.ch.H[3]).toBe(1);
    expect(lc.tap({ cell: 3, d0: 3, d1: 0, rule: A })).toBe('hot');
    lc.step(); // 2 and 4 wiped: hot for B... 4 is B's
    expect(lc.ch.rule[4]).toBe(B);
    expect(lc.tap({ cell: 4, d0: 3, d1: 0, rule: B })).toBe('hot');
    expect(lc.tap({ cell: 4, d0: 3, d1: 0, rule: A })).toBe('rival tile');
    lc.step();
    expect(lc.ch.rule[4]).toBe(0);
    expect(lc.tap({ cell: 3, d0: 3, d1: 0, rule: A })).toBeNull();
  });

  it('LineCA never refuses for the respawn delay: that is the host\'s', () => {
    const { board, rules } = row(3);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    expect(lc.tap({ cell: 1, d0: 3, d1: 0, rule: A })).toBeNull();
  });

  it('addRule: indices from 1, the same rule and owner again is the same index, owners 1..15', () => {
    const { board, rules } = row(3, ['A', 'B']);
    const lc = new LineCA(board, table);
    expect(lc.addRule(rules.A, 1)).toBe(1);
    expect(lc.addRule(rules.B, 2)).toBe(2);
    expect(lc.addRule(rules.A, 1)).toBe(1);
    expect(lc.addRule(rules.A, 3)).toBe(3); // another player: another identity
    expect(() => lc.addRule(rules.A, 16)).toThrow();
    expect(lc.ownerOf).toEqual([0, 1, 2, 3]);
  });

  it('changed() lists the cells whose chords or owner changed, the host\'s writes included', () => {
    const { board, rules } = row(5);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    lc.step();
    expect(Array.from(lc.changed())).toEqual([]);
    lc.tap({ cell: 2, d0: 3, d1: 0, rule: A });
    lc.step(); // gen 2, a go step: the tap's tile (the host's write) and the two it grew into
    expect(Array.from(lc.changed()).sort()).toEqual([1, 2, 3]);
    lc.step(); // tips retire: no chord changes
    expect(Array.from(lc.changed())).toEqual([]);
    lc.step();
    expect(Array.from(lc.changed()).sort()).toEqual([0, 4]);
  });
});

describe('engine hooks the line CA adds (writeState, touch, lastChanged)', () => {
  it('writeState writes a state channel and lists the slot; a const channel is refused', () => {
    const { board, rules } = row(5);
    const lc = new LineCA(board, table);
    const A = lc.addRule(rules.A, 1);
    expect(() => lc.ca.writeState('geo', 0, 3)).toThrow(/const/);
    lc.ca.writeState('rule', 2, A); // off the board (the sentinel) is ignored, as `write` does
    lc.ca.writeState('rule', lc.n, A);
    expect(lc.ch.rule[lc.n]).toBe(0);
    lc.ca.writeState('rule', 2, 0);
    lc.step();
    expect(Array.from(lc.ca.lastChanged())).toEqual([]);
    lc.tap({ cell: 2, d0: 3, d1: 0, rule: A });
    lc.step(); // gen 2, a go step: cells 1 and 3 draw (the tap's own cell is the host's write, not the step's)
    expect(Array.from(lc.ca.lastChanged()).sort()).toEqual([1, 3]);
    expect(lc.ca.changed).toBe(2);
    expect(Array.from(lc.changed()).sort()).toEqual([1, 2, 3]);
  });

  it('touch lists a quiet slot: without it a waiting tip would never be seen on its go step', () => {
    const { board, rules } = row(5);
    const lc = new LineCA(board, table, { period: 4 });
    const A = lc.addRule(rules.A, 1);
    lc.tap({ cell: 0, d0: 3, d1: 0, rule: A });
    run(lc, 3); // gen 1-3: nothing changes after the rim tip retires, the active set empties
    expect(lc.ca.active).toBe(0);
    lc.step(); // gen 4: the cell ahead of the tip was touched
    expect(lc.ch.D[1]).toBe(bits(0, 3));
  });
});

// ---------------------------------------------------------------- the walker at double time (§2.10 (a))

describe('line CA: single taps against walk at double time', () => {
  it.each(['l2', 'l3'] as const)('%s', (name) => {
    const board = realBoards[name];
    const N = board.h * board.w;
    const rand = rng(name === 'l2' ? 11 : 12);
    for (let trial = 0; trial < (name === 'l2' ? 150 : 60); trial++) {
      const rule = table.sample(rand, 'train');
      const ex = table.exits(rule, board);
      const cands: [number, number, number][] = [];
      for (let i = 0; i < board.n; i++) for (let d = 0; d < 6; d++) {
        const e = ex[d * N + board.pos[i]];
        if (e >= 0) cands.push([i, d, e]);
      }
      if (!cands.length) continue;
      const [cell, d0, d1] = cands[Math.floor(rand() * cands.length)];
      const lc = new LineCA(board, table);
      const A = lc.addRule(rule, 1);
      expect(lc.tap({ cell, d0, d1, rule: A })).toBeNull();
      const p = board.pos[cell];
      const s = walk(ex, board, Math.floor(p / board.w), p % board.w, d0, d1);
      const L = s.rows.length;
      // chord at walk index i is drawn at 2|i| (a loop: 2·min(i, L - i))
      const due = new Map<number, number>();
      s.rows.forEach((r, k) => {
        const c = board.cellOf[r * board.w + s.cols[k]];
        const i = s.index[k];
        const t = 2 * (s.closed ? Math.min(i, L - i) : Math.abs(i));
        due.set(c * 6 + s.ins[k], t);
        due.set(c * 6 + s.outs[k], t);
      });
      const T = 3 * L + 10; // a loop: closing, the probe's round trip, the flood
      for (let g = 1; g <= T; g++) {
        lc.step();
        for (const [k, t] of due) {
          const on = (lc.ch.D[Math.floor(k / 6)] >> (k % 6)) & 1;
          if (on !== +(g >= t)) expect.fail(`${name} trial ${trial} ${table.describe(rule)}: chord end ${k} at gen ${g}, due ${t}`);
        }
      }
      let drawn = 0;
      for (let c = 0; c < lc.n; c++) for (let d = 0; d < 6; d++) drawn += (lc.ch.D[c] >> d) & 1;
      expect(drawn).toBe(due.size);
      expect(lc.countTips(1)).toBe(0);
      const comps = lc.components();
      expect(comps).toHaveLength(1);
      expect(comps[0].closed).toBe(s.closed);
      if (s.closed) for (let c = 0; c < lc.n; c++) expect(lc.ch.closed[c]).toBe(lc.ch.D[c]);
      else for (let c = 0; c < lc.n; c++) expect(lc.ch.closed[c]).toBe(0);
    }
  });
});

// ---------------------------------------------------------------- fuzz (§5)

interface FuzzCase { seed: number; board: 'l2' | 'l3'; players: number; knobs: Partial<LineKnobs>; steps: number }

function fuzzCase(seed: number): FuzzCase {
  const rand = rng(1000 + seed);
  const variants: Partial<LineKnobs>[] = [{}, {}, {}, { period: 4 }, { period: 3 }, { fuel: 3 }, { oneWay: true }, { probe: false }, { maxTips: 4 }];
  return {
    seed,
    board: seed % 4 === 3 ? 'l3' : 'l2',
    players: 2 + Math.floor(rand() * 3),
    knobs: variants[seed % variants.length],
    steps: 2000,
  };
}

function playFuzz(fc: FuzzCase): { taps: number; hits: number; loops: number } {
  const board = realBoards[fc.board];
  const N = board.h * board.w;
  const rand = rng(fc.seed);
  const lc = new LineCA(board, table, fc.knobs);
  const twin = new LineCA(board, table, fc.knobs);
  twin.full = true;
  const label = `seed ${fc.seed} (${fc.board}, ${fc.players} players, ${JSON.stringify(fc.knobs)})`;
  const rules: number[] = [];
  const exits: Int8Array[] = [];
  const keys = new Set<string>();
  while (rules.length < fc.players) {
    const rule = table.sample(rand, 'train');
    if (keys.has(table.key(rule))) continue;
    keys.add(table.key(rule));
    const o = rules.length + 1;
    rules.push(lc.addRule(rule, o));
    twin.addRule(rule, o);
    exits.push(table.exits(rule, board));
  }
  const inv = new Invariants(lc, label);
  let taps = 0, hits = 0, loops = 0;
  for (let s = 0; s < fc.steps; s++) {
    // a few tap attempts: a random chord of a random player's rule; now and then a retap on a drawn chord
    const tries = rand() < 0.3 ? 1 + Math.floor(rand() * 3) : 0;
    for (let k = 0; k < tries; k++) {
      const pl = Math.floor(rand() * fc.players);
      const cell = Math.floor(rand() * board.n);
      const p = board.pos[cell];
      const ds: [number, number][] = [];
      for (let d = 0; d < 6; d++) {
        const e = exits[pl][d * N + p];
        if (e >= 0) ds.push([d, e]);
      }
      if (!ds.length) continue;
      const [d0, d1] = ds[Math.floor(rand() * ds.length)];
      const blocked = rand() < 0.05 ? () => true : undefined;
      if (rand() < 0.15) {
        const a = lc.retap(cell, d0, d1, rules[pl]);
        expect(twin.retap(cell, d0, d1, rules[pl]), label).toBe(a);
      } else {
        const why = lc.tap({ cell, d0, d1, rule: rules[pl] }, blocked);
        expect(twin.tap({ cell, d0, d1, rule: rules[pl] }, blocked), label).toBe(why);
        if (!why) taps++;
      }
    }
    const pre = inv.before();
    lc.step();
    twin.step();
    for (let c = 0; c < lc.n; c++) hits += lc.ch.H[c];
    inv.after(pre, twin);
    if (s % 50 === 49) checkStretches(lc, `${label} gen ${lc.ca.generation}`);
  }
  settle(lc);
  checkStretches(lc, `${label} at rest`);
  // at rest: closed ⇔ components().closed, chord for chord
  const sets = lineSets(lc);
  checkComponents(lc, sets, `${label} at rest`);
  for (const s of sets) {
    const closedLine = s.loose.length === 0;
    if (closedLine) loops++;
    for (const k of s.keys) {
      if (isOn(lc.ch.closed, k) !== (closedLine && lc.knobs.probe)) expect.fail(`${label} at rest: closed flag of chord end ${k}`);
    }
  }
  return { taps, hits, loops };
}

describe('line CA: fuzz, every invariant after every step', () => {
  // the doc's 40 seeds (fewer when FUZZ_SEEDS says so); LINE_FUZZ_SEEDS asks for more
  const n = Number(process.env.LINE_FUZZ_SEEDS) || Math.min(FUZZ_SEEDS, 40);
  it.each(seeds(n))('seed %i', (seed) => {
    const r = playFuzz(fuzzCase(seed));
    expect(r.taps).toBeGreaterThan(10);
  });

  it('the fuzz exercises hits, loops and refusals', () => {
    let hits = 0, loops = 0;
    for (const seed of [0, 1, 2]) {
      const r = playFuzz({ ...fuzzCase(seed), steps: 1500 });
      hits += r.hits;
      loops += r.loops;
    }
    expect(hits).toBeGreaterThan(5);
    expect(loops).toBeGreaterThan(0);
  });

  it('the defaults are the doc\'s', () => {
    expect(DEFAULT_LINE_KNOBS).toEqual({ period: 2, oneWay: false, fuel: 0, probe: true, maxTips: 2 });
  });
});
