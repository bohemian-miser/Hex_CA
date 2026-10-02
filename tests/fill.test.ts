import { describe, expect, it } from 'vitest';
import { CA } from '../src/engine.js';
import { Field, hexField, maskedField, presetWalls } from '../src/field.js';
import { FILLED, OFF, ON, WALL, bump, fillCA, fillRule, paint, scramble, stateOf } from '../src/fill.js';
import { coordsOf, hexDistance, indexOf, makeBoard } from '../src/hex.js';
import { randomLoop, rng } from '../src/lines.js';
import { oracle, settleBound } from '../src/oracle.js';
import { channel, count } from './helpers.js';

// The pictures of DESIGN.md §8 item 1 through the CA, each on the full
// hexagon and on a blob-masked field. Every edit is settled under watch: at
// every step the latch invariant (a certified cell shows exactly its
// certified picture's fill) and no overshoot (FILLED only where the old
// quiet state or the new oracle has it); at quiet, state == oracle and every
// commit landed within settleBound.

const R = 10;
/** Far past any bound at R = 10: a run that gets here never settles. */
const MAX_STEPS = 20_000;

interface Picture {
  on?: number[];
  wall?: number[];
}

/** A booted automaton that remembers the oracle of every picture it was shown, by serial. */
class Session {
  readonly ca: CA;
  /** serial → the oracle of the picture the edits with that stamp made. */
  readonly pictures = new Map<number, Uint8Array>();

  constructor(readonly field: Field, readonly wallsBound = false) {
    this.ca = fillCA(field, { wallsBound });
    // Epoch 0 certifies the boot picture.
    this.pictures.set(0, this.oracle());
  }

  picture(): (slot: number) => number {
    const p = channel(this.ca, 'paint');
    return (slot) => p[slot];
  }

  oracle(): Uint8Array {
    return oracle(this.field, this.picture(), this.wallsBound);
  }

  /** The field slots of `slots` whose paint is not already `v`: what an edit changes. */
  changes(slots: readonly number[], v: number): number[] {
    const p = channel(this.ca, 'paint');
    const { isField } = this.field.topo;
    return slots.filter((s) => isField[s] === 1 && p[s] !== v);
  }

  /** Paint a picture as one batch (one serial) and settle it. */
  edit(pic: Picture, label: string): Settled {
    const edited = [...this.changes(pic.wall ?? [], WALL), ...this.changes(pic.on ?? [], ON)];
    expect(edited.length, `${label}: the edit changes something`).toBeGreaterThan(0);
    paint(this.ca, pic.wall ?? [], WALL);
    paint(this.ca, pic.on ?? [], ON);
    return this.settle(this.ca.generation + 1, edited, label);
  }

  /** Record the picture for `serial` and step to quiet under the per-step checks. */
  settle(serial: number, edited: number[], label: string): Settled {
    const { ca, field } = this;
    this.pictures.set(serial, this.oracle());
    const bound = settleBound(field, this.picture(), edited);
    const after = this.pictures.get(serial)!;
    const before = stateOf(ca); // the quiet state the last picture left; paint changes no state
    const cells = field.topo.cells;
    const paintCh = channel(ca, 'paint');
    const epoch = channel(ca, 'epoch');
    const commit = channel(ca, 'commit');
    const state = channel(ca, 'state');

    let steps = 0;
    let rootCommit = -1;
    let allCommit = -1;
    for (;;) {
      ca.step();
      steps++;
      let all = true;
      for (const s of cells) {
        const filled = state[s] === FILLED;
        if (commit[s] !== serial) all = false;
        if (paintCh[s] === OFF && commit[s] === epoch[s]) {
          const certified = this.pictures.get(commit[s]);
          if (!certified) throw new Error(`${label}: step ${steps}: slot ${s} certified for an unknown serial ${commit[s]}`);
          if (filled !== (certified[s] === FILLED)) {
            throw new Error(`${label}: step ${steps}: slot ${s} is certified for serial ${commit[s]} but shows the wrong fill`);
          }
        }
        if (filled && before[s] !== FILLED && after[s] !== FILLED) {
          throw new Error(`${label}: step ${steps}: slot ${s} shows FILLED, which neither the old state nor the new oracle has`);
        }
      }
      if (rootCommit < 0 && commit[field.root] === serial) rootCommit = steps;
      if (allCommit < 0 && all) allCommit = steps;
      if (ca.changed === 0) break;
      if (steps >= MAX_STEPS) throw new Error(`${label}: not quiet after ${MAX_STEPS} steps`);
    }

    expect(ca.active, `${label}: quiet means nothing listed`).toBe(0);
    expectState(stateOf(ca), after, label);
    for (const s of cells) {
      if (commit[s] !== serial || epoch[s] !== serial) {
        throw new Error(`${label}: slot ${s} at quiet has epoch ${epoch[s]}, commit ${commit[s]}; expected ${serial}`);
      }
    }
    expect(allCommit, `${label}: every cell committed by the bound`).toBeLessThanOrEqual(bound);
    return { steps, rootCommit, allCommit, bound, filled: count(after, FILLED) };
  }
}

interface Settled {
  steps: number;
  /** Steps until the root's commit was the serial. */
  rootCommit: number;
  /** Steps until every field slot's commit was the serial. */
  allCommit: number;
  bound: number;
  /** FILLED cells in the oracle of the settled picture. */
  filled: number;
}

function expectState(actual: Uint8Array, expected: Uint8Array, label: string): void {
  let same = actual.length === expected.length;
  for (let i = 0; same && i < actual.length; i++) same = actual[i] === expected[i];
  if (!same) expect(actual, `${label}: state == oracle`).toEqual(expected);
}

/** Every field slot satisfying a coordinate predicate. */
function slotsWhere(field: Field, pred: (q: number, r: number, d: number) => boolean): number[] {
  const out: number[] = [];
  for (const slot of field.topo.cells) {
    const [q, r] = coordsOf(field.board, slot);
    if (pred(q, r, hexDistance(q, r))) out.push(slot);
  }
  return out;
}

/**
 * The longest of a few random loops drawn on a hexagon of radius `inner`,
 * placed on the field's board: on a masked field it stays inside the mask.
 */
function loopIn(field: Field, inner: number, seeds: number[]): number[] {
  const small = makeBoard(inner);
  let best: number[] = [];
  for (const seed of seeds) {
    const loop = randomLoop(small, rng(seed));
    if (loop && loop.length > best.length) best = loop;
  }
  expect(best.length).toBeGreaterThan(6);
  return best.map((s) => {
    const [q, r] = coordsOf(small, s);
    return indexOf(field.board, q, r);
  });
}

interface Arena {
  name: string;
  field: Field;
  /** A hexagon radius whose interior lies within the field: where random loops go. */
  inner: number;
}

/** The two fields every picture runs on: the full hexagon and a blob-masked one of the same board. */
function arenas(): Arena[] {
  const board = makeBoard(R);
  const mask = new Uint8Array(board.size);
  for (const s of board.cells) {
    const [q, r] = coordsOf(board, s);
    // The blob rim of presetWalls, as a mask: radius R-1 or R-2 depending on the angle.
    if (hexDistance(q, r) <= R - 1 - Math.floor(2 * Math.abs(Math.sin(0.7 * q + 1.3 * r + 1)))) mask[s] = 1;
  }
  return [
    { name: 'hexagon', field: hexField(R), inner: R },
    { name: 'blob mask', field: maskedField(board, mask), inner: 7 },
  ];
}

const ARENAS = arenas();

describe('fill rule: the channel table (DESIGN.md §2)', () => {
  it('declares the 22 channels in order with the table kinds, dead values and watch set', () => {
    const names = fillRule.channels.map((c) => c.name);
    expect(names).toEqual([
      'field', 'id', 'gpar', 'area', 'paint', 'stamp',
      'epoch', 'rim', 'redge', 'leader', 'dist', 'rpar', 'cdone', 'sub', 'rsize',
      'gmax', 'M', 'qt', 'run', 'commit', 'want', 'state',
    ]);
    const by = (name: string) => fillRule.channels.find((c) => c.name === name)!;
    for (const n of ['field', 'id', 'gpar', 'area']) expect(by(n).kind).toBe('const');
    for (const n of ['paint', 'stamp']) expect(by(n).kind).toBe('input');
    for (const n of ['gmax', 'M', 'qt', 'run', 'commit']) expect(by(n).kind).toBe('gate');
    for (const n of ['want', 'state']) expect(by(n).kind).toBe('out');
    const watched = fillRule.channels.filter((c) => c.watch).map((c) => c.name);
    expect(watched).toEqual(['epoch', 'rim', 'redge', 'leader', 'dist', 'rpar', 'cdone', 'sub', 'rsize']);
    expect(by('paint').dead).toBe(WALL);
    expect(by('state').dead).toBe(WALL);
    expect(by('rim').dead).toBe(1);
    expect(by('qt').dead).toBe(1);
    expect(by('qt').init).toBe(0);
    expect(by('commit').init).toBe(-1);
    expect(by('commit').dead).toBe(-1);
    // dist's init and dead are the field's CAP on a built automaton.
    const field = hexField(4);
    const ca = fillCA(field);
    const dist = ca.rule.channels.find((c) => c.name === 'dist')!;
    expect(dist.init).toBe(field.CAP);
    expect(dist.dead).toBe(field.CAP);
    expect(ca.u).toEqual({ N: field.N, CAP: field.CAP, K: field.K, step: 1, wallsBound: 0 });
  });
});

describe('fill rule: boot', () => {
  it.each(ARENAS)('$name: an empty board boots to quiet, fills nothing, and then costs nothing', ({ field }) => {
    const session = new Session(field);
    // The boot is at worst an edit at the root: everything is fresh at step 1.
    const boot = session.settle(0, [field.root], 'boot');
    expect(boot.filled).toBe(0);
    expect(count(stateOf(session.ca), FILLED)).toBe(0);
    for (const s of field.topo.cells) expect(session.ca.get('state', s)).toBe(OFF);
    expect(session.ca.active).toBe(0);
    const snapshot = session.ca.ch.map((a) => a.slice());
    session.ca.step();
    expect(session.ca.changed).toBe(0);
    expect(session.ca.active).toBe(0);
    session.ca.ch.forEach((a, c) => expect(a).toEqual(snapshot[c]));
  });
});

/** The §8 pictures. `fills` says whether the oracle fills anything, per arena, pinning what each case means. */
const PICTURES: Array<{
  name: string;
  picture: (a: Arena) => Picture;
  fills: Record<string, boolean>;
}> = [
  {
    name: 'a loop fills its inside',
    picture: (a) => ({ on: loopIn(a.field, a.inner, [1, 2, 3, 4, 5, 6]) }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'a halving bridge (row 0) is a tie on the hexagon',
    picture: (a) => ({ on: slotsWhere(a.field, (_q, r) => r === 0) }),
    fills: { hexagon: false, 'blob mask': true },
  },
  {
    name: 'an off-centre bridge (row 1) fills the smaller side',
    picture: (a) => ({ on: slotsWhere(a.field, (_q, r) => r === 1) }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'nested loops (d = 2, 6) fill both interiors',
    picture: (a) => ({ on: slotsWhere(a.field, (_q, _r, d) => d === 2 || d === 6) }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'a loop round more than half the field fills everything inside',
    // d = R-1 on the hexagon; the blob's rim is at R-1 or R-2, so d = 6 there.
    picture: (a) => ({ on: slotsWhere(a.field, (_q, _r, d) => d === (a.name === 'hexagon' ? R - 1 : 6)) }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'two parallel bridges (rows -3, 5) fill the two smaller strips',
    picture: (a) => ({ on: slotsWhere(a.field, (_q, r) => r === -3 || r === 5) }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'a loop round a wall island fills the annulus',
    picture: (a) => ({
      wall: slotsWhere(a.field, (_q, _r, d) => d <= 1),
      on: slotsWhere(a.field, (_q, _r, d) => d === 4),
    }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'a wall ring near the rim with a loop inside fills only the loop',
    picture: (a) => ({
      wall: slotsWhere(a.field, (_q, _r, d) => d >= R - 2),
      on: slotsWhere(a.field, (_q, _r, d) => d === 3),
    }),
    fills: { hexagon: true, 'blob mask': true },
  },
  {
    name: 'a bridge onto that ring fills its smaller side',
    picture: (a) => ({
      wall: slotsWhere(a.field, (_q, _r, d) => d >= R - 2),
      on: slotsWhere(a.field, (q, _r, d) => q === 2 && d < R - 2),
    }),
    fills: { hexagon: true, 'blob mask': true },
  },
];

describe('fill rule: the §8 pictures, one batch each on a booted automaton', () => {
  for (const arena of ARENAS) {
    describe(arena.name, () => {
      it.each(PICTURES)('$name', ({ name, picture, fills }) => {
        const session = new Session(arena.field);
        session.settle(0, [arena.field.root], `${arena.name} boot`);
        const settled = session.edit(picture(arena), `${arena.name}: ${name}`);
        expect(settled.filled > 0, `${name}: the oracle ${fills[arena.name] ? 'fills' : 'fills nothing'}`).toBe(fills[arena.name]);
        expect(settled.rootCommit).toBeGreaterThan(0);
        expect(settled.allCommit).toBeGreaterThanOrEqual(settled.rootCommit);
      });
    });
  }
});

describe('fill rule: sequences of edits', () => {
  it.each(ARENAS)('$name: a circuit drawn in two halves fills once closed', ({ name, field, inner }) => {
    const session = new Session(field);
    session.settle(0, [field.root], 'boot');
    const loop = loopIn(field, inner, [7, 8, 9, 10]);
    const half = Math.floor(loop.length / 2);
    const first = session.edit({ on: loop.slice(0, half) }, `${name}: first half`);
    expect(first.filled).toBe(0); // an open line splits nothing
    const second = session.edit({ on: loop.slice(half) }, `${name}: second half`);
    expect(second.filled).toBeGreaterThan(0);
    expect(count(stateOf(session.ca), FILLED)).toBe(second.filled);
  });

  it.each(ARENAS)('$name: erasing a loop cell un-fills at the next commit, never before', ({ name, field, inner }) => {
    const session = new Session(field);
    session.settle(0, [field.root], 'boot');
    const loop = loopIn(field, inner, [11, 12, 13]);
    const closed = session.edit({ on: loop }, `${name}: loop`);
    expect(closed.filled).toBeGreaterThan(0);
    const commit = channel(session.ca, 'commit');
    const state = channel(session.ca, 'state');
    const inside = stateOf(session.ca);
    // Erase: the old fill must stay until the root's new commit arrives, then go.
    const gap = loop[2];
    const serial = session.ca.generation + 1;
    paint(session.ca, [gap], OFF);
    let stale = 0;
    for (let steps = 1; commit[field.root] !== serial; steps++) {
      session.ca.step();
      if (steps >= MAX_STEPS) throw new Error('no commit');
      for (const s of field.topo.cells) if (state[s] === FILLED) stale++;
      expect(state[gap]).toBe(OFF); // the erased cell itself clears at once
      for (let i = 0; i < inside.length; i++) {
        if (i !== gap && state[i] !== inside[i]) throw new Error(`${name}: slot ${i} changed before the commit`);
      }
    }
    expect(stale).toBeGreaterThan(0);
    // From here settle() takes over with its checks (the serial is the erase's).
    const erased = session.settle(serial, [loop[2]], `${name}: erase`);
    expect(erased.filled).toBe(0);
    expect(count(stateOf(session.ca), FILLED)).toBe(0);
  });

  it.each(ARENAS)('$name: a bump re-certifies the same picture without changing what is shown', ({ name, field, inner }) => {
    const session = new Session(field);
    session.settle(0, [field.root], 'boot');
    const loop = loopIn(field, inner, [14, 15, 16]);
    session.edit({ on: loop }, `${name}: loop`);
    const shown = stateOf(session.ca);
    const state = channel(session.ca, 'state');
    const serial = session.ca.generation + 1;
    bump(session.ca);
    // Every step shows exactly what it showed before the bump.
    const ca = session.ca;
    let steps = 0;
    do {
      ca.step();
      steps++;
      for (let i = 0; i < shown.length; i++) if (state[i] !== shown[i]) throw new Error(`${name}: the bump changed slot ${i} at step ${steps}`);
    } while (ca.changed > 0 && steps < MAX_STEPS);
    expect(ca.changed).toBe(0);
    for (const s of field.topo.cells) expect(ca.get('commit', s)).toBe(serial);
    expect(steps).toBeLessThanOrEqual(settleBound(field, session.picture(), [field.root]));
  });
});

describe('fill rule: walls', () => {
  it.each(ARENAS)('$name: with wallsBound the island counts as exterior and the annulus still fills', ({ name, field }) => {
    const session = new Session(field, true);
    session.settle(0, [field.root], 'boot');
    const settled = session.edit({
      wall: slotsWhere(field, (_q, _r, d) => d <= 1),
      on: slotsWhere(field, (_q, _r, d) => d === 4),
    }, `${name}: island, wallsBound`);
    expect(settled.filled).toBe(slotsWhere(field, (_q, _r, d) => d === 2 || d === 3).length);
  });

  it("a hexagon with the 'blob' walls painted is a ragged field: a loop inside fills", () => {
    const field = hexField(R);
    const session = new Session(field);
    session.settle(0, [field.root], 'boot');
    const rim = session.edit({ wall: presetWalls(field, 'blob', 3) }, 'blob walls');
    expect(rim.filled).toBe(0);
    // A loop at d = 6: its inside (91 cells) outweighs the strip between it
    // and the blob's rim (at d = 8 or 9), which is the field's only edge region.
    const settled = session.edit({ on: slotsWhere(field, (_q, _r, d) => d === 6) }, 'loop inside the blob');
    const state = channel(session.ca, 'state');
    const insideLoop = slotsWhere(field, (_q, _r, d) => d < 6);
    expect(settled.filled).toBe(insideLoop.filter((s) => state[s] === FILLED).length);
    expect(settled.filled).toBeGreaterThan(80);
    // A wall spoke from the loop out to the rim is exterior (it is 6-connected
    // to the dead ring through walls), so the inside becomes an edge region —
    // and the largest one: the fill moves to the smaller strip outside.
    const spoke = session.edit({ wall: slotsWhere(field, (q, r, d) => q === 0 && r < 0 && d >= 6) }, 'wall spoke to the rim');
    // (The preset's island may sit inside the loop: a wall cell, not a fill.)
    for (const s of insideLoop) if (state[s] !== WALL) expect(state[s], `inside slot ${s}`).toBe(OFF);
    expect(spoke.filled).toBeGreaterThan(0);
    expect(spoke.filled).toBeLessThan(insideLoop.length);
  });
});

describe('fill rule: the settle bound where the rim branch dominates (DESIGN.md §5)', () => {
  it('a wall snake from the rim into a corridor commits within settleBound, past what Wd + Er allowed', () => {
    // The corridor: L OFF cells along row 0 ending one in from the east rim
    // (depth K - 1). A WALL snake along row 0 from the west rim to the
    // corridor's west end (Wd = 2R - L - 1 > 2Er); everything else ON. The
    // edit erases the corridor's east end (Dw = 2R - 1). redge enters at the
    // west end only once rim has run the whole snake, then floods the whole
    // corridor (L - 1 ≈ 2Er, not Er): allCommit = 7R - 2 here, over the
    // bound as first written (Wd + Er) by Er - 1. Needs the corridor's
    // leader (its max id) in the middle, so Er = (L - 1) / 2: scan idSeeds.
    const L = 9;
    const m = (L - 1) / 2;
    let field: Field | null = null;
    for (let idSeed = 0; idSeed < 400 && !field; idSeed++) {
      const f = hexField(R, { idSeed });
      const corridor: number[] = [];
      for (let q = R - L; q <= R - 1; q++) corridor.push(indexOf(f.board, q, 0));
      let leader = corridor[0];
      for (const s of corridor) if (f.ids[s] > f.ids[leader]) leader = s;
      if (corridor.indexOf(leader) === m) field = f;
    }
    expect(field, 'an idSeed with the leader mid-corridor').not.toBeNull();
    const board = field!.board;
    const corridor: number[] = [];
    for (let q = R - L; q <= R - 1; q++) corridor.push(indexOf(board, q, 0));
    const snake: number[] = [];
    for (let q = -R; q <= R - L - 1; q++) snake.push(indexOf(board, q, 0));
    const east = corridor[corridor.length - 1];
    const on = field!.topo.cells.filter((s) => !corridor.includes(s) && !snake.includes(s));

    const session = new Session(field!);
    session.settle(0, [field!.root], 'boot');
    session.edit({ on: [...on, east], wall: snake }, 'corridor, snake, the rest on');
    const serial = session.ca.generation + 1;
    paint(session.ca, [east], OFF);
    const settled = session.settle(serial, [east], 'erase the corridor east end'); // asserts allCommit <= settleBound
    // The components by hand, and the bound as §5 first had it: this case must exceed that.
    const Dw = 2 * R - 1, Er = m, Wd = snake.length - 1, K = field!.K;
    expect(settled.bound).toBe(Dw + Math.max(3 * Er, Wd + 2 * Er) + 3 * K + 3);
    expect(settled.allCommit).toBeGreaterThan(Dw + Math.max(3 * Er, Wd + Er) + 3 * K + 3);
    expect(settled.filled).toBe(0); // the corridor is the only region and touches the exterior
  });
});

describe('fill rule: scramble', () => {
  it.each(ARENAS)('$name: garbage then a bump recovers the right fill', ({ name, field, inner }) => {
    const session = new Session(field);
    session.settle(0, [field.root], 'boot');
    const loop = loopIn(field, inner, [17, 18, 19]);
    session.edit({ on: loop }, `${name}: loop`);
    const expected = session.oracle();
    scramble(session.ca, rng(5));
    expect(session.ca.active).toBe(field.N);
    // Garbage in the gate may show anything until the bump's commit: assert the end only.
    bump(session.ca);
    const steps = session.ca.run(MAX_STEPS);
    expect(session.ca.changed).toBe(0);
    expectState(stateOf(session.ca), expected, `${name}: after scramble + bump`);
    expect(steps).toBeLessThanOrEqual(settleBound(field, session.picture(), [field.root]));
  });
});
