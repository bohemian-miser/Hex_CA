import { describe, expect, it } from 'vitest';
import { CA, ChannelSpec, Rule, Topology, Uniforms } from '../src/engine.js';
import { fillCA, paint } from '../src/fill.js';
import { makeBoard } from '../src/hex.js';
import { rng } from '../src/lines.js';
import { fuzzStream, seeds } from './helpers.js';

/** A hexagon of radius R as a Topology, optionally masked down to `keep`. */
function hexTopo(R: number, keep?: (slot: number) => boolean): Topology {
  const b = makeBoard(R);
  const isField = new Uint8Array(b.size);
  const cells: number[] = [];
  for (const c of b.cells) {
    if (keep && !keep(c)) continue;
    isField[c] = 1;
    cells.push(c);
  }
  return { size: b.size, cells: Int32Array.from(cells), nbr: b.neighbours, isField };
}

const spec = (name: string, kind: ChannelSpec['kind'], init: number, dead: number): ChannelSpec => ({
  name, kind, bank: 0, init, dead, watch: kind === 'hidden',
});

// Toy rule, symmetric in its taps: a decaying max-flood of the input `src`,
// each cell charging its own const `cost` per hop; `sum` is the flood summed
// over the seven taps.
const SRC = 1, COST = 0, FLOOD = 2, SUM = 3;
const floodRule: Rule = {
  channels: [spec('cost', 'const', 1, 0), spec('src', 'input', 0, 0), spec('flood', 'hidden', 0, 0), spec('sum', 'out', 0, 0)],
  update(P, out) {
    let m = 0;
    for (let t = 1; t <= 6; t++) m = Math.max(m, P[FLOOD * 7 + t]);
    const flood = Math.max(P[SRC * 7], m - P[COST * 7]);
    out[FLOOD] = flood;
    let s = 0;
    for (let t = 0; t <= 6; t++) s += P[FLOOD * 7 + t];
    out[SUM] = s;
  },
};

const U: Omit<Uniforms, 'step'> = { N: 0, CAP: 0, K: 0, wallsBound: 0 };

function makeCA(topo: Topology, rule = floodRule): CA {
  const cost = new Int32Array(topo.size);
  for (let i = 0; i < topo.size; i++) cost[i] = 1 + (i % 3 === 0 ? 1 : 0);
  return new CA(topo, rule, U, { cost });
}

function expectSame(a: CA, b: CA, label: string): void {
  expect(a.changed, `${label}: changed`).toBe(b.changed);
  expect(a.generation, `${label}: generation`).toBe(b.generation);
  for (let c = 0; c < a.ch.length; c++) {
    const x = a.ch[c], y = b.ch[c];
    // toEqual on every step is slow; it runs only to report a difference.
    let same = x.length === y.length;
    for (let i = 0; same && i < x.length; i++) same = x[i] === y[i];
    if (!same) expect(x, `${label}: channel ${c}`).toEqual(y);
  }
}

/** A random edit stream: a few writes of `src` before some steps. */
function edits(topo: Topology, seed: number, steps: number): Array<Array<[number, number]>> {
  const rand = rng(seed);
  const plan: Array<Array<[number, number]>> = [];
  for (let s = 0; s < steps; s++) {
    const batch: Array<[number, number]> = [];
    if (rand() < 0.3) {
      const n = 1 + Math.floor(rand() * 4);
      for (let k = 0; k < n; k++) {
        // Some writes miss the field: they must be ignored.
        const slot = Math.floor(rand() * topo.size);
        batch.push([slot, rand() < 0.3 ? 0 : Math.floor(rand() * 12)]);
      }
    }
    plan.push(batch);
  }
  return plan;
}

/** A ragged 6-connected field: the hexagon minus a seeded scatter of cells away from the centre. */
function raggedTopo(R: number, seed: number): Topology {
  const b = makeBoard(R);
  const rand = rng(seed);
  const drop = new Set(b.cells.filter(() => rand() < 0.15));
  return hexTopo(R, (c) => !drop.has(c));
}

describe('engine', () => {
  it('fills field slots with init (and constants) and dead slots with dead', () => {
    const topo = hexTopo(4, (c) => c % 5 !== 0);
    const ca = makeCA(topo);
    for (let i = 0; i < topo.size; i++) {
      if (topo.isField[i]) {
        expect(ca.get('cost', i)).toBe(1 + (i % 3 === 0 ? 1 : 0));
        expect(ca.get('src', i)).toBe(0);
      } else {
        expect(ca.get('cost', i)).toBe(0);
      }
    }
    expect(ca.active).toBe(topo.cells.length);
  });

  it('step and stepFull run in lockstep, identical every step', () => {
    for (const [seed, topo] of [
      [1, hexTopo(8)],
      [2, raggedTopo(8, 7)],
      [3, raggedTopo(10, 11)],
    ] as const) {
      const a = makeCA(topo);
      const b = makeCA(topo);
      edits(topo, seed, 300).forEach((batch, s) => {
        for (const [slot, v] of batch) {
          a.write('src', slot, v);
          b.write('src', slot, v);
        }
        a.step();
        b.stepFull();
        expectSame(a, b, `seed ${seed} step ${s}`);
        expect(a.active).toBeLessThanOrEqual(topo.cells.length);
      });
    }
  });

  it('a random tapOrder gives identical results for a symmetric rule', () => {
    const topo = raggedTopo(8, 3);
    const a = makeCA(topo);
    const b = makeCA(topo);
    const rand = rng(99);
    const order = new Int32Array(topo.size * 6);
    for (let i = 0; i < topo.size; i++) {
      const p = [0, 1, 2, 3, 4, 5];
      for (let k = 5; k > 0; k--) {
        const j = Math.floor(rand() * (k + 1));
        [p[k], p[j]] = [p[j], p[k]];
      }
      order.set(p, i * 6);
    }
    b.tapOrder = order;
    edits(topo, 5, 300).forEach((batch, s) => {
      for (const [slot, v] of batch) {
        a.write('src', slot, v);
        b.write('src', slot, v);
      }
      a.step();
      b.step();
      expectSame(a, b, `step ${s}`);
    });
  });

  it('tapOrder is honoured (an asymmetric rule sees the permutation)', () => {
    const topo = hexTopo(3);
    // out = the src of tap 1.
    const rule: Rule = {
      channels: [spec('src', 'input', 0, 0), spec('first', 'out', 0, 0)],
      update(P, out) { out[1] = P[1]; },
    };
    const ca = new CA(topo, rule, U);
    const order = new Int32Array(topo.size * 6);
    for (let i = 0; i < topo.size; i++) order.set([3, 0, 1, 2, 4, 5], i * 6);
    ca.tapOrder = order;
    const centre = topo.cells[Math.floor(topo.cells.length / 2)];
    const w = topo.nbr[centre * 6 + 3];
    ca.write('src', w, 7);
    ca.step();
    expect(ca.get('first', centre)).toBe(7);
  });

  it('settles: changed == 0 and active == 0, and stays put', () => {
    const topo = hexTopo(6);
    const ca = makeCA(topo);
    ca.write('src', topo.cells[10], 9);
    ca.write('src', topo.cells[50], 5);
    const steps = ca.run(1000);
    expect(steps).toBeLessThan(1000);
    expect(ca.changed).toBe(0);
    expect(ca.active).toBe(0);
    const snap = ca.ch.map((a) => a.slice());
    ca.step();
    expect(ca.changed).toBe(0);
    expect(ca.active).toBe(0);
    ca.ch.forEach((a, c) => expect(a).toEqual(snap[c]));
    // A full sweep of a settled board changes nothing either.
    ca.stepFull();
    expect(ca.changed).toBe(0);
    expect(ca.active).toBe(0);
  });

  it('write activates the slot and its field taps, once each', () => {
    const topo = raggedTopo(6, 4);
    const ca = makeCA(topo);
    ca.run();
    expect(ca.active).toBe(0);
    const slot = topo.cells.find((c) => {
      let n = 0;
      for (let k = 0; k < 6; k++) n += topo.isField[topo.nbr[c * 6 + k]];
      return n < 6;
    })!;
    let taps = 0;
    for (let k = 0; k < 6; k++) taps += topo.isField[topo.nbr[slot * 6 + k]];
    ca.write('src', slot, 4);
    expect(ca.active).toBe(1 + taps);
    ca.write('src', slot, 5); // already listed
    expect(ca.active).toBe(1 + taps);
    ca.step();
    expect(ca.changed).toBeGreaterThan(0);
  });

  it('write ignores dead slots and refuses non-input channels', () => {
    const topo = hexTopo(4, (c) => c % 7 !== 0);
    const ca = makeCA(topo);
    ca.run();
    const dead = [...Array(topo.size).keys()].find((i) => !topo.isField[i])!;
    ca.write('src', dead, 9);
    ca.write('src', -1, 9);
    ca.write('src', topo.size, 9);
    expect(ca.get('src', dead)).toBe(0);
    expect(ca.active).toBe(0);
    expect(() => ca.write('flood', topo.cells[0], 1)).toThrow();
    expect(() => ca.write('cost', topo.cells[0], 1)).toThrow();
    expect(() => ca.write('nope', topo.cells[0], 1)).toThrow();
  });

  it('hands the rule step = generation + 1 and never writes dead slots', () => {
    const topo = hexTopo(3, (c) => c % 4 !== 1);
    const rule: Rule = {
      channels: [spec('s', 'out', 0, -5)],
      update(P, out, u) { out[0] = u.step; },
    };
    const ca = new CA(topo, rule, U);
    for (let g = 0; g < 5; g++) {
      ca.step();
      for (const c of topo.cells) expect(ca.get('s', c)).toBe(g + 1);
    }
    for (let i = 0; i < topo.size; i++) if (!topo.isField[i]) expect(ca.get('s', i)).toBe(-5);
  });

  // DESIGN.md §8 item 5: the fill rule itself, replayed off T5's fuzz streams
  // (ragged fields, walls, loops/bridges/scribbles, erases, idle gaps) —
  // three automata in lockstep: the active set (`step`), the reference sweep
  // (`stepFull`), and an active set under a random per-slot `tapOrder`. Every
  // channel at every field slot must agree after every single step, not just
  // at quiet, since a divergence could self-correct by the next edit.
  it.each(seeds(10))('fill rule: step, stepFull and a random tapOrder agree on fuzz stream seed %i', (seed) => {
    const stream = fuzzStream(seed);
    const { field, wallsBound } = stream;
    const label = `seed ${seed} (R ${stream.R}${stream.ragged ? ', ragged' : ''}${wallsBound ? ', wallsBound' : ''})`;

    const a = fillCA(field, { wallsBound }); // active step
    const b = fillCA(field, { wallsBound }); // stepFull: the reference sweep
    const c = fillCA(field, { wallsBound }); // active step, taps shuffled per slot
    const rand = rng(0xca11 + seed * 97);
    const order = new Int32Array(field.topo.size * 6);
    for (const slot of field.topo.cells) {
      const p = [0, 1, 2, 3, 4, 5];
      for (let k = 5; k > 0; k--) {
        const j = Math.floor(rand() * (k + 1));
        [p[k], p[j]] = [p[j], p[k]];
      }
      order.set(p, slot * 6);
    }
    c.tapOrder = order;

    const step = (tag: string) => {
      a.step();
      b.stepFull();
      c.step();
      expectSame(a, b, `${label}: ${tag} (stepFull)`);
      expectSame(a, c, `${label}: ${tag} (tapOrder)`);
    };

    for (const op of stream.ops) {
      if (op.kind === 'step') {
        for (let i = 0; i < op.n; i++) step('op step');
      } else {
        paint(a, op.slots, op.v);
        paint(b, op.slots, op.v);
        paint(c, op.slots, op.v);
      }
    }
    while (a.changed > 0 || a.active > 0) step('settle');

    expect(a.active, `${label}: active == 0 once quiet`).toBe(0);
    expect(b.active, `${label}: active == 0 once quiet`).toBe(0);
    expect(c.active, `${label}: active == 0 once quiet`).toBe(0);
  });
});
