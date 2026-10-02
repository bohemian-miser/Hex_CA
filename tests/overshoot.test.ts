import { describe, expect, it } from 'vitest';
import { CA } from '../src/engine.js';
import { Field, hexField } from '../src/field.js';
import { FILLED, OFF, ON, fillCA, paint, stateOf } from '../src/fill.js';
import { randomLine, randomLoop, rng } from '../src/lines.js';
import { settleBound } from '../src/oracle.js';
import { LatchCheck, channel, count, firstDiff, oracleOf, seeds } from './helpers.js';

// DESIGN.md §8 item 4. One edit on a settled board never shows a fill that
// neither the old picture nor the new one has, and no cell's fill flips
// more than once on the way. And the serial contract: a stroke painted a
// cell every two steps, each cell stamped with the step that absorbs it,
// settles within the bound counted from its last cell.

/** Seeds for the overshoot and stroke tests. */
const SEEDS = seeds(20);
/** Far past any bound at R ≤ 12. */
const MAX_SETTLE = 20_000;

/** Radius 6..12 per seed: small enough for per-step checks, big enough for real regions. */
const radiusOf = (seed: number) => 6 + (seed % 7);

/** A random loop (even seeds) or bridge (odd seeds) on the field's board, in drawing order. */
function shapeFor(field: Field, seed: number): number[] {
  const rand = rng(1000 + seed);
  for (let attempt = 0; attempt < 20; attempt++) {
    const shape = seed % 2 === 0 ? randomLoop(field.board, rand) : randomLine(field.board, rand);
    if (shape && shape.length >= 4) return shape;
  }
  throw new Error(`seed ${seed}: no shape`);
}

/** A booted automaton, run to quiet (latch-checked). */
function booted(field: Field): { ca: CA; latch: LatchCheck } {
  const ca = fillCA(field);
  const latch = new LatchCheck(ca, field);
  settle(ca, latch, 'boot');
  return { ca, latch };
}

/** Steps to quiet under the latch check; returns the number of steps that changed something. */
function settle(ca: CA, latch: LatchCheck, label: string, each?: (step: number) => void): number {
  let steps = 0;
  let last = 0;
  while (ca.changed > 0 || ca.active > 0) {
    ca.step();
    steps++;
    if (ca.changed > 0) last = steps;
    latch.check(label);
    each?.(steps);
    if (steps > MAX_SETTLE) throw new Error(`${label}: not quiet after ${MAX_SETTLE} steps`);
  }
  return last;
}

function expectOracle(ca: CA, field: Field, label: string): void {
  const expected = oracleOf(ca, field);
  const diff = firstDiff(stateOf(ca), expected);
  if (diff >= 0) expect(stateOf(ca), `${label}: state == oracle (first difference at slot ${diff})`).toEqual(expected);
}

/**
 * Paint one edit on a quiet board and settle it, asserting at every step
 * FILLED ⊆ before ∪ after and at most one flip of each cell's FILLED bit.
 * Returns how many cells the new picture fills.
 */
function overshoot(ca: CA, latch: LatchCheck, field: Field, slots: number[], v: 0 | 1, label: string): number {
  const before = stateOf(ca);
  paint(ca, slots, v);
  latch.record();
  const after = oracleOf(ca, field);
  const state = channel(ca, 'state');
  const flips = new Uint8Array(state.length);
  const shown = Uint8Array.from(before, (s) => (s === FILLED ? 1 : 0));
  const p = channel(ca, 'paint');
  const bound = settleBound(field, (s) => p[s], slots);
  const last = settle(ca, latch, label, (step) => {
    for (const s of field.topo.cells) {
      const filled = state[s] === FILLED ? 1 : 0;
      if (filled && before[s] !== FILLED && after[s] !== FILLED) {
        throw new Error(`${label}: step ${step}: slot ${s} shows FILLED, which neither the old state nor the new oracle has`);
      }
      if (filled !== shown[s]) {
        shown[s] = filled;
        if (++flips[s] > 1) throw new Error(`${label}: step ${step}: slot ${s} flipped its fill a second time`);
      }
    }
  });
  expectOracle(ca, field, label);
  expect(last, `${label}: settled within the bound`).toBeLessThanOrEqual(bound);
  return count(after, FILLED);
}

describe('overshoot: one edit on a settled board', () => {
  let fills = 0;
  it.each(SEEDS)('seed %i: the closing cell fills at most once, never outside before ∪ after; erasing it un-fills likewise', (seed) => {
    const field = hexField(radiusOf(seed), { idSeed: seed + 1 });
    const shape = shapeFor(field, seed);
    const label = `seed ${seed} (R ${field.board.radius}, ${seed % 2 === 0 ? 'loop' : 'bridge'})`;
    const { ca, latch } = booted(field);
    const closing = shape[shape.length - 1];
    paint(ca, shape.slice(0, -1), ON);
    latch.record();
    settle(ca, latch, `${label}: open shape`);
    expectOracle(ca, field, `${label}: open shape`);
    const filled = overshoot(ca, latch, field, [closing], ON, `${label}: close`);
    if (filled > 0) fills++;
    overshoot(ca, latch, field, [closing], OFF, `${label}: reopen`);
  });

  it('most of those closing cells fill something', () => {
    // A bridge may tie; loops always fill. Guards against shapes that never close.
    expect(fills).toBeGreaterThanOrEqual(SEEDS.length / 2);
  });
});

describe('serial contract: a stroke painted a cell every two steps', () => {
  it.each(SEEDS)('seed %i: settles within the bound after its last cell', (seed) => {
    const field = hexField(radiusOf(seed), { idSeed: seed + 1 });
    const loop = shapeFor(field, seed * 2); // always a loop: it fills when the last cell lands
    const label = `seed ${seed} (R ${field.board.radius})`;
    const { ca, latch } = booted(field);
    for (const cell of loop) {
      paint(ca, [cell], ON); // each its own call: stamp = generation + 1, a new serial per cell
      latch.record();
      for (let i = 0; i < 2; i++) {
        ca.step();
        latch.check(label);
      }
    }
    // Counted from the last cell's paint: the two steps already taken, then the rest.
    const last = channel(ca, 'paint');
    const bound = settleBound(field, (s) => last[s], [loop[loop.length - 1]]);
    const quiet = settle(ca, latch, label);
    expect(2 + quiet, `${label}: steps from the last cell to its last change`).toBeLessThanOrEqual(bound);
    expectOracle(ca, field, label);
    expect(count(stateOf(ca), FILLED)).toBeGreaterThan(0);
  });
});
