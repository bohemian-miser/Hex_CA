import { describe, expect, it } from 'vitest';
import { CA, Topology } from '../src/engine.js';
import { Field, hexField } from '../src/field.js';
import { ON, fillCA, paint, stateOf } from '../src/fill.js';
import { Board, coordsOf, indexOf } from '../src/hex.js';
import { randomLoop, randomWalls, rng } from '../src/lines.js';
import { firstDiff, seeds } from './helpers.js';

// DESIGN.md §8 item 7. The fill rule reads nothing but the hex grid and the
// picture (§4's oracle has no notion of direction), and a full hexField is
// itself symmetric under a 60-degree turn about its centre — hexDistance(q,
// r) = max(|q|,|r|,|q+r|) is invariant under (q, r) -> (-r, q + r), since
// that just permutes the three terms' signs. So painting a rotated picture
// on its own booted automaton must settle to the rotation of the original's
// state. `id`/`gpar`/`leader`/etc. are NOT compared: ids are a seeded
// permutation over slot order, not over geometry, so they differ completely
// between the two fields even though the drawn `state` must agree.

const SEEDS = seeds(20);
const MAX_SETTLE = 20_000;

/** Radius 5..11: big enough for randomLoop (wants >= 4), small enough for six automata per seed. */
const radiusOf = (seed: number): number => 5 + (seed % 7);

/** One 60-degree turn about the centre, applied `k` times (DESIGN.md §8 item 7). */
function rotateCoord(q: number, r: number, k: number): [number, number] {
  let cq = q, cr = r;
  for (let i = 0; i < k; i++) [cq, cr] = [-cr, cq + cr];
  return [cq, cr];
}

/** `slots` (field cells) carried to where they land after `k` rotations. */
function rotateSlots(board: Board, slots: readonly number[], k: number): number[] {
  return slots.map((s) => {
    const [q, r] = coordsOf(board, s);
    const [rq, rr] = rotateCoord(q, r, k);
    return indexOf(board, rq, rr);
  });
}

/** `state`, carried to where each of its field cells lands after `k` rotations (dead slots untouched). */
function rotateState(board: Board, topo: Topology, state: Uint8Array, k: number): Uint8Array {
  const out = Uint8Array.from(state); // dead slots keep their own WALL; every field slot gets overwritten below
  for (const s of topo.cells) {
    const [q, r] = coordsOf(board, s);
    const [rq, rr] = rotateCoord(q, r, k);
    out[indexOf(board, rq, rr)] = state[s];
  }
  return out;
}

function runToQuiet(ca: CA, max = MAX_SETTLE): void {
  let steps = 0;
  while (ca.changed > 0 || ca.active > 0) {
    ca.step();
    steps++;
    if (steps > max) throw new Error(`not quiet after ${max} steps`);
  }
}

/** A random picture: a few ON walls/shapes plus a closed loop, so some region is sure to fill. */
function randomPicture(field: Field, rand: () => number): number[] {
  const { board, topo } = field;
  const slots = new Set<number>();
  for (const s of randomWalls(board, rand)) if (topo.isField[s] === 1) slots.add(s);
  const loop = randomLoop(board, rand);
  if (loop) for (const s of loop) if (topo.isField[s] === 1) slots.add(s);
  return [...slots];
}

describe('rotate: a full hexagon field fills the same way under all six 60-degree turns', () => {
  it.each(SEEDS)('seed %i', (seed) => {
    const R = radiusOf(seed);
    const field = hexField(R, { idSeed: seed + 1 });
    const { board, topo } = field;
    const label = `seed ${seed} (R ${R})`;
    const onSlots = randomPicture(field, rng(7000 + seed));

    // The baseline: the picture as drawn (k = 0).
    const base = fillCA(field);
    paint(base, onSlots, ON);
    runToQuiet(base);
    const baseState = stateOf(base);

    for (let k = 1; k <= 5; k++) {
      const rotated = fillCA(field); // its own booted automaton, as the spec asks
      paint(rotated, rotateSlots(board, onSlots, k), ON);
      runToQuiet(rotated);

      const expected = rotateState(board, topo, baseState, k);
      const actual = stateOf(rotated);
      const diff = firstDiff(actual, expected);
      if (diff >= 0) {
        expect(actual, `${label}: rotating ${k * 60} degrees (first difference at slot ${diff})`).toEqual(expected);
      }
    }
  });
});
