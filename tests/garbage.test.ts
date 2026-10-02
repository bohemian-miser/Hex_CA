import { describe, expect, it } from 'vitest';
import { CA } from '../src/engine.js';
import { Field, hexField } from '../src/field.js';
import { ON, bump, fillCA, paint, scramble, stateOf } from '../src/fill.js';
import { randomWalls, rng } from '../src/lines.js';
import { settleBound } from '../src/oracle.js';
import { channel, eqState, firstDiff, oracleOf, seeds } from './helpers.js';

// DESIGN.md §8 item 6. `scramble` drops in-range garbage into every hidden,
// gate and out channel (the picture itself is untouched). A `bump` re-stamps
// the root with a new epoch over the same picture, so the whole field goes
// fresh and must recompute: that is guaranteed to certify the true oracle
// again, within settleBound of the bump. Without a bump nothing is
// guaranteed, and §8.6 says the test must not expect it: the design refutes
// nothing (§11), so garbage sharing the top epoch can stick (a stale `redge`
// next to a triangle certifies a wrong picture; a bipartite checkerboard
// never goes quiet). Most random garbage happens to recover anyway. So
// below: (a) is a hard per-seed assertion; (b) only claims most seeds
// recover, as a record of how the unbumped case behaves, not a guarantee.

const SEEDS = seeds(20);
/** Far past any real bound at R <= 12: a run that gets here is a bug, not slow luck. */
const MAX_SETTLE = 20_000;

/** Radius 6..12 per seed: small enough to run 20 of these quickly, big enough for real regions. */
const radiusOf = (seed: number): number => 6 + (seed % 7);

/** Steps the CA until nothing is active or changed, up to `max`. Returns steps taken. */
function runToQuiet(ca: CA, max = MAX_SETTLE): number {
  let steps = 0;
  while (ca.changed > 0 || ca.active > 0) {
    ca.step();
    steps++;
    if (steps > max) throw new Error(`not quiet after ${max} steps`);
  }
  return steps;
}

/** A booted, settled automaton with `randomWalls(field.board, rand)` painted ON. */
function withRandomWalls(field: Field, rand: () => number): CA {
  const ca = fillCA(field);
  const walls = [...randomWalls(field.board, rand)].filter((s) => field.topo.isField[s] === 1);
  paint(ca, walls, ON);
  runToQuiet(ca);
  return ca;
}

describe('garbage: scramble then bump recovers within the bound (DESIGN.md §8 item 6)', () => {
  it.each(SEEDS)('seed %i', (seed) => {
    const R = radiusOf(seed);
    const field = hexField(R, { idSeed: seed + 1 });
    const label = `seed ${seed} (R ${R})`;
    const ca = withRandomWalls(field, rng(5000 + seed));

    scramble(ca, rng(6000 + seed));
    const editGen = ca.generation; // bump's stamp will be editGen + 1
    bump(ca);
    let lastChangeGen = editGen;
    let steps = 0;
    while (ca.changed > 0 || ca.active > 0) {
      ca.step();
      steps++;
      if (ca.changed > 0) lastChangeGen = ca.generation;
      if (steps > MAX_SETTLE) throw new Error(`${label}: scramble + bump never settled`);
    }

    const p = channel(ca, 'paint');
    const bound = settleBound(field, (slot) => p[slot], [field.root]);
    expect(lastChangeGen - editGen, `${label}: steps from the bump to the last change`).toBeLessThanOrEqual(bound);

    const expected = oracleOf(ca, field);
    const diff = firstDiff(stateOf(ca), expected);
    if (diff >= 0) expect(stateOf(ca), `${label}: state == oracle (first difference at slot ${diff})`).toEqual(expected);
  });
});

describe('garbage: scramble without a bump — not guaranteed, usually recovers anyway (DESIGN.md §8 item 6)', () => {
  let recovered = 0;
  it.each(SEEDS)('seed %i', (seed) => {
    const R = radiusOf(seed);
    const field = hexField(R, { idSeed: seed + 1 });
    // Same walls and the same garbage seed as the bumped case above, so the
    // only thing that differs between the two tests is the bump itself.
    const ca = withRandomWalls(field, rng(5000 + seed));
    scramble(ca, rng(6000 + seed));

    const p = channel(ca, 'paint');
    const bound = settleBound(field, (slot) => p[slot], [field.root]);
    const cap = 3 * field.N + bound;
    let steps = 0;
    while ((ca.changed > 0 || ca.active > 0) && steps < cap) {
      ca.step();
      steps++;
    }
    const quiet = ca.changed === 0 && ca.active === 0;
    if (quiet && eqState(stateOf(ca), oracleOf(ca, field))) recovered++;
  });

  it('recovers for most of the 20 seeds within 3N + settleBound (not all — DESIGN.md §8 item 6 says this is not guaranteed)', () => {
    expect(recovered).toBeGreaterThanOrEqual(Math.floor(SEEDS.length * 0.7));
  });
});
