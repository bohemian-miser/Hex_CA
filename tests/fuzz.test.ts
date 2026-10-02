import { describe, expect, it } from 'vitest';
import { fillCA, paint, stateOf } from '../src/fill.js';
import { settleBound } from '../src/oracle.js';
import { LatchCheck, channel, edits, firstDiff, fuzzStream, oracleOf, seeds } from './helpers.js';

// DESIGN.md §8 item 3: seeded edit streams (ragged rims, islands, loops,
// bridges and scribbles drawn in two halves, erases, wall specks) played
// into the fill automaton. After every step the latch invariant holds; at
// quiet the state is the oracle's, every cell committed the last edit's
// serial, and the last change came within settleBound of that edit.

/** Far past any bound at R ≤ 15: a run that gets here never settles. */
const MAX_SETTLE = 20_000;

describe('fuzz: edit streams keep the latch invariant and settle to the oracle', () => {
  it.each(seeds())('seed %i', (seed) => {
    const stream = fuzzStream(seed);
    const { field, wallsBound } = stream;
    const label = `seed ${seed} (R ${stream.R}${stream.ragged ? ', ragged' : ''}${wallsBound ? ', wallsBound' : ''})`;
    const ca = fillCA(field, { wallsBound });
    const latch = new LatchCheck(ca, field, wallsBound);

    // The last paint call that changed something: its generation and cells.
    // Before any, the boot is the edit (everything is fresh at step 1).
    let lastEditGen = 0;
    let lastSerial = 0;
    let lastEdited = [field.root];
    let lastChangeGen = 0;
    const step = () => {
      ca.step();
      if (ca.changed > 0) lastChangeGen = ca.generation;
      latch.check(label);
    };

    for (const op of stream.ops) {
      if (op.kind === 'step') {
        for (let i = 0; i < op.n; i++) step();
        continue;
      }
      const changed = edits(ca, op.slots, op.v);
      paint(ca, op.slots, op.v);
      latch.record();
      if (changed.length > 0) {
        lastEditGen = ca.generation;
        lastSerial = ca.generation + 1;
        lastEdited = changed;
      }
    }
    while (ca.changed > 0 || ca.active > 0) {
      step();
      if (ca.generation - lastEditGen > MAX_SETTLE) throw new Error(`${label}: not quiet ${MAX_SETTLE} steps after the last edit`);
    }

    const expected = oracleOf(ca, field, wallsBound);
    const diff = firstDiff(stateOf(ca), expected);
    if (diff >= 0) expect(stateOf(ca), `${label}: state == oracle (first difference at slot ${diff})`).toEqual(expected);
    const serial = lastSerial;
    const epoch = channel(ca, 'epoch');
    const commit = channel(ca, 'commit');
    for (const s of field.topo.cells) {
      if (epoch[s] !== serial || commit[s] !== serial) {
        throw new Error(`${label}: slot ${s} at quiet has epoch ${epoch[s]}, commit ${commit[s]}; expected ${serial}`);
      }
    }
    const p = channel(ca, 'paint');
    const bound = settleBound(field, (slot) => p[slot], lastEdited);
    expect(lastChangeGen - lastEditGen, `${label}: steps from the last edit to its last change`).toBeLessThanOrEqual(bound);
  });
});
