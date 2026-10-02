// Benchmark (DESIGN.md §7, not a test): settle steps, ms and µs per update
// at R = 20, 40, 80. Run with `npm run bench`.
//
// Per radius: the boot of an empty field to quiet; a random loop and a random
// bridge, each painted all but its last cell on a quiet board and then
// closed — steps until the root commits the closing edit (beside the
// picture's settleBound), steps to quiet, wall time and µs per cell computed;
// and 1000 steps of a settled board, which should cost nothing.

import { CA } from '../src/engine.js';
import { Field, hexField } from '../src/field.js';
import { ON, fillCA, paint } from '../src/fill.js';
import { randomLine, randomLoop, rng } from '../src/lines.js';
import { settleBound } from '../src/oracle.js';

/** Far past any bound at these radii. */
const MAX_STEPS = 100_000;

/** A booted automaton, run to quiet. */
function booted(field: Field): CA {
  const ca = fillCA(field);
  ca.run();
  return ca;
}

/** Steps to the root's commit of the serial, steps to quiet, ms, and µs per cell computed. */
function settle(ca: CA, field: Field, serial: number): { commit: number; quiet: number; ms: number; usPerUpdate: number } {
  const t0 = performance.now();
  const from = ca.generation;
  let commit = 0;
  let updates = 0;
  do {
    updates += ca.active;
    ca.step();
    if (!commit && ca.get('commit', field.root) === serial) commit = ca.generation - from;
  } while (ca.changed > 0 && ca.generation - from < MAX_STEPS);
  const ms = performance.now() - t0;
  return { commit, quiet: ca.generation - from, ms, usPerUpdate: updates ? (ms * 1000) / updates : 0 };
}

/** Paint `shape` all but its last cell, settle, then close it: the closing edit's settle, and its bound. */
function close(field: Field, shape: number[], label: string): void {
  const ca = booted(field);
  paint(ca, shape.slice(0, -1), ON);
  ca.run();
  const last = shape[shape.length - 1];
  const serial = ca.generation + 1;
  paint(ca, [last], ON);
  const p = ca.ch[ca.rule.channels.findIndex((s) => s.name === 'paint')];
  const bound = settleBound(field, (slot) => p[slot], [last]);
  const r = settle(ca, field, serial);
  console.log(
    `${label}: commit ${r.commit} (bound ${bound}), quiet ${r.quiet}, ${r.ms.toFixed(1)} ms, ${r.usPerUpdate.toFixed(2)} µs/update`,
  );
}

for (const R of [20, 40, 80]) {
  const field = hexField(R);
  console.log(`\n=== R = ${R}: N = ${field.N} ===`);

  {
    const ca = fillCA(field);
    const t0 = performance.now();
    const steps = ca.run();
    console.log(`boot: ${steps} steps, ${(performance.now() - t0).toFixed(1)} ms`);
  }

  const rand = rng(3 * R + 1);
  const loop = randomLoop(field.board, rand);
  if (loop) close(field, loop, 'loop');
  else console.log('loop: none found');

  const bridge = randomLine(field.board, rand, { blocked: new Set(loop ?? []) });
  if (bridge) close(field, bridge, 'bridge');
  else console.log('bridge: none found');

  {
    const ca = booted(field);
    const t0 = performance.now();
    for (let i = 0; i < 1000; i++) ca.step();
    console.log(`settled, 1000 steps: ${(performance.now() - t0).toFixed(3)} ms`);
  }
}
