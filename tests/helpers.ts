// Shared by the property tests (fuzz, overshoot, and the engine's lockstep
// twins): channel access, pictures, and the seeded edit streams of
// DESIGN.md §8 item 3. Nothing here checks anything; the tests do.

import { CA } from '../src/engine.js';
import { Field, hexField, presetWalls } from '../src/field.js';
import { FILLED, OFF, ON, WALL } from '../src/fill.js';
import { coordsOf, hexDistance } from '../src/hex.js';
import { randomLine, randomLoop, randomScribble, rng } from '../src/lines.js';
import { oracle } from '../src/oracle.js';

/** Seeds per property test: FUZZ_SEEDS from the environment, 40 by default (the Pi's one-minute budget). */
export const FUZZ_SEEDS = positiveInt(process.env.FUZZ_SEEDS, 40);

function positiveInt(s: string | undefined, fallback: number): number {
  const n = Number(s);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}

/** 0, 1, …, n - 1: the seeds an `it.each` runs over. */
export function seeds(n = FUZZ_SEEDS): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** A channel's values by name (ch[c][slot]), so a per-step check reads the array directly. */
export function channel(ca: CA, name: string): Int32Array {
  const c = ca.rule.channels.findIndex((s) => s.name === name);
  if (c < 0) throw new Error(`no channel ${name}`);
  return ca.ch[c];
}

/** The picture: paint per slot (a copy; WALL at dead slots). */
export function pictureOf(ca: CA): Uint8Array {
  return Uint8Array.from(channel(ca, 'paint'));
}

/** The oracle (DESIGN.md §4) of the automaton's current picture. */
export function oracleOf(ca: CA, field: Field, wallsBound = false): Uint8Array {
  const p = channel(ca, 'paint');
  return oracle(field, (slot) => p[slot], wallsBound);
}

/** Whether two state (or picture) arrays are identical. */
export function eqState(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  return firstDiff(a, b) < 0;
}

/** The first index where two arrays differ (a length mismatch counts at the shorter length), or -1. */
export function firstDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

/** How many entries equal `v`. */
export function count(a: ArrayLike<number>, v: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === v) n++;
  return n;
}

/** One op of an edit stream: a `paint` call (one batch, one serial) or `n` steps. */
export type Op =
  | { kind: 'paint'; slots: number[]; v: 0 | 1 | 3 }
  | { kind: 'step'; n: number };

export interface Stream {
  seed: number;
  R: number;
  field: Field;
  wallsBound: boolean;
  /** Whether the stream opens with the 'blob' walls (a ragged field). */
  ragged: boolean;
  ops: Op[];
}

/** Share of seeds that open with the 'blob' rim (DESIGN.md §8 item 3). */
const RAGGED_SHARE = 0.4;
/** Share of streams run with every wall exterior; their oracle differs, so they get their own seeds. */
const WALLS_BOUND_SHARE = 0.15;
/** Share of shapes followed by erasing one of their cells. */
const ERASE_SHARE = 0.3;
/** Share of gaps long enough (3R..12R steps) that a commit lands mid-stream. */
const LONG_GAP_SHARE = 0.3;

/**
 * A seeded edit stream (DESIGN.md §8 item 3) on a hexagon of radius 4..15:
 * maybe the 'blob' walls (the preset's own island included) plus a second
 * wall island, then 2–5 random loops, bridges and scribbles, each drawn in
 * two halves with an idle gap between, cells 0–3 per paint call with a 50%
 * chance of 1–3 steps after each call; 30% of shapes then lose one cell to
 * an erase, and wall specks (and wall erases) are sprinkled in. Shapes skip
 * cells that are not OFF when drawn. Pure: the same seed gives the same ops,
 * so lockstep twins can replay it. The stream may end with steps or a paint.
 */
export function fuzzStream(seed: number): Stream {
  const rand = rng(0x5eed + seed * 7919);
  const R = 4 + Math.floor(rand() * 12);
  const field = hexField(R, { idSeed: seed + 1 });
  const { board, topo } = field;
  const cells = topo.cells;
  const ragged = rand() < RAGGED_SHARE;
  const wallsBound = rand() < WALLS_BOUND_SHARE;
  const ops: Op[] = [];
  // The picture the ops leave, so shapes can skip what is not OFF.
  const pic = new Uint8Array(board.size).fill(WALL);
  for (const s of cells) pic[s] = OFF;

  const pick = <T>(xs: ArrayLike<T>): T => xs[Math.floor(rand() * xs.length)];
  const steps = (n: number) => {
    if (n > 0) ops.push({ kind: 'step', n });
  };
  const put = (slots: number[], v: 0 | 1 | 3) => {
    ops.push({ kind: 'paint', slots, v });
    for (const s of slots) pic[s] = v;
  };
  // Most gaps are shorter than a settle; some let a commit land.
  const gap = () =>
    rand() < LONG_GAP_SHARE ? 3 * R + Math.floor(rand() * 9 * R) : Math.floor(rand() * (3 * R + 1));
  // A run of cells as a stroke: 0–3 cells per call, a 50% chance of 1–3 steps after each.
  const stroke = (run: number[], v: 0 | 1 | 3) => {
    for (let i = 0; i < run.length; ) {
      const n = Math.floor(rand() * 4);
      put(run.slice(i, i + n), v);
      i += n;
      if (rand() < 0.5) steps(1 + Math.floor(rand() * 3));
    }
  };

  if (ragged) {
    const walls = presetWalls(field, 'blob', 1 + Math.floor(rand() * 1000));
    // A second island: a small wall disc somewhere inside the rim.
    const [cq, cr] = [Math.floor((rand() * 2 - 1) * (R / 2)), Math.floor((rand() * 2 - 1) * (R / 2))];
    const size = Math.floor(rand() * 2);
    for (const s of cells) {
      const [q, r] = coordsOf(board, s);
      if (hexDistance(q - cq, r - cr) <= size && hexDistance(q, r) <= R - 3) walls.push(s);
    }
    put(walls, WALL);
  }
  // Half the streams start editing while the boot is still settling.
  steps(rand() < 0.5 ? 0 : Math.floor(rand() * 12 * R));

  const shapes = 2 + Math.floor(rand() * 4);
  for (let k = 0; k < shapes; k++) {
    const kind = rand();
    const drawn =
      kind < 0.4 ? randomLoop(board, rand, { wiggle: rand() * 2 })
      : kind < 0.7 ? randomLine(board, rand, { wiggle: rand() * 2 })
      : randomScribble(board, rand, { maxLength: 4 + Math.floor(rand() * R * 5) });
    const shape = (drawn ?? []).filter((s) => topo.isField[s] === 1 && pic[s] === OFF);
    if (shape.length > 0) {
      const half = 1 + Math.floor(rand() * shape.length);
      stroke(shape.slice(0, half), ON);
      steps(gap());
      stroke(shape.slice(half), ON);
      if (rand() < ERASE_SHARE) {
        steps(Math.floor(rand() * (R + 1)));
        const lit = shape.filter((s) => pic[s] === ON);
        if (lit.length > 0) put([pick(lit)], OFF);
      }
    }
    // Walls in the stream: specks anywhere, and now and then a wall erased.
    if (rand() < 0.3) {
      const specks = 1 + Math.floor(rand() * 3);
      for (let i = 0; i < specks; i++) {
        put([pick(cells)], WALL);
        if (rand() < 0.5) steps(1);
      }
    }
    if (rand() < 0.2) {
      const walls = Array.from(cells).filter((s) => pic[s] === WALL);
      if (walls.length > 0) put([pick(walls)], OFF);
    }
    steps(gap());
  }
  return { seed, R, field, wallsBound, ragged, ops };
}

/** The field slots of `slots` whose paint `v` would change: what a paint call edits. */
export function edits(ca: CA, slots: readonly number[], v: number): number[] {
  const p = channel(ca, 'paint');
  const { isField, size } = ca.topo;
  const out = new Set<number>();
  for (const s of slots) if (s >= 0 && s < size && isField[s] === 1 && p[s] !== v) out.add(s);
  return [...out];
}

/**
 * The latch invariant of DESIGN.md §5, checked after every step: a field
 * cell painted OFF whose commit equals its epoch shows FILLED exactly where
 * the oracle of that serial's picture does, and a FILLED cell anywhere is
 * FILLED in the picture it last latched (no wrong fill is ever shown). Call
 * `record` after every paint call: the picture for serial generation + 1 is
 * the one after every write made before that step, so a later call before
 * the same step overwrites it. Serial 0 is the boot picture.
 */
export class LatchCheck {
  /** serial → the oracle of that serial's picture. */
  readonly pictures = new Map<number, Uint8Array>();
  /** Per slot: the serial its state last latched (commit == epoch), -1 before any. */
  private readonly latched: Int32Array;
  private readonly paintCh: Int32Array;
  private readonly epoch: Int32Array;
  private readonly commit: Int32Array;
  private readonly state: Int32Array;

  constructor(readonly ca: CA, readonly field: Field, readonly wallsBound = false) {
    this.latched = new Int32Array(ca.topo.size).fill(-1);
    this.paintCh = channel(ca, 'paint');
    this.epoch = channel(ca, 'epoch');
    this.commit = channel(ca, 'commit');
    this.state = channel(ca, 'state');
    this.pictures.set(0, oracleOf(ca, field, wallsBound));
  }

  /** Note the current picture under the serial the next step absorbs. */
  record(): void {
    this.pictures.set(this.ca.generation + 1, oracleOf(this.ca, this.field, this.wallsBound));
  }

  /** Throws, naming the slot, if the state just stepped breaks the invariant. */
  check(label: string): void {
    const { paintCh, epoch, commit, state, latched, pictures } = this;
    const g = this.ca.generation;
    for (const s of this.field.topo.cells) {
      const filled = state[s] === FILLED;
      if (filled && paintCh[s] !== OFF) throw new Error(`${label}: step ${g}: slot ${s} is FILLED but painted ${paintCh[s]}`);
      if (commit[s] === epoch[s]) {
        latched[s] = commit[s];
        if (paintCh[s] !== OFF) continue;
        const certified = pictures.get(commit[s]);
        if (!certified) throw new Error(`${label}: step ${g}: slot ${s} certified for unknown serial ${commit[s]}`);
        if (filled !== (certified[s] === FILLED)) {
          throw new Error(`${label}: step ${g}: slot ${s} certified for serial ${commit[s]} shows ${filled ? 'FILLED' : 'not FILLED'}; its oracle says otherwise`);
        }
      } else if (filled) {
        const was = pictures.get(latched[s]);
        if (!was || was[s] !== FILLED) {
          throw new Error(`${label}: step ${g}: slot ${s} shows FILLED, which its last latched picture (serial ${latched[s]}) does not`);
        }
      }
    }
  }
}
