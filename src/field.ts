// A field: the slots a CA runs over, plus the host-constant static tree and
// ids the fill rule's leader flood and gate need (DESIGN.md §1, §6).
//
// Built once per board/mask and handed to the CA as `constants` (field, id,
// gpar, area); everything here is plain host-side bookkeeping, never part of
// a step.

import { Board, coordsOf, hexDistance, makeBoard } from './hex.js';
import { Topology } from './engine.js';
import { rng } from './lines.js';

export interface Field {
  board: Board;
  topo: Topology;
  /** Number of field slots. */
  N: number;
  /** N + 1: one past any id, the "unknown"/"infinite" sentinel. */
  CAP: number;
  /** The root's eccentricity (R on a full hexagon). */
  K: number;
  /** The field slot nearest the centre (lowest slot index on ties). */
  root: number;
  /** ids[slot]: a seeded permutation of 1..N over field slots, 0 elsewhere. */
  ids: Int32Array;
  /** gpar[slot]: the id of the static parent one step closer to root, 0 at the root. */
  gpar: Int32Array;
}

interface FieldOptions {
  idSeed?: number;
}

/** The default hexagon of radius R: every cell within hex distance R of the centre. */
export function hexField(R: number, opts: FieldOptions = {}): Field {
  const board = makeBoard(R);
  const mask = new Uint8Array(board.size).fill(1);
  return buildField(board, mask, opts);
}

/** Any 6-connected subset of `board`'s inside cells. Throws if it isn't 6-connected. */
export function maskedField(board: Board, mask: Uint8Array, opts: FieldOptions = {}): Field {
  return buildField(board, mask, opts);
}

function buildField(board: Board, mask: Uint8Array, opts: FieldOptions): Field {
  const size = board.size;
  const isField = new Uint8Array(size);
  const cells: number[] = [];
  // board.cells is already every inside slot in slot order, so filtering it
  // keeps that order and is exactly `board.inside && mask`.
  for (const i of board.cells) {
    if (mask[i]) {
      isField[i] = 1;
      cells.push(i);
    }
  }
  const N = cells.length;
  if (N === 0) throw new Error('field has no cells');

  // 6-connectivity, asserted: BFS from one field slot must reach them all.
  const seenFrom = new Int32Array(size).fill(-1);
  const bfsQueue = [cells[0]];
  seenFrom[cells[0]] = 0;
  for (let qi = 0; qi < bfsQueue.length; qi++) {
    const slot = bfsQueue[qi];
    for (let k = 0; k < 6; k++) {
      const j = board.neighbours[slot * 6 + k];
      if (j >= 0 && isField[j] && seenFrom[j] < 0) {
        seenFrom[j] = 0;
        bfsQueue.push(j);
      }
    }
  }
  if (bfsQueue.length !== N) throw new Error('field is not 6-connected');

  // root: smallest hex distance to the centre, lowest slot index on ties.
  let root = cells[0];
  let bestD = Infinity;
  for (const i of cells) {
    const [q, r] = coordsOf(board, i);
    const d = hexDistance(q, r);
    if (d < bestD || (d === bestD && i < root)) {
      bestD = d;
      root = i;
    }
  }

  // BFS from the root over field slots: distance and K, the static tree's depth.
  const rdist = new Int32Array(size).fill(-1);
  const rootQueue = [root];
  rdist[root] = 0;
  let K = 0;
  for (let qi = 0; qi < rootQueue.length; qi++) {
    const slot = rootQueue[qi];
    if (rdist[slot] > K) K = rdist[slot];
    for (let k = 0; k < 6; k++) {
      const j = board.neighbours[slot * 6 + k];
      if (j >= 0 && isField[j] && rdist[j] < 0) {
        rdist[j] = rdist[slot] + 1;
        rootQueue.push(j);
      }
    }
  }

  const CAP = N + 1;

  // ids: a seeded permutation of 1..N over field slots (random ids cut leader churn).
  const ids = new Int32Array(size);
  const rand = rng(opts.idSeed ?? 7);
  const perm = Array.from({ length: N }, (_, i) => i + 1);
  for (let k = N - 1; k > 0; k--) {
    const j = Math.floor(rand() * (k + 1));
    [perm[k], perm[j]] = [perm[j], perm[k]];
  }
  cells.forEach((slot, i) => {
    ids[slot] = perm[i];
  });

  // gpar: the id of the highest-id neighbour one step closer to the root.
  const gpar = new Int32Array(size);
  for (const slot of cells) {
    if (slot === root) continue; // stays 0
    let best = 0;
    for (let k = 0; k < 6; k++) {
      const j = board.neighbours[slot * 6 + k];
      if (j >= 0 && isField[j] && rdist[j] === rdist[slot] - 1 && ids[j] > best) best = ids[j];
    }
    gpar[slot] = best;
  }

  const topo: Topology = { size, cells: Int32Array.from(cells), nbr: board.neighbours, isField };
  return { board, topo, N, CAP, K, root, ids, gpar };
}

/**
 * Wall pictures on a full hexagon, as the slots to paint WALL.
 *
 * 'hexagon': none (a clean field). 'blob': a wobbly rim a cell or two thick,
 * plus one random interior wall cell as an island. 'lobes': two discs joined
 * by a neck, walls everywhere else. 'ring': a rim two cells thick plus a
 * central wall disc. Every rim touches the dead ring, so it is exterior (the
 * field inside it is ragged, not enclosed): a ring one cell in would be an
 * island, and the whole interior would fill before anything was drawn.
 */
export function presetWalls(field: Field, preset: 'hexagon' | 'blob' | 'lobes' | 'ring', seed = 1): number[] {
  const { board, topo } = field;
  const R = board.radius;
  const out: number[] = [];

  if (preset === 'hexagon') return out;

  if (preset === 'blob') {
    const rim = new Set<number>();
    for (const slot of topo.cells) {
      const [q, r] = coordsOf(board, slot);
      const threshold = R - 1 - Math.floor(2 * Math.abs(Math.sin(0.7 * q + 1.3 * r + seed)));
      if (hexDistance(q, r) >= threshold) rim.add(slot);
    }
    out.push(...rim);
    const interior = topo.cells.filter((slot) => !rim.has(slot));
    if (interior.length > 0) {
      const rand = rng(seed * 104729 + 1);
      out.push(interior[Math.floor(rand() * interior.length)]);
    }
    return out;
  }

  if (preset === 'lobes') {
    const cx = Math.round(0.5 * R);
    const radius = Math.max(1, Math.round(0.45 * R));
    for (const slot of topo.cells) {
      const [q, r] = coordsOf(board, slot);
      const inDisc1 = hexDistance(q + cx, r) <= radius;
      const inDisc2 = hexDistance(q - cx, r) <= radius;
      const inNeck = Math.abs(r) <= 1 && q >= -cx && q <= cx;
      if (!inDisc1 && !inDisc2 && !inNeck) out.push(slot);
    }
    return out;
  }

  // 'ring': the outer two rings plus a central wall disc.
  const inner = Math.max(0, Math.floor(R / 5));
  for (const slot of topo.cells) {
    const [q, r] = coordsOf(board, slot);
    const d = hexDistance(q, r);
    if (d >= R - 1 || d <= inner) out.push(slot);
  }
  return out;
}
