// The reference predicate (DESIGN.md §4) and the settle bound (§5), computed
// directly over a Field — no automaton involved. This is the ground truth the
// CA (src/fill.ts) is checked against, so it is written straight from the
// spec rather than shared code with the rule.

import { Field } from './field.js';
import { Topology } from './engine.js';

// Mirrors the picture values src/fill.ts exports (OFF/ON/WALL); not exported
// here since this module only consumes a picture, never defines one. FILLED
// is a *state*, output only.
const OFF = 0;
const ON = 1;
const FILLED = 2;
const WALL = 3;

/**
 * The oracle state per slot (0 off, 1 on, 2 filled, 3 wall; 3 at dead slots),
 * computed exactly as DESIGN.md §4:
 *
 *   X   = dead slots ∪ { WALL field cells 6-connected to a dead slot through
 *         WALL cells } (wallsBound: every WALL cell)
 *   Φ   = field cells with paint OFF; regions = 6-connected components of Φ
 *   edge(R) ⇔ some cell of R has a tap in X; size(R) = |R|
 *   M   = max{ size(R) : edge(R) }, 0 if none
 *   fill(R) ⇔ !edge(R) || size(R) < M
 */
export function oracle(field: Field, paint: (slot: number) => number, wallsBound = false): Uint8Array {
  const topo = field.topo;
  const { size, cells, nbr, isField } = topo;
  const state = new Uint8Array(size).fill(WALL); // dead slots: 3, forever

  // Exterior WALL cells: X beyond the dead slots themselves.
  const exterior = new Uint8Array(size);
  const wallQueue: number[] = [];
  for (const slot of cells) {
    if (paint(slot) !== WALL) continue;
    if (wallsBound) {
      exterior[slot] = 1;
      wallQueue.push(slot);
      continue;
    }
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j < 0 || !isField[j]) {
        exterior[slot] = 1;
        wallQueue.push(slot);
        break;
      }
    }
  }
  if (!wallsBound) {
    // Flood exterior-ness through WALL-to-WALL adjacency from those seeds.
    for (let qi = 0; qi < wallQueue.length; qi++) {
      const slot = wallQueue[qi];
      for (let k = 0; k < 6; k++) {
        const j = nbr[slot * 6 + k];
        if (j >= 0 && isField[j] && paint(j) === WALL && !exterior[j]) {
          exterior[j] = 1;
          wallQueue.push(j);
        }
      }
    }
  }

  const isX = (slot: number): boolean => {
    if (slot < 0 || !isField[slot]) return true; // a dead tap
    return paint(slot) === WALL && exterior[slot] === 1;
  };

  // Regions: 6-connected components of OFF field cells.
  const regionOf = new Int32Array(size).fill(-1);
  const regionSize: number[] = [];
  const regionEdge: boolean[] = [];
  for (const slot of cells) {
    if (paint(slot) !== OFF || regionOf[slot] !== -1) continue;
    const idx = regionSize.length;
    const comp = [slot];
    regionOf[slot] = idx;
    let edge = false;
    for (let qi = 0; qi < comp.length; qi++) {
      const s = comp[qi];
      for (let k = 0; k < 6; k++) {
        const j = nbr[s * 6 + k];
        if (isX(j)) edge = true;
        if (j >= 0 && isField[j] && paint(j) === OFF && regionOf[j] === -1) {
          regionOf[j] = idx;
          comp.push(j);
        }
      }
    }
    regionSize.push(comp.length);
    regionEdge.push(edge);
  }

  let M = 0;
  for (let ri = 0; ri < regionSize.length; ri++) if (regionEdge[ri] && regionSize[ri] > M) M = regionSize[ri];

  for (const slot of cells) {
    const p = paint(slot);
    if (p === ON) state[slot] = ON;
    else if (p === WALL) state[slot] = WALL;
    else {
      const ri = regionOf[slot];
      const fills = !regionEdge[ri] || regionSize[ri] < M;
      state[slot] = fills ? FILLED : OFF;
    }
  }
  return state;
}

/** BFS eccentricity of `start` over every field cell, any paint (the epoch wave's reach). */
function eccentricityOverField(topo: Topology, start: number): number {
  const { size, nbr, isField } = topo;
  const dist = new Int32Array(size).fill(-1);
  const queue = [start];
  dist[start] = 0;
  let max = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const slot = queue[qi];
    if (dist[slot] > max) max = dist[slot];
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j >= 0 && isField[j] && dist[j] < 0) {
        dist[j] = dist[slot] + 1;
        queue.push(j);
      }
    }
  }
  return max;
}

/** BFS eccentricity of `start` within `members` only (a region's own cells). */
function eccentricityWithin(topo: Topology, members: readonly number[], start: number): number {
  const { size, nbr, isField } = topo;
  const inRegion = new Uint8Array(size);
  for (const s of members) inRegion[s] = 1;
  const dist = new Int32Array(size).fill(-1);
  const queue = [start];
  dist[start] = 0;
  let max = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const slot = queue[qi];
    if (dist[slot] > max) max = dist[slot];
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j >= 0 && isField[j] && inRegion[j] && dist[j] < 0) {
        dist[j] = dist[slot] + 1;
        queue.push(j);
      }
    }
  }
  return max;
}

/**
 * The settle bound of DESIGN.md §5 for an edit touching `edited` on the
 * picture `paint` (the picture *after* the edit): Dw (epoch wave) + the
 * worse of 3·Er or Wd + Er (leader/size vs. rim) + 2K (the root's quiet run)
 * + depth(c) + 3, with depth(c) = K throughout (the deepest cell, i.e. the
 * whole field commits by this step).
 */
export function settleBound(field: Field, paint: (slot: number) => number, edited: number[]): number {
  const topo = field.topo;
  const { size, cells, nbr, isField } = topo;
  const K = field.K;

  let Dw = 0;
  for (const e of edited) {
    if (!isField[e]) continue;
    const ecc = eccentricityOverField(topo, e);
    if (ecc > Dw) Dw = ecc;
  }

  // Regions (OFF components) and each one's max-id leader's eccentricity within it.
  const regionOf = new Int32Array(size).fill(-1);
  const regions: number[][] = [];
  for (const slot of cells) {
    if (paint(slot) !== OFF || regionOf[slot] !== -1) continue;
    const idx = regions.length;
    const comp = [slot];
    regionOf[slot] = idx;
    for (let qi = 0; qi < comp.length; qi++) {
      const s = comp[qi];
      for (let k = 0; k < 6; k++) {
        const j = nbr[s * 6 + k];
        if (j >= 0 && isField[j] && paint(j) === OFF && regionOf[j] === -1) {
          regionOf[j] = idx;
          comp.push(j);
        }
      }
    }
    regions.push(comp);
  }

  let Er = 0;
  for (const comp of regions) {
    let leader = comp[0];
    for (const s of comp) if (field.ids[s] > field.ids[leader]) leader = s;
    const ecc = eccentricityWithin(topo, comp, leader);
    if (ecc > Er) Er = ecc;
  }

  // Wd: depth of the wall-exterior flood from the dead ring (0 with no such walls).
  const dist = new Int32Array(size).fill(-1);
  const queue: number[] = [];
  for (const slot of cells) {
    if (paint(slot) !== WALL) continue;
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j < 0 || !isField[j]) {
        dist[slot] = 0;
        queue.push(slot);
        break;
      }
    }
  }
  let Wd = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const slot = queue[qi];
    if (dist[slot] > Wd) Wd = dist[slot];
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j >= 0 && isField[j] && paint(j) === WALL && dist[j] < 0) {
        dist[j] = dist[slot] + 1;
        queue.push(j);
      }
    }
  }

  return Dw + Math.max(3 * Er, Wd + Er) + 3 * K + 3;
}
