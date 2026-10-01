// A static, self-stabilising cellular automaton for "fill the inside".
//
// THE QUESTION. Walls are painted on a hexagon-shaped field, any cells, in
// any order, at any time. Walls cut the field into regions (6-connected
// groups of non-wall cells). A region is INSIDE, and fills, when it holds
// less than half of the open border (border cells that aren't walls):
//
//     fill(region) ⇔ 2·b < T     b = open border cells in the region
//                                 T = open border cells on the whole field
//
// A region walled off from the border has b = 0, so a circuit's inside
// always fills. A single wall from edge to edge splits the open border into
// two arcs and fills the side of the shorter one. Two equal halves: neither
// fills. At most one region can hold half or more; that one is the outside.
//
// THE MACHINE. Every cell holds a few small integers and runs the same rule,
// synchronously, from its own state and its six neighbours' (the hex kernel:
// the cell and its 6 neighbours). The only input is each cell's `wall` bit;
// the rule never sees a history, so the answer depends only on the walls
// there are now. From ANY state the fields settle to the same fixed point
// (they are self-stabilising), so a wall painted or erased half-way through
// is simply absorbed.
//
// Each field is one kernel formula; every one reads only the cell and its
// six neighbours. With N cells on the field and P border cells:
//
//   ring order  on the border, from each cell's outside neighbours:
//                 next = the direction just past its outside neighbours
//                 (anticlockwise), prev = the one just before them.
//   idx         position round the border: 0 at the east corner, else
//                 idx(prev) + 1.                                   [0, P)
//   cnt         open border cells so far: open + cnt(prev).        [0, P]
//   tot         T, read at the east corner from the last cell's cnt,
//                 then passed on: tot(prev).                       [0, P]
//   lab, d      region label = the smallest idx of an open border cell
//                 in the region, d = distance to it. A lexicographic
//                 min-plus kernel:
//                   (lab, d) = min( (idx, 0) if an open border cell,
//                                   (lab_k, d_k + 1) over non-wall k )
//                 with d capped at N: a label whose source is gone counts
//                 up to N and dies, so a walled-off region ends up with no
//                 label at all.                         lab ∈ [0, P) ∪ {none}
//   par         the neighbour one step nearer the label's source, with the
//                 same label (lowest direction on a tie): a spanning tree
//                 of the region, rooted at its label's source.     dir ∪ {none}
//   s           open border cells in my subtree: open + Σ s(children),
//                 children = neighbours whose par points at me.   [0, P]
//   v           the verdict: at the root (d = 0), 2·s < tot; elsewhere
//                 copied from par.                                 bit
//   fill        not a wall, and (no label, or v).                  bit
//
// Settling time after the last change: about P for the ring, then up to N for
// a cut-off label to die, then twice the region's depth for s and v.

import { Board, opp } from './hex.js';

export const NONE = -1;

export interface Cell {
  outside: boolean;
  /** The input. */
  wall: boolean;
  idx: number;
  cnt: number;
  tot: number;
  lab: number;
  d: number;
  par: number;
  s: number;
  v: boolean;
}

export const OUTSIDE: Cell = Object.freeze({
  outside: true,
  wall: false,
  idx: 0,
  cnt: 0,
  tot: 0,
  lab: NONE,
  d: 0,
  par: NONE,
  s: 0,
  v: false,
});

export function blankCell(): Cell {
  return { outside: false, wall: false, idx: 0, cnt: 0, tot: 0, lab: NONE, d: 0, par: NONE, s: 0, v: false };
}

/** The constants a rule may know: field size bounds (cap for d and counters). */
export interface Limits {
  /** Cap for distances: at least the number of cells. */
  N: number;
  /** Cap for border counters: at least the number of border cells. */
  P: number;
}

/**
 * Ring order from the outside neighbours alone: going anticlockwise round the
 * field, the next border cell is in the direction just past the run of
 * outside neighbours, the previous one just before it. -1 if not on the border.
 */
export function ringDirs(n: readonly Cell[]): { next: number; prev: number; east: boolean } {
  let next = -1;
  let prev = -1;
  let mask = 0;
  for (let k = 0; k < 6; k++) if (n[k].outside) mask |= 1 << k;
  if (!mask) return { next, prev, east: false };
  for (let k = 0; k < 6; k++) {
    if (!(mask & (1 << k))) continue;
    if (!(mask & (1 << ((k + 1) % 6)))) next = (k + 1) % 6;
    if (!(mask & (1 << ((k + 5) % 6)))) prev = (k + 5) % 6;
  }
  // The east corner: outside to the east and on both sides of it.
  const east = mask === ((1 << 5) | (1 << 0) | (1 << 1));
  return { next, prev, east };
}

/** The rule. Pure: the next state of a cell from itself and its six neighbours. */
export function rule(self: Cell, n: readonly Cell[], lim: Limits): Cell {
  if (self.outside) return self;
  const { prev, east } = ringDirs(n);
  const onRing = prev >= 0;
  const open = onRing && !self.wall;

  // Border bookkeeping: position, running count of open cells, the total.
  let idx = 0;
  let cnt = 0;
  let tot = 0;
  if (onRing) {
    const p = n[prev];
    idx = east ? 0 : Math.min(p.idx + 1, lim.P);
    cnt = Math.min((east ? 0 : p.cnt) + (open ? 1 : 0), lim.P);
    tot = east ? p.cnt : p.tot;
  }

  if (self.wall) {
    return { ...self, idx, cnt, tot, lab: NONE, d: lim.N, par: NONE, s: 0, v: false };
  }

  // Region label and distance: lexicographic min-plus over the kernel.
  let lab = open ? idx : NONE;
  let d = open ? 0 : lim.N;
  for (let k = 0; k < 6; k++) {
    const m = n[k];
    if (m.outside || m.wall || m.lab === NONE || m.d + 1 >= lim.N) continue;
    if (lab === NONE || m.lab < lab || (m.lab === lab && m.d + 1 < d)) {
      lab = m.lab;
      d = m.d + 1;
    }
  }

  // Tree: the parent is a neighbour one step nearer the source, same label.
  let par = NONE;
  if (lab !== NONE && d > 0) {
    for (let k = 0; k < 6 && par === NONE; k++) {
      const m = n[k];
      if (!m.outside && !m.wall && m.lab === lab && m.d === d - 1) par = k;
    }
  }

  // Subtree sum of open border cells: my own, plus my children's.
  let s = open ? 1 : 0;
  if (lab !== NONE) {
    for (let k = 0; k < 6; k++) {
      const m = n[k];
      if (!m.outside && !m.wall && m.par === opp(k) && m.lab === lab) s += m.s;
    }
  }
  s = Math.min(s, lim.P);

  // Verdict: decided at the root, copied down the tree.
  let v = false;
  if (lab !== NONE) {
    if (d === 0) v = 2 * s < tot;
    else if (par !== NONE) v = n[par].v;
  }

  return { outside: false, wall: false, idx, cnt, tot, lab, d, par, s, v };
}

/** The output of a cell. */
export function filled(c: Cell): boolean {
  return !c.outside && !c.wall && (c.lab === NONE || c.v);
}

function same(a: Cell, b: Cell): boolean {
  return (
    a.wall === b.wall && a.idx === b.idx && a.cnt === b.cnt && a.tot === b.tot && a.lab === b.lab &&
    a.d === b.d && a.par === b.par && a.s === b.s && a.v === b.v
  );
}

/** A running automaton on a board. */
export class Automaton {
  readonly board: Board;
  readonly limits: Limits;
  cells: Cell[];
  private next: Cell[];
  generation = 0;
  /** Cells that changed in the last step. */
  changed = 0;

  constructor(board: Board) {
    this.board = board;
    this.limits = { N: board.cells.length, P: Math.max(1, 6 * board.radius) };
    this.cells = new Array(board.size);
    for (let i = 0; i < board.size; i++) this.cells[i] = board.inside[i] ? blankCell() : OUTSIDE;
    this.next = this.cells.slice();
  }

  /** The input: make a cell a wall, or not. Any cell, any time, any number per step. */
  setWall(i: number, wall: boolean): void {
    if (!this.board.inside[i] || this.cells[i].wall === wall) return;
    this.cells[i] = { ...this.cells[i], wall };
  }

  step(): void {
    const { board, cells, next, limits } = this;
    const nb: Cell[] = new Array(6);
    let changed = 0;
    for (const i of board.cells) {
      for (let k = 0; k < 6; k++) {
        const j = board.neighbours[i * 6 + k];
        nb[k] = j < 0 ? OUTSIDE : cells[j];
      }
      const c = rule(cells[i], nb, limits);
      if (same(c, cells[i])) next[i] = cells[i];
      else {
        next[i] = c;
        changed++;
      }
    }
    this.next = cells;
    this.cells = next;
    this.generation++;
    this.changed = changed;
  }

  /** Step until nothing changes (or `max` steps). Returns steps taken. */
  run(max = 1e6): number {
    const start = this.generation;
    do this.step();
    while (this.changed > 0 && this.generation - start < max);
    return this.generation - start;
  }

  /** A bound on the steps to settle from any state, for these limits. */
  get settleBound(): number {
    return 3 * this.limits.P + this.limits.N + 2 * this.limits.N + 10;
  }

  filled(): Set<number> {
    const out = new Set<number>();
    for (const i of this.board.cells) if (filled(this.cells[i])) out.add(i);
    return out;
  }

  walls(): Set<number> {
    const out = new Set<number>();
    for (const i of this.board.cells) if (this.cells[i].wall) out.add(i);
    return out;
  }
}

