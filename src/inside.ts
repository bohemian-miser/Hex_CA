// The "inside" rule: a layer stack and one update, for the layered engine.
//
// Input: the `wall` layer (1 = wall), written anywhere at any time.
// Output: the `fill` layer.
//
// A region (6-connected non-wall cells) is inside, and fills, when it holds
// less than half of the open border (border cells that aren't walls):
//
//     fill ⇔ 2·b < T     b = the region's open border cells, T = the field's
//
// So a walled-in pocket (b = 0) always fills, and a wall from edge to edge
// fills the side of the shorter arc. Equal halves fill nothing.
//
// Every formula below reads layers only through the kernel bank: self(L) is
// tap −1, nb(L, k) is neighbour k. Each hidden layer is the fixed point of
// its formula given the layers above it, so the stack settles to the same
// answer from any state, and walls changed mid-way are simply absorbed.
//
//   layer  formula (per cell, every step)
//   out    1 past the edge, 0 on the board (fixed)
//   wall   input
//   idx    border cells only. prev = the neighbour just before the run of
//          outside neighbours, going anticlockwise (read off nb(out, ·)).
//          idx = 0 at the east corner (outside exactly at taps 5, 0, 1),
//          else min(nb(idx, prev) + 1, P).                    position on the border
//   cnt    open + (east corner ? 0 : nb(cnt, prev)).          open border cells so far
//   tot    east corner ? nb(cnt, prev) : nb(tot, prev).       T, passed round
//   lab,d  wall: (none, N). Else the lexicographic minimum of
//            (idx, 0) if an open border cell, and
//            (nb(lab, k), nb(d, k) + 1) over non-wall k with nb(d, k) + 1 < N.
//          lab = the region's smallest border position; d = distance to it.
//          A label whose source is walled off counts d up to N and dies.
//   par    the first k with nb(lab, k) = lab and nb(d, k) = d − 1; −1 at the
//          root (d = 0) or with no label.                     a spanning tree
//   s      open + Σ nb(s, k) over k with nb(par, k) pointing back here
//          and nb(lab, k) = lab.                              border cells below
//   v      root: 2·s < tot. Else nb(v, par).                  the verdict
//   fill   not wall, and (lab = none or v).

import { Automaton, Consts, Perception, Rule } from './engine.js';
import { Board, opp } from './hex.js';

export const NONE = -1;

export const LAYERS = ['out', 'wall', 'idx', 'cnt', 'tot', 'lab', 'd', 'par', 's', 'v', 'fill'] as const;
export const [OUT, WALL, IDX, CNT, TOT, LAB, D, PAR, S, V, FILL] = LAYERS.map((_, i) => i);

/** The run of outside neighbours, read off the `out` layer: prev and east-corner flag. */
function borderShape(p: Perception): { prev: number; east: boolean } {
  let mask = 0;
  for (let k = 0; k < 6; k++) if (p.at(OUT, k)) mask |= 1 << k;
  if (!mask || mask === 63) return { prev: -1, east: false };
  let prev = -1;
  for (let k = 0; k < 6; k++) {
    if (mask & (1 << k) && !(mask & (1 << ((k + 5) % 6)))) prev = (k + 5) % 6;
  }
  return { prev, east: mask === ((1 << 5) | (1 << 0) | (1 << 1)) };
}

export const insideRule: Rule = {
  layers: LAYERS,
  inputs: [OUT, WALL],
  //         out wall idx cnt tot  lab   d  par s  v  fill
  outside: [1,  0,   0,  0,  0,  NONE, 0, -1, 0, 0, 0],
  initial: [0,  0,   0,  0,  0,  NONE, 0, -1, 0, 0, 0],

  update(p: Perception, out: Int32Array, { N, P }: Consts): void {
    const wall = p.at(WALL, -1) === 1;
    const { prev, east } = borderShape(p);
    const onBorder = prev >= 0;
    const open = onBorder && !wall ? 1 : 0;

    // ── Border bookkeeping ──
    let idx = 0;
    let cnt = 0;
    let tot = 0;
    if (onBorder) {
      idx = east ? 0 : Math.min(p.at(IDX, prev) + 1, P);
      cnt = Math.min((east ? 0 : p.at(CNT, prev)) + open, P);
      tot = east ? p.at(CNT, prev) : p.at(TOT, prev);
    }
    out[IDX] = idx;
    out[CNT] = cnt;
    out[TOT] = tot;

    if (wall) {
      out[LAB] = NONE;
      out[D] = N;
      out[PAR] = -1;
      out[S] = 0;
      out[V] = 0;
      out[FILL] = 0;
      return;
    }

    // ── Region label and distance: lexicographic min-plus ──
    let lab = open ? idx : NONE;
    let d = open ? 0 : N;
    for (let k = 0; k < 6; k++) {
      if (p.at(OUT, k) || p.at(WALL, k)) continue;
      const l = p.at(LAB, k);
      const dk = p.at(D, k) + 1;
      if (l === NONE || dk >= N) continue;
      if (lab === NONE || l < lab || (l === lab && dk < d)) {
        lab = l;
        d = dk;
      }
    }
    out[LAB] = lab;
    out[D] = d;

    // ── Spanning tree ──
    let par = -1;
    if (lab !== NONE && d > 0) {
      for (let k = 0; k < 6 && par < 0; k++) {
        if (!p.at(OUT, k) && !p.at(WALL, k) && p.at(LAB, k) === lab && p.at(D, k) === d - 1) par = k;
      }
    }
    out[PAR] = par;

    // ── Subtree count of open border cells ──
    let s = open;
    if (lab !== NONE) {
      for (let k = 0; k < 6; k++) {
        if (!p.at(OUT, k) && !p.at(WALL, k) && p.at(PAR, k) === opp(k) && p.at(LAB, k) === lab) s += p.at(S, k);
      }
    }
    out[S] = Math.min(s, P);

    // ── Verdict: at the root, copied down ──
    let v = 0;
    if (lab !== NONE) {
      if (d === 0) v = 2 * out[S] < tot ? 1 : 0;
      else if (par >= 0) v = p.at(V, par);
    }
    out[V] = v;

    out[FILL] = lab === NONE || v ? 1 : 0;
  },
};

// ── Helpers for hosts and tests (they only read layers, or write `wall`) ──

export function insideAutomaton(board: Board): Automaton {
  return new Automaton(board, insideRule);
}

export function setWall(ca: Automaton, i: number, wall: boolean): void {
  ca.setInput('wall', i, wall ? 1 : 0);
}

export function filledCells(ca: Automaton): Set<number> {
  const out = new Set<number>();
  for (const i of ca.board.cells) if (ca.layers[FILL][i]) out.add(i);
  return out;
}

export function wallCells(ca: Automaton): Set<number> {
  const out = new Set<number>();
  for (const i of ca.board.cells) if (ca.layers[WALL][i]) out.add(i);
  return out;
}

/** A bound on the steps to settle from any state: the ring, then a label's death, then the tree twice. */
export function settleBound(ca: Automaton): number {
  return 3 * ca.consts.P + 3 * ca.consts.N + 10;
}
