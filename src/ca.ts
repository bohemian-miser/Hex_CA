// The flood-fill cellular automaton.
//
// Every cell runs the same rule, synchronously, from its own state and the
// states of its six neighbours (direction-indexed: the rule knows which
// neighbour is east, north-east, …, as hex CAs with a fixed neighbourhood
// order do). No cell ever reads anything else.
//
// The job: a fresh line of cells crosses the board edge to edge. It splits
// the border ring into two arcs; flood the side of the line whose arc is the
// shorter (a tie floods nothing). The pieces, all local:
//
//   1. border  — a cell next to `outside` knows it is on the border ring.
//                The ring is a cycle: each border cell has exactly two
//                border neighbours (true of any hexagon-shaped board, R ≥ 1).
//   2. arm     — a fresh line cell on the border is one of the line's two
//                feet. It arms, then fires a pulse into both of its ring
//                neighbours.
//   3. race    — a pulse moves one ring cell per step, away from where it
//                came from. Pulses pass through each other and through old
//                lines' cells (walls); a fresh foot swallows them.
//   4. judge   — a foot hears pulses from its partner foot. The first to
//                arrive came along the shorter arc; the foot remembers which
//                neighbour it came from. Both pulses at once is a tie.
//                The two feet hear their partner's pulses at the same step,
//                so they always agree on the side.
//   5. flood   — the ring cell a judging foot points at is the seed; flood
//                spreads to every neighbour that is not outside and not the
//                fresh line. Walls don't stop it.
//
// Time: the feet decide at step 2 + s (s = cells on the shorter arc), the
// flood then spreads one cell per step.

import { Board, opp } from './hex.js';

export enum Terrain {
  Outside = 0,
  Empty = 1,
  /** The fresh line: the one being resolved. Blocks the flood; its feet judge. */
  Line = 2,
  /** An old line: plain ground for pulses and flood. */
  Wall = 3,
}

/** Foot states (only meaningful on a fresh line cell). */
export enum Foot {
  None = 0,
  Armed = 1,
  Listening = 2,
  Tie = 3,
  /** Decided(d) = Decided + d: the pulse that won arrived from direction d. */
  Decided = 4,
}

export interface Cell {
  terrain: Terrain;
  /** Learned at the first step: touches outside. */
  border: boolean;
  foot: number;
  /** Pulses on this ring cell: bit d set = a pulse that came in from direction d. */
  pulse: number;
  /** Flooded in this round. */
  flood: boolean;
}

export const OUTSIDE: Cell = Object.freeze({
  terrain: Terrain.Outside,
  border: false,
  foot: Foot.None,
  pulse: 0,
  flood: false,
});

export function freshCell(terrain: Terrain): Cell {
  return { terrain, border: false, foot: Foot.None, pulse: 0, flood: false };
}

/**
 * Whether neighbour `m`, in direction `d` from the cell asking, holds a pulse
 * that moves to the asker next. A ring cell has two ring neighbours; a pulse
 * moves to the one it did not come from. So `m`'s pulse comes our way unless
 * it came from us (from `m`'s side, we are direction opp(d)).
 */
function pulseTowards(m: Cell, d: number): boolean {
  return (m.pulse & ~(1 << opp(d))) !== 0;
}

/** The rule. Pure: the next state of a cell from itself and its six neighbours. */
export function rule(self: Cell, n: readonly Cell[]): Cell {
  if (self.terrain === Terrain.Outside) return self;

  let touchesOutside = false;
  for (let d = 0; d < 6; d++) if (n[d].terrain === Terrain.Outside) touchesOutside = true;
  const border = self.border || touchesOutside;

  // ── Fresh line cells: only the feet do anything. ──────────────────────────
  if (self.terrain === Terrain.Line) {
    let foot = self.foot;
    if (foot === Foot.None) {
      if (touchesOutside) foot = Foot.Armed;
    } else if (foot === Foot.Armed) {
      foot = Foot.Listening; // the pulses left this step (see below)
    } else if (foot === Foot.Listening) {
      let heard = -1;
      let count = 0;
      for (let d = 0; d < 6; d++) {
        const m = n[d];
        if (m.border && m.terrain !== Terrain.Line && pulseTowards(m, d)) {
          heard = d;
          count++;
        }
      }
      if (count === 1) foot = Foot.Decided + heard;
      else if (count > 1) foot = Foot.Tie;
    }
    if (foot === self.foot && border === self.border) return self;
    return { ...self, border, foot };
  }

  // ── Empty ground and walls. ───────────────────────────────────────────────
  let pulse = 0;
  let flood = self.flood;
  for (let d = 0; d < 6; d++) {
    const m = n[d];
    if (border && m.border) {
      // An armed foot fires into its ring neighbours…
      if (m.terrain === Terrain.Line) {
        if (m.foot === Foot.Armed) pulse |= 1 << d;
      } else if (pulseTowards(m, d)) {
        // …and a pulse on a ring cell moves on.
        pulse |= 1 << d;
      }
    }
    if (!flood) {
      if (m.flood) flood = true;
      // The seed: a foot that decided for the arc we start.
      else if (m.terrain === Terrain.Line && m.foot === Foot.Decided + opp(d)) flood = true;
    }
  }
  if (pulse === self.pulse && flood === self.flood && border === self.border) return self;
  return { ...self, border, pulse, flood };
}

/** A running automaton on a board. */
export class Automaton {
  readonly board: Board;
  cells: Cell[];
  private next: Cell[];
  generation = 0;
  /** Cells that changed in the last step. */
  changed = 0;

  constructor(board: Board, terrain: (i: number) => Terrain) {
    this.board = board;
    this.cells = new Array(board.size);
    for (let i = 0; i < board.size; i++) {
      this.cells[i] = board.inside[i] ? freshCell(terrain(i)) : OUTSIDE;
    }
    this.next = this.cells.slice();
  }

  step(): void {
    const { board, cells, next } = this;
    const nb: Cell[] = new Array(6);
    let changed = 0;
    for (const i of board.cells) {
      for (let d = 0; d < 6; d++) {
        const j = board.neighbours[i * 6 + d];
        nb[d] = j < 0 ? OUTSIDE : cells[j];
      }
      const c = rule(cells[i], nb);
      if (c !== cells[i]) changed++;
      next[i] = c;
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

  /**
   * End a round from outside the automaton — the game's "a new line is drawn"
   * input, not part of the rule: the fresh line becomes a wall, the flood is
   * handed back (for the caller to record), and `line` becomes the new fresh
   * line. Pulses have all been swallowed by the time the board is quiet.
   */
  commit(line: Iterable<number> = []): Set<number> {
    const flooded = new Set<number>();
    for (const i of this.board.cells) {
      const c = this.cells[i];
      if (c.flood) flooded.add(i);
      this.cells[i] = {
        terrain: c.terrain === Terrain.Line ? Terrain.Wall : c.terrain,
        border: c.border,
        foot: Foot.None,
        pulse: 0,
        flood: false,
      };
    }
    for (const i of line) this.cells[i] = { ...this.cells[i], terrain: Terrain.Line };
    return flooded;
  }

  flooded(): Set<number> {
    const out = new Set<number>();
    for (const i of this.board.cells) if (this.cells[i].flood) out.add(i);
    return out;
  }
}
