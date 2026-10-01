// The circuit-and-bridge cellular automaton.
//
// Every cell runs the same rule, synchronously, from its own state and the
// states of its six neighbours (direction-indexed: the rule knows which
// neighbour is east, north-east, …, as hex CAs with a fixed neighbourhood
// order do). No cell ever reads anything else.
//
// A player draws a line one cell per step, the way a line grows in the game:
// the one input from outside is `place(cell, pred)`, "the line's next cell is
// here, and it follows the cell in direction `pred`". Everything else is the
// rule. Each line cell knows its predecessor and (once placed) its successor,
// so the line is a one-dimensional medium that signals can run along.
//
// When a cell is placed, two things can happen to the line, both seen
// locally by the new head the step after it lands:
//
//   CIRCUIT — the head touches a cell of its own line other than its
//     predecessor and the one before that (that is only a sharp turn). The
//     loop runs from the touched cell (the anchor) to the head. Which side is
//     inside is a global question, answered by the line itself:
//       pass 1  a signal walks back from the head to the anchor, adding up
//               the turn the loop makes at every cell (in 60° steps). A
//               simple loop turns exactly ±360° = ±6 steps, so the sum
//               mod 8 is 6 (counter-clockwise) or 2 (clockwise): three bits
//               are enough, however long the loop. The anchor adds its own
//               turn and the head's, and keeps the answer.
//       pass 2  the head reads the answer from the anchor next to it and
//               sends it back along the loop; every cell it passes seeds the
//               flood into its neighbours on the inside of its own turn.
//
//   BRIDGE — the head lands on the border, and the line had been on the
//     border before and left it. The ring of border cells is split by the
//     line's border cells into arcs; flood the side of the shorter arc
//     next to the head (a tie floods nothing):
//       arm     the head fires a pulse both ways round the ring.
//       echo    a pulse moves one ring cell per step; at a cell of the line
//               it turns back. Old lines don't stop it.
//       judge   the head waits for its echoes. The first came back along
//               the shorter arc; the flood is seeded on that side.
//
// The flood spreads to every neighbour that is not outside and not a cell of
// the line. Old lines (walls) don't stop it.

import { Board, DIRS, opp } from './hex.js';

export enum Terrain {
  Outside = 0,
  Empty = 1,
  /** The line being drawn: blocks the flood and carries the signals. */
  Line = 2,
  /** An old line: plain ground for pulses and flood. */
  Wall = 3,
}

/** A line cell's first step is `New`: the step it looks around and decides. */
export enum Stage {
  New = 0,
  Placed = 1,
}

/** Whether the line, up to this cell, has touched the border. */
export enum Anch {
  Unknown = 0,
  /** Never on the border so far. */
  Free = 1,
  /** Was on the border, and has left it. */
  Left = 2,
  /** On the border (this cell is a border cell). */
  On = 3,
}

export enum Event {
  None = 0,
  Circuit = 1,
  Bridge = 2,
}

export enum Orient {
  None = 0,
  Ccw = 1,
  Cw = 2,
  /** The turning sum came out impossible — a bug, never seen in the tests. */
  Bad = 3,
}

/** Bridge-judge states, on the head. */
export enum Foot {
  None = 0,
  Armed = 1,
  Listening = 2,
  Tie = 3,
  /** Decided(d) = Decided + d: the first echo came back from direction d. */
  Decided = 4,
}

export interface Cell {
  terrain: Terrain;
  /** Learned at the first step: touches outside. */
  border: boolean;
  /** Pulses on this ring cell: bit d set = a pulse that came in from direction d. */
  pulse: number;
  /** Flooded in this round. */
  flood: boolean;

  // ── Line cells only ──
  /** Direction of the previous cell of the line, -1 on the first. */
  pred: number;
  /** Direction of the next cell, -1 until one is placed. */
  next: number;
  stage: Stage;
  anch: Anch;
  /** On the head: what its placing did. */
  event: Event;
  foot: number;
  /** A head closed a loop on this cell: the direction of that head. */
  anchorDir: number;
  /** Pass 1 in flight here: the turning sum so far, mod 8 (else -1). */
  sum: number;
  /** On the anchor, after pass 1: which way the loop runs. */
  decided: Orient;
  /** On a circuit's head: the direction of the anchor, once it has answered. */
  loopX: number;
  /** Pass 2 in flight here: the loop's orientation. */
  sweep: Orient;
}

const LINE_DEFAULTS = {
  pred: -1,
  next: -1,
  stage: Stage.Placed,
  anch: Anch.Unknown,
  event: Event.None,
  foot: Foot.None,
  anchorDir: -1,
  sum: -1,
  decided: Orient.None,
  loopX: -1,
  sweep: Orient.None,
} as const;

export const OUTSIDE: Cell = Object.freeze({
  terrain: Terrain.Outside,
  border: false,
  pulse: 0,
  flood: false,
  ...LINE_DEFAULTS,
});

export function blankCell(terrain: Terrain, border = false): Cell {
  return { terrain, border, pulse: 0, flood: false, ...LINE_DEFAULTS };
}

/** dirSum[a][b] = the direction equal to vec(a) + vec(b), or -1 if none. */
const dirSum: number[][] = DIRS.map(([aq, ar]) =>
  DIRS.map(([bq, br]) => DIRS.findIndex(([q, r]) => q === aq + bq && r === ar + br)),
);

/** The turn from travelling in direction `inDir` to leaving in `outDir`, in 60° steps (+ = counter-clockwise). */
function turn(outDir: number, inDir: number): number {
  return [0, 1, 2, 0, -2, -1][(outDir - inDir + 6) % 6];
}

function orientOf(sum: number): Orient {
  const s = ((sum % 8) + 8) % 8;
  return s === 6 ? Orient.Ccw : s === 2 ? Orient.Cw : Orient.Bad;
}

/** Whether direction k lies strictly inside the counter-clockwise sweep from `from` to `to`. */
function inSweep(k: number, from: number, to: number): boolean {
  const span = (to - from + 6) % 6;
  const at = (k - from + 6) % 6;
  return at > 0 && at < span;
}

/**
 * Whether a loop cell's neighbour in direction k is on the inside. Travelling
 * in `inDir` and leaving by `outDir`, the left side is the sweep from
 * `outDir` round to where we came from; a counter-clockwise loop has its
 * inside on the left.
 */
function insideSide(k: number, inDir: number, outDir: number, o: Orient): boolean {
  const back = opp(inDir);
  if (o === Orient.Ccw) return inSweep(k, outDir, back);
  if (o === Orient.Cw) return inSweep(k, back, outDir);
  return false;
}

/** The loop's in/out directions at a cell carrying pass 2. */
function loopDirs(v: Cell): [number, number] {
  if (v.loopX >= 0) return [opp(v.loopX), v.pred]; // the head: in from the anchor
  if (v.decided !== Orient.None) return [opp(v.next), v.anchorDir]; // the anchor: out to the head
  return [opp(v.next), v.pred];
}

/**
 * Whether neighbour `m`, in direction `d` from the cell asking, holds a pulse
 * moving to the asker next. A ring cell has two ring neighbours; a pulse
 * moves to the one it did not come from.
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

  if (self.terrain === Terrain.Line) return lineRule(self, n, border);

  // ── Empty ground and walls: pulses on the ring, and the flood. ────────────
  let pulse = 0;
  let flood = self.flood;
  if (border) {
    for (let d = 0; d < 6; d++) {
      const m = n[d];
      if (!m.border) continue;
      if (m.terrain === Terrain.Line) {
        // An armed head fires into its ring neighbours…
        if (m.foot === Foot.Armed) pulse |= 1 << d;
        // …and a pulse of ours heading into another line cell turns back.
        else if (m.foot === Foot.None && (self.pulse & ~(1 << d)) !== 0) pulse |= 1 << d;
      } else if (pulseTowards(m, d)) {
        pulse |= 1 << d; // a pulse on a ring cell moves on
      }
    }
  }
  for (let d = 0; d < 6 && !flood; d++) {
    const m = n[d];
    if (m.flood) flood = true;
    else if (m.terrain === Terrain.Line) {
      // Bridge seed: the head decided for the arc that starts here.
      if (m.foot === Foot.Decided + opp(d)) flood = true;
      // Circuit seed: pass 2 is on a loop cell that has us on its inside.
      else if (m.sweep !== Orient.None) {
        const [inDir, outDir] = loopDirs(m);
        if (insideSide(opp(d), inDir, outDir, m.sweep)) flood = true;
      }
    }
  }
  if (pulse === self.pulse && flood === self.flood && border === self.border) return self;
  return { ...self, border, pulse, flood };
}

function lineRule(self: Cell, n: readonly Cell[], border: boolean): Cell {
  const c: Cell = { ...self, border };

  if (self.stage === Stage.New) {
    // The first look round, the step after being placed.
    const p = self.pred >= 0 ? n[self.pred] : null;
    if (border) c.anch = Anch.On;
    else if (!p) c.anch = Anch.Free;
    else if (p.anch === Anch.Unknown) return c; // can't happen at one placement per step
    else c.anch = p.anch === Anch.Free ? Anch.Free : Anch.Left;
    c.stage = Stage.Placed;

    const gp = p && p.pred >= 0 ? dirSum[self.pred][p.pred] : -1;
    let closed = false;
    for (let d = 0; d < 6; d++) {
      if (d !== self.pred && d !== gp && n[d].terrain === Terrain.Line) closed = true;
    }
    if (closed) {
      c.event = Event.Circuit;
      c.sum = 0; // pass 1 starts here
    } else if (c.anch === Anch.On && p && p.anch === Anch.Left) {
      c.event = Event.Bridge;
      c.foot = Foot.Armed;
    }
    return c;
  }

  // Learn the successor: the neighbour whose predecessor is us.
  if (c.next < 0) {
    for (let d = 0; d < 6; d++) {
      const m = n[d];
      if (m.terrain === Terrain.Line && m.pred === opp(d)) c.next = d;
    }
  }

  // A newly placed head next to us that is neither our successor nor the one
  // after it has closed a loop on us: we are an anchor.
  for (let d = 0; d < 6; d++) {
    const m = n[d];
    if (m.terrain !== Terrain.Line || m.stage !== Stage.New) continue;
    if (m.pred === opp(d)) continue; // our successor
    if (c.next >= 0 && m.pred >= 0 && dirSum[d][m.pred] === c.next) continue; // our successor's successor
    c.anchorDir = d;
  }

  // Bridge judge.
  if (c.foot === Foot.Armed) c.foot = Foot.Listening;
  else if (c.foot === Foot.Listening) {
    let heard = -1;
    let count = 0;
    for (let d = 0; d < 6; d++) {
      const m = n[d];
      if (!m.border) continue;
      // An echo coming in, or a line cell right next to us on the ring (an empty arc).
      const echo = m.terrain === Terrain.Line ? true : pulseTowards(m, d);
      if (echo) {
        heard = d;
        count++;
      }
    }
    if (count === 1) c.foot = Foot.Decided + heard;
    else if (count > 1) c.foot = Foot.Tie;
  }

  // Pass 1: take the sum from our successor and add our turn — or, on the
  // anchor, finish it with our turn and the head's.
  c.sum = -1;
  if (self.next >= 0 && n[self.next].sum >= 0 && self.decided === Orient.None) {
    const s = n[self.next].sum;
    if (self.anchorDir >= 0) {
      const head = n[self.anchorDir];
      c.decided = orientOf(s + turn(self.anchorDir, opp(self.next)) + turn(head.pred, self.anchorDir));
    } else {
      c.sum = (s + turn(self.pred, opp(self.next)) + 8) % 8;
    }
  }

  // Pass 2: the head fetches the answer from the anchor beside it; every
  // other loop cell takes it from its successor, stopping at the anchor.
  c.sweep = Orient.None;
  if (self.event === Event.Circuit && self.loopX < 0) {
    for (let d = 0; d < 6; d++) {
      const m = n[d];
      if (m.terrain === Terrain.Line && m.decided !== Orient.None) {
        c.loopX = d;
        c.sweep = m.decided;
      }
    }
  } else if (self.next >= 0) {
    const m = n[self.next];
    if (m.sweep !== Orient.None && m.decided === Orient.None) c.sweep = m.sweep;
  }

  return same(c, self) ? self : c;
}

function same(a: Cell, b: Cell): boolean {
  for (const k in a) if (a[k as keyof Cell] !== b[k as keyof Cell]) return false;
  return true;
}

/** A running automaton on a board. */
export class Automaton {
  readonly board: Board;
  cells: Cell[];
  private next: Cell[];
  generation = 0;
  /** Cells that changed in the last step. */
  changed = 0;
  /** The line being drawn, in order (kept for the caller; the rule never reads it). */
  line: number[] = [];

  constructor(board: Board) {
    this.board = board;
    this.cells = new Array(board.size);
    for (let i = 0; i < board.size; i++) {
      this.cells[i] = board.inside[i] ? blankCell(Terrain.Empty) : OUTSIDE;
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

  /** The head of the line being drawn, or -1. */
  get head(): number {
    return this.line.length ? this.line[this.line.length - 1] : -1;
  }

  /** What the head's placing did (Event.None while the line is still open or not yet looked at). */
  get event(): Event {
    const h = this.head;
    return h < 0 ? Event.None : this.cells[h].event;
  }

  /** Whether the line can take another cell now. */
  get canExtend(): boolean {
    const h = this.head;
    return h < 0 || (this.cells[h].stage === Stage.Placed && this.cells[h].event === Event.None);
  }

  /** Whether `i` may be the line's next cell. */
  canPlace(i: number): boolean {
    if (!this.board.inside[i] || this.cells[i].terrain !== Terrain.Empty || !this.canExtend) return false;
    return this.head < 0 || this.directionTo(i, this.head) >= 0;
  }

  /**
   * The input: the line's next cell. The first cell starts the line; every
   * later one must be next to the head. At most one per step — the rule
   * needs a step to look round.
   */
  place(i: number): void {
    if (!this.canPlace(i)) throw new Error(`can't place ${i}`);
    const pred = this.head < 0 ? -1 : this.directionTo(i, this.head);
    const old = this.cells[i];
    this.cells[i] = { ...blankCell(Terrain.Line, old.border), pred, stage: Stage.New };
    this.line.push(i);
  }

  /**
   * End a round from outside the automaton — a new line is about to be drawn:
   * the line becomes a wall and the flood is handed back (for the caller to
   * record). Call it once the board is quiet.
   */
  commit(): Set<number> {
    const flooded = new Set<number>();
    for (const i of this.board.cells) {
      const c = this.cells[i];
      if (c.flood) flooded.add(i);
      this.cells[i] = blankCell(c.terrain === Terrain.Line ? Terrain.Wall : c.terrain, c.border);
    }
    this.line = [];
    return flooded;
  }

  flooded(): Set<number> {
    const out = new Set<number>();
    for (const i of this.board.cells) if (this.cells[i].flood) out.add(i);
    return out;
  }

  /** Direction from cell `a` to its neighbour `b`, or -1. */
  directionTo(a: number, b: number): number {
    for (let d = 0; d < 6; d++) if (this.board.neighbours[a * 6 + d] === b) return d;
    return -1;
  }
}

/**
 * Draw a whole line, one cell per step, stopping early if it closes a
 * circuit or makes a bridge; then run until the board is quiet. Returns the
 * event and how many cells were drawn.
 */
export function drawLine(ca: Automaton, cells: readonly number[]): { event: Event; drawn: number } {
  let drawn = 0;
  for (const i of cells) {
    if (!ca.canExtend) break;
    ca.place(i);
    drawn++;
    ca.step();
  }
  ca.run();
  return { event: ca.event, drawn };
}
