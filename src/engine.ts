// A generic layered cellular automaton on a hex board.
//
// The state is a stack of integer LAYERS (channels), one value per cell per
// layer. One step, for every cell at once:
//
//   1. PERCEIVE. Each layer is read through the hex kernel bank: the cell
//      itself (tap −1) and its six neighbours (taps 0…5: E, NE, NW, W, SW,
//      SE). These seven taps are the only thing a cell ever sees.
//   2. UPDATE. The rule maps that perception to the cell's new value in
//      every layer. The same rule runs at every cell.
//
// Cells past the edge of the board are fixed: they hold the rule's
// `outside` values in every layer and never update. Layers listed in
// `inputs` are the only ones the host may write; the rule passes them
// through unchanged.
//
// The engine knows nothing about walls or fills. The rule lives elsewhere.

import { Board } from './hex.js';

/** What a cell sees: each layer through the hex kernel bank. */
export interface Perception {
  /** Layer `layer` at tap `k`: −1 = the cell itself, 0…5 = its neighbours. */
  at(layer: number, k: number): number;
}

export interface Rule {
  /** Layer names, in order. */
  readonly layers: readonly string[];
  /** Layers the host writes (the rule leaves them alone). */
  readonly inputs: readonly number[];
  /** Each layer's value in cells past the edge of the board. */
  readonly outside: readonly number[];
  /** Each layer's value on a fresh board. */
  readonly initial: readonly number[];
  /** The update: write the cell's new value for every non-input layer into `out`. */
  update(p: Perception, out: Int32Array, consts: Consts): void;
}

/** Board-wide constants a rule may use (sizes, for caps on counters). */
export interface Consts {
  /** Number of cells on the board. */
  N: number;
  /** Number of border cells. */
  P: number;
}

export class Automaton {
  readonly board: Board;
  readonly rule: Rule;
  readonly consts: Consts;
  /** layers[L][i]: the value of layer L at slot i. */
  layers: Int32Array[];
  private nextLayers: Int32Array[];
  generation = 0;
  /** Cells whose state changed in the last step. */
  changed = 0;

  constructor(board: Board, rule: Rule) {
    this.board = board;
    this.rule = rule;
    this.consts = { N: board.cells.length, P: Math.max(1, 6 * board.radius) };
    const make = () =>
      rule.layers.map((_, L) => {
        const a = new Int32Array(board.size);
        for (let i = 0; i < board.size; i++) a[i] = board.inside[i] ? rule.initial[L] : rule.outside[L];
        return a;
      });
    this.layers = make();
    this.nextLayers = make();
  }

  layer(name: string): number {
    const L = this.rule.layers.indexOf(name);
    if (L < 0) throw new Error(`no layer ${name}`);
    return L;
  }

  get(name: string, i: number): number {
    return this.layers[this.layer(name)][i];
  }

  /** The only way in: write an input layer at a cell on the board. */
  setInput(name: string, i: number, value: number): void {
    const L = this.layer(name);
    if (!this.rule.inputs.includes(L)) throw new Error(`${name} is not an input layer`);
    if (this.board.inside[i]) this.layers[L][i] = value;
  }

  step(): void {
    const { board, rule, consts } = this;
    const cur = this.layers;
    const nxt = this.nextLayers;
    const nb = board.neighbours;
    const nLayers = cur.length;
    const out = new Int32Array(nLayers);
    const isInput = rule.layers.map((_, L) => rule.inputs.includes(L));
    // Every board cell has all six neighbours in the padded array (past the
    // edge they are outside cells holding the outside values), so a tap is
    // one array read.
    let base = 0;
    let i = 0;
    const p: Perception = {
      at: (layer, k) => (k < 0 ? cur[layer][i] : cur[layer][nb[base + k]]),
    };
    let changed = 0;
    for (const c of board.cells) {
      i = c;
      base = c * 6;
      for (let L = 0; L < nLayers; L++) out[L] = cur[L][i];
      rule.update(p, out, consts);
      let diff = false;
      for (let L = 0; L < nLayers; L++) {
        const v = isInput[L] ? cur[L][i] : out[L];
        if (v !== cur[L][i]) diff = true;
        nxt[L][i] = v;
      }
      if (diff) changed++;
    }
    this.layers = nxt;
    this.nextLayers = cur;
    this.generation++;
    this.changed = changed;
  }

  /** Step until nothing changes (or `max` steps). Returns steps taken. */
  run(max = 1e7): number {
    const start = this.generation;
    do this.step();
    while (this.changed > 0 && this.generation - start < max);
    return this.generation - start;
  }
}
