// A rule-agnostic cellular automaton on a slot array (DESIGN.md §6).
//
// The state is a set of named integer CHANNELS, one Int32Array each
// (structure of arrays: ch[c][slot]). One step, for every listed field slot
// at once:
//
//   1. GATHER. Every channel at the slot itself (tap 0) and at its six
//      neighbours (taps 1…6), into P[c*7 + t]. These seven taps are all a
//      cell ever sees, and the rule must not care in which order taps 1…6
//      come (`tapOrder` shuffles them in tests to prove it).
//   2. UPDATE. The rule maps P to the slot's new value in every channel.
//
// Every read sees the pre-step state: new values are buffered and applied
// after the pass. So computing only the ACTIVE slots — those that changed
// last step, their taps, and whatever the host wrote — is bit-identical to
// the reference sweep over every field slot (`stepFull`).
//
// Dead slots (isField 0) hold each channel's `dead` value and are never
// updated or written. The engine never interprets a channel.

export type ChannelKind = 'const' | 'input' | 'hidden' | 'gate' | 'out';

export interface ChannelSpec {
  name: string;
  kind: ChannelKind;
  /** GPU texture bank; the CPU engine ignores it. */
  bank: number;
  /** Value on a fresh field slot. */
  init: number;
  /** Value on every dead slot, forever. */
  dead: number;
  /** Part of the system whose fixed point the gate certifies; the engine ignores it. */
  watch: boolean;
}

export interface Uniforms {
  N: number;
  CAP: number;
  K: number;
  /** generation + 1: the number of the step being computed. */
  step: number;
  wallsBound: 0 | 1;
}

export interface Rule {
  readonly channels: readonly ChannelSpec[];
  /** P[c*7 + t]: channel c at tap t, 0 = self, 1..6 = taps in any order. out[c] holds self on entry. */
  update(P: Int32Array, out: Int32Array, u: Uniforms): void;
}

export interface Topology {
  /** Number of slots. */
  size: number;
  /** Every field slot. */
  cells: Int32Array;
  /** nbr[slot*6 + k]: slot's neighbour k (every field slot has all six in range). */
  nbr: Int32Array;
  isField: Uint8Array;
}

/** Taps per channel in P: self plus six neighbours. */
const TAPS = 7;

export class CA {
  readonly topo: Topology;
  readonly rule: Rule;
  /** The uniforms handed to the rule; `step` is set before every pass. */
  readonly u: Uniforms;
  /** SoA state, ch[c][slot]; dead slots hold `dead`. */
  readonly ch: Int32Array[];
  generation = 0;
  /** Slots whose value changed in the last step. */
  changed = 0;
  /** Slots listed for the next step. */
  active = 0;
  /** Tests only: per-slot tap permutation, tap t+1 reads neighbour tapOrder[slot*6 + t]. */
  tapOrder: Int32Array | null = null;

  private readonly index = new Map<string, number>();
  private readonly nCh: number;
  /** Channels the rule may not change (const and input): restored after update. */
  private readonly fixed: Int32Array;
  private readonly P: Int32Array;
  private readonly out: Int32Array;
  /** The active list, and the one being stepped (swapped every step). */
  private list: Int32Array;
  private work: Int32Array;
  /** 1 while a slot is in `list`, so it is listed once. */
  private listed: Uint8Array;
  /** Pending writes: (slot, value per channel) records, flat. */
  private pending: Int32Array;

  constructor(topo: Topology, rule: Rule, u: Omit<Uniforms, 'step'>, constants?: Record<string, ArrayLike<number>>) {
    this.topo = topo;
    this.rule = rule;
    this.u = { ...u, step: 1 };
    const specs = rule.channels;
    this.nCh = specs.length;
    specs.forEach((s, c) => {
      if (this.index.has(s.name)) throw new Error(`channel ${s.name} declared twice`);
      this.index.set(s.name, c);
    });
    this.fixed = Int32Array.from(
      specs.flatMap((s, c) => (s.kind === 'const' || s.kind === 'input' ? [c] : [])),
    );

    const { size, cells, nbr, isField } = topo;
    for (const slot of cells) {
      if (!isField[slot]) throw new Error(`cell ${slot} is not a field slot`);
      for (let k = 0; k < 6; k++) {
        const j = nbr[slot * 6 + k];
        if (j < 0 || j >= size) throw new Error(`field slot ${slot} has no neighbour ${k}`);
      }
    }

    this.ch = specs.map((s) => {
      const a = new Int32Array(size);
      for (let i = 0; i < size; i++) a[i] = isField[i] ? s.init : s.dead;
      return a;
    });
    for (const [name, values] of Object.entries(constants ?? {})) {
      const c = this.index.get(name);
      if (c === undefined) throw new Error(`no channel ${name}`);
      if (specs[c].kind !== 'const') throw new Error(`${name} is not a const channel`);
      const a = this.ch[c];
      for (const slot of cells) a[slot] = values[slot];
    }

    this.P = new Int32Array(this.nCh * TAPS);
    this.out = new Int32Array(this.nCh);
    this.list = new Int32Array(size);
    this.work = new Int32Array(size);
    this.listed = new Uint8Array(size);
    this.pending = new Int32Array(Math.max(16, (this.nCh + 1) * 64));
    // Everything is computed once before anything can be quiet.
    for (const slot of cells) this.enlist(slot);
  }

  private channel(name: string): number {
    const c = this.index.get(name);
    if (c === undefined) throw new Error(`no channel ${name}`);
    return c;
  }

  get(name: string, slot: number): number {
    return this.ch[this.channel(name)][slot];
  }

  /** The only way in: an 'input' channel at a field slot. Activates the slot and its taps. */
  write(name: string, slot: number, v: number): void {
    const c = this.channel(name);
    if (this.rule.channels[c].kind !== 'input') throw new Error(`${name} is not an input channel`);
    if (!(slot >= 0 && slot < this.topo.size) || !this.topo.isField[slot]) return;
    this.ch[c][slot] = v;
    this.enlistWithTaps(slot);
  }

  /** One step over the active slots. */
  step(): void {
    // Take the list; marks go so the next list can be built from scratch.
    const work = this.list;
    const n = this.active;
    this.list = this.work;
    this.work = work;
    for (let i = 0; i < n; i++) this.listed[work[i]] = 0;
    this.active = 0;
    this.pass(work, n);
  }

  /** One step over every field slot: the reference sweep. */
  stepFull(): void {
    for (let i = 0; i < this.active; i++) this.listed[this.list[i]] = 0;
    this.active = 0;
    const cells = this.topo.cells;
    this.pass(cells, cells.length);
  }

  /** Step until nothing changes (or `max` steps). Returns steps taken. */
  run(max = 1e7): number {
    const start = this.generation;
    do this.step();
    while (this.changed > 0 && this.generation - start < max);
    return this.generation - start;
  }

  /** Compute `n` slots of `slots` against the pre-step state, then apply and list the changes. */
  private pass(slots: Int32Array, n: number): void {
    const { ch, nCh, P, out, fixed, rule } = this;
    const nbr = this.topo.nbr;
    const order = this.tapOrder;
    const u = this.u;
    u.step = this.generation + 1;
    const rec = nCh + 1;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const slot = slots[i];
      const base = slot * 6;
      for (let c = 0; c < nCh; c++) {
        const a = ch[c];
        const p = c * TAPS;
        const self = a[slot];
        P[p] = self;
        out[c] = self;
        if (order) for (let t = 0; t < 6; t++) P[p + 1 + t] = a[nbr[base + order[base + t]]];
        else for (let t = 0; t < 6; t++) P[p + 1 + t] = a[nbr[base + t]];
      }
      rule.update(P, out, u);
      for (let f = 0; f < fixed.length; f++) out[fixed[f]] = P[fixed[f] * TAPS];
      let diff = false;
      for (let c = 0; c < nCh; c++) {
        if (out[c] !== P[c * TAPS]) {
          diff = true;
          break;
        }
      }
      if (!diff) continue;
      if ((count + 1) * rec > this.pending.length) {
        const grown = new Int32Array(this.pending.length * 2);
        grown.set(this.pending);
        this.pending = grown;
      }
      const at = count * rec;
      this.pending[at] = slot;
      this.pending.set(out, at + 1);
      count++;
    }

    const pending = this.pending;
    for (let i = 0; i < count; i++) {
      const at = i * rec;
      const slot = pending[at];
      for (let c = 0; c < nCh; c++) ch[c][slot] = pending[at + 1 + c];
      this.enlistWithTaps(slot);
    }
    this.changed = count;
    this.generation++;
  }

  private enlist(slot: number): void {
    if (this.listed[slot]) return;
    this.listed[slot] = 1;
    this.list[this.active++] = slot;
  }

  private enlistWithTaps(slot: number): void {
    this.enlist(slot);
    const { nbr, isField } = this.topo;
    for (let k = 0; k < 6; k++) {
      const j = nbr[slot * 6 + k];
      if (j >= 0 && isField[j]) this.enlist(j);
    }
  }
}
