// A STAND-IN for src/game/line-ca.ts (agent A's line CA, docs/spectacle-ca-hybrid.md §2) with the frozen surface
// of §4.1, so the host and the page can be built before it lands. Not a CA: an imperative stepper with the same
// channels and the same outcomes at rest (growth one chord per `period` steps, tails, joins, mutual hits), but a
// hit wipes both lines at once instead of by waves, and there is no probe (`closed` comes from components()).
// Replaced by the real thing at integration; nothing but the host and its tests should import it.

import { Board, RuleTable, PAIR_INDEX, type Rule } from '../../strand.js';

export interface LineKnobs { period: number; oneWay: boolean; fuel: number; probe: boolean; maxTips: number }
export const DEFAULT_LINE_KNOBS: LineKnobs = { period: 2, oneWay: false, fuel: 0, probe: true, maxTips: 2 };
export type Refusal = 'no chord' | 'rival tile' | 'drawn' | 'hot' | 'heads' | 'respawn' | 'territory';
export interface TapRequest { cell: number; d0: number; d1: number; rule: number }

const opp = (d: number) => (d + 3) % 6;

/** partner[geo * 6 + d] for one rule: the edge a line entering a tile of that geo across d leaves by, -1 none. */
export function partnerTable(table: RuleTable, rule: Rule): Int8Array {
  const out = new Int8Array(108 * 6).fill(-1);
  const lp = table.subsets[rule.s].localPairs;
  for (let g = 0; g < 108; g++) {
    const t = Math.floor(g / 12);
    const rot = Math.floor(g / 2) % 6;
    const m = g % 2 ? -1 : 1;
    for (const [k0, k1] of lp[t]?.[rule.digits[t]] ?? []) {
      const a = (((m * k0 + rot) % 6) + 6) % 6;
      const b = (((m * k1 + rot) % 6) + 6) % 6;
      if (PAIR_INDEX[a * 6 + b] < 0) continue;
      out[g * 6 + a] = b;
      out[g * 6 + b] = a;
    }
  }
  return out;
}

export class LineCA {
  readonly knobs: LineKnobs;
  readonly ch: { rule: Int32Array; owner: Int32Array; D: Int32Array; T: Int32Array; W: Int32Array; H: Int32Array;
    closed: Int32Array; fuel: Int32Array };
  /** The engine CA in the real thing; absent here. */
  readonly ca = null;
  generation = 0;
  private readonly partners: Int8Array[] = [new Int8Array(108 * 6).fill(-1)];
  private readonly ownerOf: number[] = [0];
  private changedCells = new Set<number>();
  private lastChanged = new Int32Array(0);

  constructor(readonly board: Board, readonly table: RuleTable, knobs: Partial<LineKnobs> = {}) {
    this.knobs = { ...DEFAULT_LINE_KNOBS, ...knobs };
    const n = board.n;
    this.ch = {
      rule: new Int32Array(n), owner: new Int32Array(n), D: new Int32Array(n), T: new Int32Array(n),
      W: new Int32Array(n), H: new Int32Array(n), closed: new Int32Array(n), fuel: new Int32Array(n),
    };
  }

  addRule(rule: Rule, owner: number): number {
    this.partners.push(partnerTable(this.table, rule));
    this.ownerOf.push(owner);
    return this.partners.length - 1;
  }

  /** The other end of rule r's chord through edge d of cell c, -1 none. */
  pt(r: number, c: number, d: number): number {
    return r > 0 ? this.partners[r][this.geoOf(c) * 6 + d] : -1;
  }

  private geoOf(c: number): number {
    return this.board.geo[this.board.pos[c]];
  }

  tap(req: TapRequest, blocked?: (cell: number) => boolean): Refusal | null {
    const { cell: c, d0, d1, rule: r } = req;
    const { ch, knobs } = this;
    if (this.pt(r, c, d0) !== d1) return 'no chord';
    if (ch.rule[c] !== 0 && ch.rule[c] !== r) return 'rival tile';
    if ((ch.D[c] >> d0) & 1 || (ch.D[c] >> d1) & 1) return 'drawn';
    if ((ch.W[c] >> d0) & 1 || (ch.W[c] >> d1) & 1) return 'hot';
    const owner = this.ownerOf[r];
    if (this.countTips(owner) >= knobs.maxTips) return 'heads';
    if (blocked?.(c)) return 'territory';
    ch.rule[c] = r;
    ch.owner[c] = owner;
    ch.D[c] |= (1 << d0) | (1 << d1);
    ch.T[c] |= 1 << d1;
    if (!knobs.oneWay) ch.T[c] |= 1 << d0;
    ch.fuel[c] = knobs.fuel;
    this.changedCells.add(c);
    return null;
  }

  retap(cell: number, d0: number, d1: number, rule: number): boolean {
    const { ch, knobs } = this;
    if (ch.rule[cell] !== rule || !((ch.D[cell] >> d0) & 1)) return false;
    if (knobs.fuel <= 0) return false;
    // Every loose end of the line through this chord gets a tip and fresh fuel.
    let any = false;
    for (const [c, d] of this.looseEnds(cell, d0)) {
      if (this.countTips(this.ownerOf[rule]) >= knobs.maxTips) break;
      if ((ch.T[c] >> d) & 1) continue;
      const n = this.board.nbr[c * 6 + d];
      if (n < 0 || this.pt(rule, n, opp(d)) < 0) continue;
      ch.T[c] |= 1 << d;
      ch.fuel[c] = knobs.fuel;
      this.changedCells.add(c);
      any = true;
    }
    return any;
  }

  /** The loose ends (cell, edge) of the stretch through edge d of cell c. */
  private looseEnds(c0: number, d0: number): [number, number][] {
    const out: [number, number][] = [];
    const r = this.ch.rule[c0];
    for (const start of [d0, this.pt(r, c0, d0)]) {
      let c = c0;
      let d = start;
      for (let k = 0; k < this.board.n * 6; k++) {
        const n = this.board.nbr[c * 6 + d];
        const e = opp(d);
        if (n < 0 || this.ch.rule[n] !== r || !((this.ch.D[n] >> e) & 1)) { out.push([c, d]); break; }
        c = n;
        d = this.pt(r, n, e);
        if (c === c0 && (d === d0 || d === this.pt(r, c0, d0))) break; // a loop: no loose end
      }
    }
    return out;
  }

  step(go?: Uint8Array): void {
    const { ch, board, knobs } = this;
    const n = board.n;
    this.generation++;
    // Last step's pulses go; a tile kept only by its wave is free again.
    for (let c = 0; c < n; c++) {
      if (ch.W[c] || ch.H[c]) {
        ch.W[c] = 0;
        ch.H[c] = 0;
        if (!ch.D[c]) { ch.rule[c] = 0; ch.owner[c] = 0; ch.T[c] = 0; }
        this.changedCells.add(c);
      }
    }
    const phase = this.generation % knobs.period === 0;
    const moves: [number, number][] = [];
    for (let c = 0; c < n; c++) {
      const t = ch.T[c];
      if (!t) continue;
      const owner = ch.owner[c];
      if (!(go ? go[owner] : phase)) continue;
      for (let d = 0; d < 6; d++) if ((t >> d) & 1) moves.push([c, d]);
    }
    const hit = new Set<number>();
    for (const [c, d] of moves) {
      if (!((ch.T[c] >> d) & 1)) continue;
      const r = ch.rule[c];
      const nb = board.nbr[c * 6 + d];
      const e = opp(d);
      const f = nb >= 0 ? this.pt(r, nb, e) : -1;
      if (nb < 0 || f < 0 || (knobs.fuel > 0 && ch.fuel[c] <= 0)) {
        ch.T[c] &= ~(1 << d); // a tail
        this.changedCells.add(c);
        continue;
      }
      const rn = ch.rule[nb];
      if (rn !== 0 && rn !== r) {
        hit.add(c);
        hit.add(nb);
        continue;
      }
      ch.T[c] &= ~(1 << d);
      this.changedCells.add(c);
      if ((ch.D[nb] >> e) & 1) continue; // met my own line: a join, or a loop closed
      ch.rule[nb] = r;
      ch.owner[nb] = ch.owner[c];
      ch.D[nb] |= (1 << e) | (1 << f);
      if (!((ch.T[nb] >> e) & 1)) ch.T[nb] |= 1 << f;
      else ch.T[nb] &= ~(1 << e);
      ch.fuel[nb] = Math.max(0, ch.fuel[c] - 1);
      this.changedCells.add(nb);
    }
    if (hit.size) this.wipe(hit);
    this.markClosed();
    this.lastChanged = Int32Array.from(this.changedCells);
    this.changedCells.clear();
  }

  /** Every line through the hit tiles goes at once (the real CA sends waves). */
  private wipe(hit: Set<number>): void {
    const { ch } = this;
    const comps = this.componentsRaw();
    for (const comp of comps) {
      if (!comp.cells.some((c) => hit.has(c))) continue;
      for (const [c, d] of comp.ends6) {
        ch.D[c] &= ~(1 << d);
        ch.W[c] |= 1 << d;
        ch.T[c] &= ~(1 << d);
        this.changedCells.add(c);
      }
    }
    for (const c of hit) { ch.H[c] = 1; this.changedCells.add(c); }
  }

  private markClosed(): void {
    const { ch } = this;
    ch.closed.fill(0);
    for (const comp of this.componentsRaw()) {
      if (comp.ends) continue;
      for (const [c, d] of comp.ends6) ch.closed[c] |= 1 << d;
    }
  }

  changed(): Int32Array {
    return this.lastChanged;
  }

  countTips(owner: number): number {
    let k = 0;
    for (let c = 0; c < this.board.n; c++) {
      if (this.ch.owner[c] !== owner) continue;
      for (let t = this.ch.T[c]; t; t &= t - 1) k++;
    }
    return k;
  }

  lineTiles(owner: number): number {
    let k = 0;
    for (let c = 0; c < this.board.n; c++) if (this.ch.owner[c] === owner && this.ch.D[c]) k++;
    return k;
  }

  private componentsRaw(): { owner: number; rule: number; cells: number[]; ends6: [number, number][]; ends: number }[] {
    const { ch, board } = this;
    const n = board.n;
    const parent = new Int32Array(n * 6).map((_, i) => i);
    const find = (x: number): number => {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };
    const join = (a: number, b: number) => { parent[find(a)] = find(b); };
    for (let c = 0; c < n; c++) {
      const D = ch.D[c];
      if (!D) continue;
      const r = ch.rule[c];
      for (let d = 0; d < 6; d++) {
        if (!((D >> d) & 1)) continue;
        const p = this.pt(r, c, d);
        if (p >= 0) join(c * 6 + d, c * 6 + p);
        const nb = board.nbr[c * 6 + d];
        if (nb >= 0 && ch.rule[nb] === r && (ch.D[nb] >> opp(d)) & 1) join(c * 6 + d, nb * 6 + opp(d));
      }
    }
    const byRoot = new Map<number, { owner: number; rule: number; cells: number[]; ends6: [number, number][]; ends: number }>();
    for (let c = 0; c < n; c++) {
      const D = ch.D[c];
      if (!D) continue;
      for (let d = 0; d < 6; d++) {
        if (!((D >> d) & 1)) continue;
        const root = find(c * 6 + d);
        let comp = byRoot.get(root);
        if (!comp) byRoot.set(root, (comp = { owner: ch.owner[c], rule: ch.rule[c], cells: [], ends6: [], ends: 0 }));
        if (!comp.cells.includes(c)) comp.cells.push(c);
        comp.ends6.push([c, d]);
        const nb = board.nbr[c * 6 + d];
        if (!(nb >= 0 && ch.rule[nb] === ch.rule[c] && (ch.D[nb] >> opp(d)) & 1)) comp.ends++;
      }
    }
    return [...byRoot.values()];
  }

  components(): { owner: number; rule: number; cells: Int32Array; ends: number; closed: boolean }[] {
    return this.componentsRaw().map((c) => ({
      owner: c.owner, rule: c.rule, cells: Int32Array.from(c.cells), ends: c.ends, closed: c.ends === 0,
    }));
  }
}
