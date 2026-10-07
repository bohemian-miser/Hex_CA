// The line CA of docs/spectacle-ca-hybrid.md §2: Spectacle's lines — taps, two-way growth, tails, own-line joins,
// mutual tile-rule collisions, wipe waves, loop closure — as a deterministic, synchronous, radius-1 integer CA on
// the strand `Board`. Every cell reads only itself, its six neighbours, its constant `geo` and the per-rule chord
// table `partner[rule][geo][d]`; the host (src/game/host.ts) only writes taps between steps and pulses `go`.
//
// The update is the doc's §2.3, with six changes the doc's own invariants (§2.2, §5) call for; each is marked
// "(fix n)" where it is made:
//   1. hitAt also needs a chord ahead: a tip at a tail into a rival's tile is no hitter (the victim side never saw
//      it as aimed, so the doc's form killed the hitter alone).
//   2. A crash counts only aimers whose edge here carries no W yet, and its W goes out only on such edges: the
//      crashed tips are still there on the next step, and the doc's form crashed again (two H pulses, W for two).
//   3. A tip killed by a crash's wave pulses H on its own tile too, so the host can charge the respawn delay to
//      the two players (the crash tile itself has owner 0).
//   4. A probe carries the edge it must come back across (cell·6 + edge + 1, not cell + 1): on a tile the strand
//      visits twice, a probe coming back into the other chord is a second visit, not a loop.
//   5. `closed` also crosses into the tile ahead in the same step (cross(pt(d))), so it floods one tile a step
//      and reaches the whole loop within the doc's 1.5 L.
//   6. Neither a probe nor `closed` is taken across an edge where my own wave went out last step (W[d]): that
//      wave is erasing the neighbour's chord this very step, and a tip that drew in right behind it would
//      otherwise pick up the dead loop's flag.
// One change to the taps: 'hot' also refuses a chord a wave is about to cross into (a neighbour's W pointing at
// either end): the doc checks only the tile's own W, and such a tap was erased a step later as part of the dying
// line, with no hit (Spectacle's tap would live).
// And three choices where the doc is silent: a tip is never made where the tile's fuel is 0 (bounded growth stops
// with the chord drawn, no tip waits there); drawing beside on a tile of one's own takes the smaller fuel; and no
// owner ever goes on two steps running (period ≥ 2, and `go` pulses are clamped). A wave moves a tile a step, so
// only a tip slower than that is always caught (§2.10): at a chord a step, a doomed line on a loop strand chases
// its own wave round the loop for ever.

import { CA, type ChannelSpec, type Rule as CARule, type Topology } from '../engine.js';
import { Board, type Rule, RuleTable } from '../strand.js';

export interface LineKnobs {
  /** Steps per chord a tip grows: a tile draws only on its owner's `go` step. At least 2 (a wave must outrun tips). */
  period: number;
  /** A tap grows one way (toward its d1) instead of both. */
  oneWay: boolean;
  /** Bounded growth (§2.8): a tap lays at most this many chords each way; 0 = unbounded. */
  fuel: number;
  /** The loop probe (§2.9), which keeps `closed`. */
  probe: boolean;
  /** Live tips per player (§0 decision 6): 2 = one two-way line. */
  maxTips: number;
}

export const DEFAULT_LINE_KNOBS: LineKnobs = { period: 2, oneWay: false, fuel: 0, probe: true, maxTips: 2 };

export type Refusal = 'no chord' | 'rival tile' | 'drawn' | 'hot' | 'heads' | 'respawn' | 'territory';

/** A tap of chord (d0, d1) of a cell under a rule index (from `addRule`); it grows out of d1, and d0 unless oneWay. */
export interface TapRequest { cell: number; d0: number; d1: number; rule: number }

/** A connected set of drawn chords of one rule (a stretch of one strand), from union-find over the chords. */
export interface Component {
  owner: number;
  rule: number;
  /** The distinct cells it runs through, ascending. */
  cells: Int32Array;
  /** Chord ends with no continuation drawn beyond them (0 for a loop, 2 for a line). */
  ends: number;
  closed: boolean;
  /** How many chords (a twice-visited tile counts two). Not in the doc's interface; an addition. */
  chords: number;
}

/** geo = type·12 + rot·2 + mirror: 9 leaf types. */
export const GEOS = 108;
/** Rules are small integers 1..MAX_RULES (10 bits); 0 = none. */
export const MAX_RULES = 1023;
/** Owners are 1..MAX_OWNER (4 bits); 0 = none. */
export const MAX_OWNER = 15;
export const OPP = [3, 4, 5, 0, 1, 2] as const;

// Channel indices (ca.ch[c]); the order is the doc's §4.1 comment, with `id` (a cell's own index, for the probe).
export const CH = { geo: 0, id: 1, rule: 2, owner: 3, D: 4, T: 5, W: 6, H: 7, fuel: 8, closed: 9, probe: 10 } as const;
const NCH = 16;
const G6 = GEOS * 6;
const TP = 7; // taps per channel in P (src/engine.ts TAPS)
const GEO7 = CH.geo * TP, ID7 = CH.id * TP, RULE7 = CH.rule * TP, OWNER7 = CH.owner * TP, D7 = CH.D * TP;
const T7 = CH.T * TP, W7 = CH.W * TP, FUEL7 = CH.fuel * TP, CLOSED7 = CH.closed * TP, PROBE7 = CH.probe * TP;

const spec = (name: string, kind: ChannelSpec['kind'], dead: number): ChannelSpec => ({
  name, kind, bank: 0, init: kind === 'const' ? dead : 0, dead, watch: kind !== 'const',
});

const CHANNELS: ChannelSpec[] = [
  spec('geo', 'const', -1),
  spec('id', 'const', -1),
  spec('rule', 'hidden', 0),
  spec('owner', 'hidden', 0),
  spec('D', 'hidden', 0),
  spec('T', 'hidden', 0),
  spec('W', 'hidden', 0),
  spec('H', 'out', 0),
  spec('fuel', 'hidden', 0),
  spec('closed', 'out', 0),
  ...[0, 1, 2, 3, 4, 5].map((d) => spec(`probe${d}`, 'hidden', 0)),
];
if (CHANNELS.length !== NCH) throw new Error('line-ca: channel count');

/** The state the update reads besides P: the chord table, the go pulses and the knobs. */
interface Shared {
  partner: Int8Array;
  go: Uint8Array;
  fuelMax: number;
  probe: boolean;
}

/** The §2.3 update as an engine Rule. Taps 1..6 must be in direction order (Topology.nbr), as the rule reads them. */
function lineRule(sh: Shared): CARule {
  const fA = new Int8Array(6);
  const inV = new Int32Array(6);
  return {
    channels: CHANNELS,
    update(P: Int32Array, out: Int32Array): void {
      const rule = P[RULE7];
      const W = P[W7];
      if (rule === 0 && W === 0) {
        // Nothing on me: only a tip aimed at me can change that.
        let any = 0;
        for (let t = 1; t <= 6; t++) any |= P[T7 + t];
        if (any === 0) {
          out[CH.H] = 0;
          out[CH.fuel] = 0;
          return;
        }
      }
      const pt = sh.partner;
      const fuelMax = sh.fuelMax;
      const go = sh.go;
      const geo = P[GEO7];
      const D = P[D7], T = P[T7], owner = P[OWNER7], fuel = P[FUEL7];
      const meG = geo * 6;
      const myRow = rule * G6;
      const fuelOut = fuelMax > 0 && fuel === 0;

      let aimed = 0, waveIn = 0, waveZero = 0, hit = 0, tail = 0, moved = 0;
      let victim = false, crash = false, crashRule = 0;
      for (let d = 0; d < 6; d++) {
        const t = d + 1;
        const o = OPP[d];
        const bit = 1 << d;
        const gd = P[GEO7 + t];
        if (gd < 0) { // off the board
          tail |= bit;
          continue;
        }
        const rd = P[RULE7 + t];
        // aimed(d): a live tip in N_d pointed at me, with a chord here under its rule
        if ((P[T7 + t] >> o) & 1 && !(fuelMax > 0 && P[FUEL7 + t] === 0)) {
          const f = pt[rd * G6 + meG + d];
          if (f >= 0) {
            aimed |= bit;
            fA[d] = f;
            if (rd !== rule) {
              if (rule !== 0) victim = true;
              else if (!((W >> d) & 1)) { // (fix 2) an aimer already answered by a crash wave is not counted again
                if (crashRule === 0) crashRule = rd;
                else if (crashRule !== rd) crash = true;
              }
            }
          }
        }
        // waveIn(d): a wipe wave arriving across d along my strand (a crash tile's carries rule 0)
        if ((P[W7 + t] >> o) & 1 && (rd === rule || rd === 0)) {
          waveIn |= bit;
          if (rd === 0) waveZero |= bit;
        }
        if ((T >> d) & 1) {
          if (fuelOut || pt[myRow + gd * 6 + o] < 0) tail |= bit;
          else if (rd !== 0 && rd !== rule) hit |= bit; // (fix 1) hitAt: a chord ahead, and a rival's tile
          if (rd === rule && (P[D7 + t] >> o) & 1) moved |= bit;
        }
      }

      // 1-2. collisions and erasing
      let gone = 0;
      if (D !== 0) {
        if (victim) gone = D;
        else {
          const x = (waveIn | hit) & D;
          if (x !== 0) {
            for (let d = 0; d < 6; d++) if ((x >> d) & 1) gone |= (1 << d) | (1 << pt[myRow + meG + d]);
          }
        }
      }
      let D2 = D & ~gone;
      const W2 = (gone & ~waveIn) | (crash ? aimed & ~W : 0);
      let T2 = T & ~gone & ~tail & ~moved;

      // 4. receiving tips
      let enter = 0, eRule = 0, eOwner = 0, eFuel = 1 << 30;
      if (aimed !== 0 && !victim && !crash) {
        const free = aimed & ~D & ~W;
        for (let d = 0; d < 6; d++) {
          if (!((free >> d) & 1)) continue;
          const t = d + 1;
          const rd = P[RULE7 + t];
          const od = P[OWNER7 + t];
          if (!go[od] || (rule !== 0 && rule !== rd)) continue;
          enter |= 1 << d;
          eRule = rd;
          eOwner = od;
          eFuel = Math.min(eFuel, P[FUEL7 + t]);
        }
      }
      let newTips = 0;
      if (enter !== 0) {
        for (let d = 0; d < 6; d++) {
          if (!((enter >> d) & 1)) continue;
          const f = fA[d];
          D2 |= (1 << d) | (1 << f);
          if (!((enter >> f) & 1)) newTips |= 1 << f;
        }
      }
      const keep = D2 !== 0 || W2 !== 0;
      const nr = enter !== 0 ? (rule !== 0 ? rule : eRule) : keep ? rule : 0;
      const no = enter !== 0 ? (rule !== 0 ? owner : eOwner) : keep ? owner : 0;
      let fuel2 = 0;
      if (fuelMax > 0 && nr !== 0) {
        if (enter !== 0) {
          const m = Math.max(0, eFuel - 1);
          fuel2 = rule === 0 ? m : Math.min(fuel, m);
        } else fuel2 = fuel;
      }
      if (newTips !== 0 && (fuelMax === 0 || fuel2 > 0)) T2 |= newTips;
      const H2 = victim || crash || hit !== 0 || (T & waveZero) !== 0 ? 1 : 0; // (fix 3) the crash's victims

      // §2.9 the loop probe and `closed`
      let closed2 = 0;
      if (sh.probe && D2 !== 0) {
        const id = P[ID7];
        const nRow = nr * G6 + meG;
        let met = 0;
        const landed = aimed & D & ~gone;
        if (landed !== 0) {
          for (let d = 0; d < 6; d++) if ((landed >> d) & 1 && P[RULE7 + d + 1] === rule) met |= 1 << d;
        }
        if (enter !== 0) {
          for (let d = 0; d < 6; d++) if ((enter >> d) & 1 && (enter >> fA[d]) & 1) met |= 1 << d;
        }
        let cross = 0;
        for (let d = 0; d < 6; d++) {
          const t = d + 1;
          const o = OPP[d];
          inV[d] = 0;
          if (P[GEO7 + t] < 0 || P[RULE7 + t] !== nr || !((P[D7 + t] >> o) & 1) || (W >> d) & 1) continue; // (fix 6)
          inV[d] = P[PROBE7 + o * TP + t];
          if ((P[CLOSED7 + t] >> o) & 1) cross |= 1 << d;
        }
        const C = P[CLOSED7];
        for (let d = 0; d < 6; d++) {
          if (!((D2 >> d) & 1)) {
            out[CH.probe + d] = 0;
            continue;
          }
          const e = pt[nRow + d];
          const te = id * 6 + e + 1; // (fix 4) my probe, back across e
          out[CH.probe + d] = (met >> e) & 1 ? te : inV[e] !== te ? inV[e] : 0;
          const loop = inV[d] === id * 6 + d + 1 || inV[e] === te;
          if (loop || ((C | cross) >> d) & 1 || ((C | cross) >> e) & 1) closed2 |= 1 << d; // (fix 5) cross(pt(d))
        }
      } else {
        for (let d = 0; d < 6; d++) out[CH.probe + d] = 0;
      }

      out[CH.rule] = nr;
      out[CH.owner] = no;
      out[CH.D] = D2;
      out[CH.T] = T2;
      out[CH.W] = W2;
      out[CH.H] = H2;
      out[CH.fuel] = fuel2;
      out[CH.closed] = closed2;
    },
  };
}

let strip: Board | null = null;
/** One cell of every geo, in a row: what `RuleTable.exits` renders a rule's whole chord table from. */
function geoStrip(): Board {
  strip ??= new Board({ level: 0, root: 'geo', h: 1, w: GEOS, mirror: 1, tiles: GEOS, geo: Array.from({ length: GEOS }, (_, g) => g) });
  return strip;
}

export class LineCA {
  readonly ca: CA;
  readonly board: Board;
  readonly table: RuleTable;
  readonly knobs: LineKnobs;
  /** Views on ca.ch, indexed by board cell (length n + 1: slot n is the off-board sentinel). `fuel` is an addition. */
  readonly ch: {
    rule: Int32Array; owner: Int32Array; D: Int32Array; T: Int32Array; W: Int32Array; H: Int32Array;
    closed: Int32Array; fuel: Int32Array;
  };
  /** Board cells; slot n is the dead slot every off-board neighbour points at. */
  readonly n: number;
  /** ownerOf[rule index]; rules[rule index]: the strand rule (index 0: none). */
  readonly ownerOf: number[] = [0];
  readonly rules: (Rule | null)[] = [null];
  /** Tests only: step with the engine's reference sweep (`stepFull`) instead of the active set. */
  full = false;

  private readonly sh: Shared;
  /** Whose go was on last step: nobody goes two steps running. */
  private readonly lastGo = new Uint8Array(MAX_OWNER + 1);
  private readonly nbr: Int32Array;
  // the cells holding tips (maintained from the changed slots; may hold stale entries until compacted)
  private readonly tipCells: Int32Array;
  private readonly inTips: Uint8Array;
  private tipCount = 0;
  // cells the host wrote since the last step
  private readonly dirty: number[] = [];
  private readonly isDirty: Uint8Array;
  // D and owner as last reported by changed()
  private readonly shD: Int32Array;
  private readonly shOwner: Int32Array;
  private readonly changedBuf: Int32Array;
  private changedCount = 0;

  constructor(board: Board, table: RuleTable, knobs?: Partial<LineKnobs>) {
    this.board = board;
    this.table = table;
    this.knobs = { ...DEFAULT_LINE_KNOBS, ...knobs };
    const k = this.knobs;
    if (!Number.isInteger(k.period) || k.period < 2) {
      throw new Error(`period ${k.period}: an integer ≥ 2 (at a chord a step a fleeing tip outruns its wipe wave)`);
    }
    if (!Number.isInteger(k.fuel) || k.fuel < 0 || k.fuel > 255) throw new Error(`fuel ${k.fuel}: 0..255`);
    if (!Number.isInteger(k.maxTips) || k.maxTips < 0) throw new Error(`maxTips ${k.maxTips}`);
    const n = board.n;
    this.n = n;
    const size = n + 1;
    const nbr = new Int32Array(size * 6).fill(n);
    for (let i = 0; i < n * 6; i++) if (board.nbr[i] >= 0) nbr[i] = board.nbr[i];
    this.nbr = nbr;
    const isField = new Uint8Array(size);
    isField.fill(1, 0, n);
    const topo: Topology = { size, cells: Int32Array.from({ length: n }, (_, i) => i), nbr, isField };
    const geo = new Int32Array(size).fill(-1);
    const id = new Int32Array(size).fill(-1);
    for (let i = 0; i < n; i++) {
      geo[i] = board.geo[board.pos[i]];
      id[i] = i;
    }
    this.sh = { partner: new Int8Array(16 * G6).fill(-1), go: new Uint8Array(MAX_OWNER + 1), fuelMax: k.fuel, probe: k.probe };
    this.ca = new CA(topo, lineRule(this.sh), { N: n, CAP: 0, K: 0, wallsBound: 0 }, { geo, id });
    const c = this.ca.ch;
    this.ch = {
      rule: c[CH.rule], owner: c[CH.owner], D: c[CH.D], T: c[CH.T], W: c[CH.W], H: c[CH.H], closed: c[CH.closed],
      fuel: c[CH.fuel],
    };
    this.tipCells = new Int32Array(n);
    this.inTips = new Uint8Array(n);
    this.isDirty = new Uint8Array(n);
    this.shD = new Int32Array(n);
    this.shOwner = new Int32Array(n);
    this.changedBuf = new Int32Array(n);
  }

  /** A rule for `owner` (1..15) → its index ≥ 1, with partner[ix][geo][d] built. The same rule and owner again: the same index. */
  addRule(rule: Rule, owner: number): number {
    if (!Number.isInteger(owner) || owner < 1 || owner > MAX_OWNER) throw new Error(`owner ${owner}: 1..${MAX_OWNER}`);
    const key = this.table.key(rule);
    for (let i = 1; i < this.rules.length; i++) {
      if (this.ownerOf[i] === owner && this.table.key(this.rules[i]!) === key) return i;
    }
    const ix = this.rules.length;
    if (ix > MAX_RULES) throw new Error(`more than ${MAX_RULES} rules`);
    if ((ix + 1) * G6 > this.sh.partner.length) {
      const grown = new Int8Array(Math.min(MAX_RULES + 1, 2 * (ix + 1)) * G6).fill(-1);
      grown.set(this.sh.partner);
      this.sh.partner = grown;
    }
    const ex = this.table.exits(rule, geoStrip());
    const row = this.sh.partner;
    for (let g = 0; g < GEOS; g++) for (let d = 0; d < 6; d++) row[(ix * GEOS + g) * 6 + d] = ex[d * GEOS + g];
    this.rules.push({ s: rule.s, digits: rule.digits.slice() });
    this.ownerOf.push(owner);
    return ix;
  }

  /** partner[rule][geo of cell][d]: the edge a line of `rule` entering `cell` across d leaves by, -1 none. */
  pt(rule: number, cell: number, d: number): number {
    if (!(rule >= 1 && rule < this.rules.length) || !(cell >= 0 && cell < this.n) || !(d >= 0 && d < 6)) return -1;
    return this.sh.partner[(rule * GEOS + this.board.geo[this.board.pos[cell]]) * 6 + d];
  }

  /** The host's tap, between steps: Spectacle's refusals in its order, then the tap's chord and tips. */
  tap(req: TapRequest, blocked?: (cell: number) => boolean): Refusal | null {
    const { cell, d0, d1, rule } = req;
    if (this.pt(rule, cell, d0) !== d1) return 'no chord';
    const ch = this.ch;
    const r = ch.rule[cell];
    if (r !== 0 && r !== rule) return 'rival tile';
    if (r === 0 && ch.W[cell] !== 0) return 'hot'; // a crash tile, this step
    if ((ch.D[cell] >> d0) & 1) return 'drawn';
    if ((ch.W[cell] >> d0) & 1 || (ch.W[cell] >> d1) & 1 || this.waveAt(cell, d0, rule) || this.waveAt(cell, d1, rule)) return 'hot';
    const owner = this.ownerOf[rule];
    const fuel = this.knobs.fuel;
    const tips = (1 << d1) | (this.knobs.oneWay ? 0 : 1 << d0);
    let add = 0;
    for (let d = 0; d < 6; d++) if ((tips >> d) & 1 && this.live(cell, d, rule, fuel)) add++;
    if (this.countTips(owner) + add > this.knobs.maxTips) return 'heads';
    if (blocked?.(cell)) return 'territory';
    this.write(cell, rule, owner, ch.D[cell] | (1 << d0) | (1 << d1), ch.T[cell] | tips, fuel);
    return null;
  }

  /**
   * Restart growth from a drawn chord of your own (Spectacle's tapOwnLine → turnRound / setBack): a tip on each of
   * its loose ends that has somewhere to go, within maxTips; refuels it under bounded growth. Whether any was set.
   */
  retap(cell: number, d0: number, d1: number, rule: number): boolean {
    if (this.pt(rule, cell, d0) !== d1) return false;
    const ch = this.ch;
    if (ch.rule[cell] !== rule || !((ch.D[cell] >> d0) & 1)) return false;
    const owner = this.ownerOf[rule];
    const fuel = this.knobs.fuel;
    let have = this.countTips(owner);
    let set = 0;
    for (const d of [d1, d0]) {
      if ((ch.T[cell] >> d) & 1 || have >= this.knobs.maxTips || !this.live(cell, d, rule, fuel)) continue;
      set |= 1 << d;
      have++;
    }
    if (set === 0) return false;
    this.write(cell, rule, owner, ch.D[cell], ch.T[cell] | set, fuel > 0 ? fuel : ch.fuel[cell]);
    return true;
  }

  /**
   * One synchronous step. `go[owner]` (owners 0..15): whose tiles draw this step; default: all on steps ≡ 0 mod
   * period. An owner who went last step does not go this one (a line grows at most a chord every two steps).
   */
  step(go?: Uint8Array): void {
    const g = this.sh.go;
    if (go) {
      g.fill(0);
      g.set(go.length > g.length ? go.subarray(0, g.length) : go);
    } else g.fill((this.ca.generation + 1) % this.knobs.period === 0 ? 1 : 0);
    for (let o = 0; o < g.length; o++) {
      if (this.lastGo[o]) g[o] = 0;
      this.lastGo[o] = g[o] ? 1 : 0;
    }
    this.compactTips();
    if (this.full) this.ca.stepFull();
    else {
      // `go` is the one input that changes without a state change: list the tile ahead of every tip that goes.
      const { T, owner } = this.ch;
      for (let i = 0; i < this.tipCount; i++) {
        const c = this.tipCells[i];
        if (!g[owner[c]]) continue;
        const t = T[c];
        for (let d = 0; d < 6; d++) if ((t >> d) & 1) this.ca.touch(this.nbr[c * 6 + d]);
      }
      this.ca.step();
    }
    this.afterStep();
  }

  /** Cells whose D or owner changed in the last step (the host's writes before it included); a copy. */
  changed(): Int32Array {
    return this.changedBuf.slice(0, this.changedCount);
  }

  /** The owner's live tips: tip bits whose tile ahead is on the board, has a chord for it and has not drawn it yet. */
  countTips(owner: number): number {
    const { T, owner: own, rule, fuel } = this.ch;
    let n = 0;
    for (let i = 0; i < this.tipCount; i++) {
      const c = this.tipCells[i];
      const t = T[c];
      if (t === 0 || own[c] !== owner) continue;
      for (let d = 0; d < 6; d++) if ((t >> d) & 1 && this.live(c, d, rule[c], fuel[c])) n++;
    }
    return n;
  }

  /** The owner's line tiles: cells of theirs with a chord drawn. */
  lineTiles(owner: number): number {
    const { D, owner: own } = this.ch;
    let k = 0;
    for (let c = 0; c < this.n; c++) if (D[c] !== 0 && own[c] === owner) k++;
    return k;
  }

  /** Union-find over the drawn chords: every line and loop, each a stretch of one strand. */
  components(): Component[] {
    const { D, rule, owner } = this.ch;
    const n = this.n;
    const parent = new Int32Array(n * 6).fill(-1);
    const find = (x: number): number => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const union = (a: number, b: number): void => {
      const x = find(a), y = find(b);
      if (x !== y) parent[Math.max(x, y)] = Math.min(x, y);
    };
    for (let c = 0; c < n; c++) for (let d = 0; d < 6; d++) if ((D[c] >> d) & 1) parent[c * 6 + d] = c * 6 + d;
    const isLoose = new Uint8Array(n * 6);
    for (let c = 0; c < n; c++) {
      const dc = D[c];
      if (dc === 0) continue;
      for (let d = 0; d < 6; d++) {
        if (!((dc >> d) & 1)) continue;
        const p = this.pt(rule[c], c, d);
        if (p < 0 || !((dc >> p) & 1)) throw new Error(`cell ${c}: chord end ${d} without its partner ${p}`);
        union(c * 6 + d, c * 6 + p);
        const m = this.board.nbr[c * 6 + d];
        if (m >= 0 && rule[m] === rule[c] && (D[m] >> OPP[d]) & 1) union(c * 6 + d, m * 6 + OPP[d]);
        else isLoose[c * 6 + d] = 1;
      }
    }
    const byRoot = new Map<number, { ends: number; ends2: number; cells: Set<number> }>();
    for (let x = 0; x < n * 6; x++) {
      if (parent[x] < 0) continue;
      const r = find(x);
      let g = byRoot.get(r);
      if (!g) byRoot.set(r, (g = { ends: 0, ends2: 0, cells: new Set() }));
      g.ends2++; // chord ends: chords = ends2 / 2
      g.ends += isLoose[x];
      g.cells.add(Math.floor(x / 6));
    }
    const out: Component[] = [];
    for (const [root, g] of [...byRoot].sort((a, b) => a[0] - b[0])) {
      const c = Math.floor(root / 6);
      out.push({
        owner: owner[c], rule: rule[c], cells: Int32Array.from([...g.cells].sort((a, b) => a - b)), ends: g.ends,
        closed: g.ends === 0, chords: g.ends2 / 2,
      });
    }
    return out;
  }

  // ---- internals

  /**
   * A wipe wave about to cross edge d into `cell` along `rule` (the neighbour's W pointing here, of the rule or a
   * crash's): a chord tapped there would be taken by it next step, as the continuation of the dying line. Such a
   * tap is refused as 'hot' (the doc checks only the tile's own W; this is the same race one tile earlier).
   */
  private waveAt(cell: number, d: number, rule: number): boolean {
    const m = this.board.nbr[cell * 6 + d];
    if (m < 0 || !((this.ch.W[m] >> OPP[d]) & 1)) return false;
    return this.ch.rule[m] === rule || this.ch.rule[m] === 0;
  }

  /** Whether a tip at (cell, d) under `rule` can still move: the tile ahead exists, has its chord, hasn't drawn it. */
  private live(cell: number, d: number, rule: number, fuel: number): boolean {
    if (this.knobs.fuel > 0 && fuel === 0) return false;
    const m = this.board.nbr[cell * 6 + d];
    if (m < 0 || this.pt(rule, m, OPP[d]) < 0) return false;
    return !(this.ch.rule[m] === rule && (this.ch.D[m] >> OPP[d]) & 1);
  }

  private write(cell: number, rule: number, owner: number, D: number, T: number, fuel: number): void {
    const ca = this.ca;
    ca.writeState('rule', cell, rule);
    ca.writeState('owner', cell, owner);
    ca.writeState('D', cell, D);
    ca.writeState('T', cell, T);
    ca.writeState('fuel', cell, fuel);
    if (!this.isDirty[cell]) {
      this.isDirty[cell] = 1;
      this.dirty.push(cell);
    }
    this.noteTip(cell);
  }

  private noteTip(c: number): void {
    if (this.ch.T[c] !== 0 && !this.inTips[c]) {
      this.inTips[c] = 1;
      this.tipCells[this.tipCount++] = c;
    }
  }

  private compactTips(): void {
    const T = this.ch.T;
    let k = 0;
    for (let i = 0; i < this.tipCount; i++) {
      const c = this.tipCells[i];
      if (T[c] !== 0) this.tipCells[k++] = c;
      else this.inTips[c] = 0;
    }
    this.tipCount = k;
  }

  private afterStep(): void {
    const { D, owner } = this.ch;
    let k = 0;
    const visit = (c: number): void => {
      if (c >= this.n) return;
      this.noteTip(c);
      if (D[c] !== this.shD[c] || owner[c] !== this.shOwner[c]) {
        this.shD[c] = D[c];
        this.shOwner[c] = owner[c];
        this.changedBuf[k++] = c;
      }
    };
    for (const c of this.dirty) {
      this.isDirty[c] = 0;
      visit(c);
    }
    this.dirty.length = 0;
    const slots = this.ca.lastChanged();
    for (let i = 0; i < slots.length; i++) visit(slots[i]);
    this.changedCount = k;
  }
}
