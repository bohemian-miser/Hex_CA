// The referee of the hybrid game (docs/spectacle-ca-hybrid.md §2.4, §3.4, §4.1): players and their rules, taps
// (board point → tile → the rule's nearest free chord, Spectacle's refusals in Spectacle's order), the clock
// (line steps every `stepMs`, flood steps behind them), respawn after a hit, the readouts (territory and scores)
// and the event list the page draws sparks and the log from. The line CA and the area layer do the work; the host
// only decides what goes in and reads what comes out. Pure: no DOM, deterministic given the times it is handed.
//
// Board units: tile (row, col) of the strand Board is axial (q, r) = (col, row) with circumradius 1, so its centre
// is (√3·(col + row/2), 1.5·row) — src/draw.ts's unitCentre.

import { Board, PAIRS, RuleTable, type Rule } from '../strand.js';
import type { NCAWeights } from '../nca.js';
import { axialAt, edgeMid, rankChords, unitCentre } from '../draw.js';
import { DEFAULT_LINE_KNOBS, LineCA, type LineKnobs, type Refusal } from './line-ca.js';
import { cpuArea, type AreaLayer } from './area.js';

export type { LineKnobs, Refusal };

export interface GameKnobs extends LineKnobs {
  /** Score = line tiles + the tiles your flood fills for you alone (§0 decision 2); off: line tiles only. */
  scoreFill: boolean;
  /** After a hit, the players in it can't tap for this long (game ms). */
  respawnMs: number;
  /** Game ms per line step. */
  stepMs: number;
  /** Flood steps per line step (a page may run fewer when its frame budget is spent: `tick`'s budget). */
  floodPerStep: number;
  /** Line steps between readouts of territory and scores. */
  scoreEvery: number;
  /** v1.1 (§3.4): a rival's line wholly inside your fill, for `convertPerR` × R flood steps, turns into your rule's
   * chords on its tiles. Not in the doc's interface; an addition. */
  convert: boolean;
  /** How long a line must stay inside before it converts, in flood steps per unit of the flood's radius. */
  convertPerR: number;
}

export const DEFAULT_GAME_KNOBS: GameKnobs = {
  ...DEFAULT_LINE_KNOBS, scoreFill: true, respawnMs: 500, stepMs: 50, floodPerStep: 1, scoreEvery: 1,
  convert: true, convertPerR: 2,
};

export type GameEvent = {
  /** Game ms (the time handed to the last `tick`). */
  t: number;
  kind: 'tap' | 'refused' | 'hit' | 'tail' | 'closed' | 'convert';
  owner: number;
  cell: number;
  why?: Refusal;
};

/** Seats when the host builds its own area layer (one flood each; a seat with no lines costs nothing). */
export const DEFAULT_SEATS = 8;
/** At most this many line steps per tick: a stalled tab drops the backlog rather than racing to catch up. */
export const MAX_STEPS_PER_TICK = 8;
/** The flood is left alone once nothing has changed for this many of its steps per unit of the frame's radius:
 * settled (the fill page's evaluation reads at 16 R), so stepping on would only burn CPU. */
export const SETTLE_PER_R = 16;

export interface Player {
  owner: number;
  name: string;
  rule: Rule;
  /** The line CA's index for `rule` (a rule change adds another; older lines keep theirs). */
  ruleIx: number;
  /** The rule's chords per board position (RuleTable.render). */
  chords: Int16Array;
  /** Game ms before which a tap is refused ('respawn'). */
  respawnAt: number;
  bot: boolean;
}

const opp = (d: number) => (d + 3) % 6;
const UNIT = { size: 1, ox: 0, oy: 0 };

export class Game {
  readonly lines: LineCA;
  readonly area: AreaLayer;
  readonly knobs: GameKnobs;
  /** By owner (1..area.owners); index 0 unused. */
  readonly players: (Player | undefined)[] = [];
  /** Line steps and flood steps run so far. */
  steps = 0;
  floodSteps = 0;
  /** The flood's radius in its own frame (for the settle rule). */
  readonly floodR: number;

  /** By line-CA rule index: whose it is, the rule, and its chords per board position (for drawing a cell's
   * drawn edges as chords). */
  private readonly ixInfo: ({ owner: number; rule: Rule; chords: Int16Array } | undefined)[] = [];
  private events: GameEvent[] = [];
  private now = 0;
  private last: number | null = null;
  private acc = 0;
  private floodDue = 0;
  /** Flood steps since a wall last changed. */
  private quiet = 0;
  private sinceRead = 0;
  /** The territory shown (two-read hysteresis on fill tiles), and the last two raw reads it is made from: the
   * latest, and the one before it that a flood step separates from it. */
  private terr: Int8Array;
  private rawNow: Int8Array;
  private rawBefore: Int8Array;
  private rawAt = -1;
  private readonly wasLine: Uint8Array;
  /** Per line cell: the one rival whose fill covers it (0 none), and the flood step that began. */
  private readonly insideOf: Int8Array;
  private readonly insideAt: Float64Array;
  private score: Int32Array;
  private readonly prevT: Int32Array;
  private readonly prevClosed: Int32Array;

  constructor(readonly board: Board, readonly table: RuleTable, weights: NCAWeights, area?: AreaLayer,
    knobs: Partial<GameKnobs> = {}) {
    this.knobs = { ...DEFAULT_GAME_KNOBS, ...knobs };
    const k = this.knobs;
    this.lines = new LineCA(board, table, { period: k.period, oneWay: k.oneWay, fuel: k.fuel, probe: k.probe, maxTips: k.maxTips });
    this.area = area ?? cpuArea(board, weights, DEFAULT_SEATS);
    this.floodR = (Math.max(board.h, board.w) + 2) >> 1;
    this.terr = new Int8Array(board.n);
    this.rawNow = new Int8Array(board.n);
    this.rawBefore = new Int8Array(board.n);
    this.wasLine = new Uint8Array(board.n);
    this.insideOf = new Int8Array(board.n);
    this.insideAt = new Float64Array(board.n);
    this.score = new Int32Array(this.area.owners + 1);
    this.prevT = new Int32Array(board.n);
    this.prevClosed = new Int32Array(board.n);
  }

  // ── Players ──────────────────────────────────────────────────────────────

  /** A new player with `rule`: their owner number (1..area.owners), or -1 if another player holds that rule or
   * every seat is taken. */
  addPlayer(name: string, rule: Rule, bot = false): number {
    if (this.holder(rule) > 0) return -1;
    let owner = 1;
    while (owner <= this.area.owners && this.players[owner]) owner++;
    if (owner > this.area.owners) return -1;
    const { ix, chords } = this.addRule(rule, owner);
    this.players[owner] = { owner, name, rule: { s: rule.s, digits: rule.digits.slice() }, ruleIx: ix, chords, respawnAt: -Infinity, bot };
    return owner;
  }

  private addRule(rule: Rule, owner: number): { ix: number; chords: Int16Array } {
    const ix = this.lines.addRule(rule, owner);
    const chords = this.table.render(rule, this.board);
    this.ixInfo[ix] = { owner, rule: { s: rule.s, digits: rule.digits.slice() }, chords };
    return { ix, chords };
  }

  /** A line-CA rule index's owner, rule and chords (undefined for 0 or an index the host never added). */
  ruleInfo(ix: number): { owner: number; rule: Rule; chords: Int16Array } | undefined {
    return this.ixInfo[ix];
  }

  /** The owner whose current rule is `rule`, or 0. */
  holder(rule: Rule): number {
    const key = this.table.describe(rule);
    for (const p of this.players) if (p && this.table.describe(p.rule) === key) return p.owner;
    return 0;
  }

  /** A player's rule changes (refused, false, if another player holds it): leaving and rejoining, as Spectacle's
   * setRule without its regrow — every line of theirs is wiped at once (to the line CA a new rule is a new rival
   * index, so old and new lines would cut each other), and their next taps use the new rule. */
  setRule(owner: number, rule: Rule): boolean {
    const p = this.player(owner);
    const h = this.holder(rule);
    if (h > 0) return h === owner;
    this.wipe(owner);
    const { ix, chords } = this.addRule(rule, owner);
    p.rule = { s: rule.s, digits: rule.digits.slice() };
    p.ruleIx = ix;
    p.chords = chords;
    return true;
  }

  /** Every line of a player's, gone at once (no waves, no hit: nobody's respawn is charged). */
  private wipe(owner: number): void {
    const { owner: own, D } = this.lines.ch;
    const cells: number[] = [];
    for (let c = 0; c < this.board.n; c++) if (own[c] === owner && D[c]) cells.push(c);
    if (!cells.length) return;
    const list = Int32Array.from(cells);
    this.lines.convert(list, 0);
    this.area.update(this.lines, list);
    this.quiet = 0;
    this.read();
  }

  player(owner: number): Player {
    const p = this.players[owner];
    if (!p) throw new RangeError(`no player ${owner}`);
    return p;
  }

  /** Every seated player, by owner. */
  seated(): Player[] {
    return this.players.filter((p): p is Player => !!p);
  }

  // ── Taps ─────────────────────────────────────────────────────────────────

  /** The tile under board point (x, y), -1 off the board. */
  cellAt(x: number, y: number): number {
    const [q, r] = axialAt(UNIT, x, y);
    return this.board.on(r, q) ? this.board.cellOf[r * this.board.w + q] : -1;
  }

  /** The player's rule's chords on a tile, as edge pairs. */
  chordsOf(owner: number, cell: number): [number, number][] {
    const bits = this.player(owner).chords[this.board.pos[cell]];
    const out: [number, number][] = [];
    for (let p = 0; p < PAIRS.length; p++) if ((bits >> p) & 1) out.push([PAIRS[p][0], PAIRS[p][1]]);
    return out;
  }

  /**
   * A tap at board point (x, y): the tile under it, the player's chord there nearest the point (falling back to
   * the next nearest free one, Spectacle's freeChord), oriented so the line heads out of the end nearer the point
   * (that end is the tip a one-way tap grows). A tap whose nearest chord is the player's own drawn line re-taps
   * it (`retap`: with `fuel`, the stuck ends grow on). Applied at once, between steps. null = accepted.
   */
  tap(owner: number, x: number, y: number): Refusal | null {
    const cell = this.cellAt(x, y);
    if (cell < 0) return this.refuse(owner, -1, 'no chord');
    const p = this.board.pos[cell];
    const [cx, cy] = unitCentre(p % this.board.w, Math.floor(p / this.board.w));
    const ranked = rankChords(this.chordsOf(owner, cell), cx, cy, 1, x, y).map(([a, b]) => {
      const [ax, ay] = edgeMid(cx, cy, a, 1);
      const [bx, by] = edgeMid(cx, cy, b, 1);
      return (ax - x) ** 2 + (ay - y) ** 2 < (bx - x) ** 2 + (by - y) ** 2 ? [b, a] as [number, number] : [a, b] as [number, number];
    });
    return this.tapRanked(owner, cell, ranked);
  }

  /** A tap of one given chord (d0 → d1: d1 is the one-way tip): what a bot or a test asks for. */
  tapChord(owner: number, cell: number, d0: number, d1: number): Refusal | null {
    return this.tapRanked(owner, cell, [[d0, d1]]);
  }

  private tapRanked(owner: number, cell: number, ranked: [number, number][]): Refusal | null {
    const pl = this.player(owner);
    if (this.now < pl.respawnAt) return this.refuse(owner, cell, 'respawn');
    if (!ranked.length) return this.refuse(owner, cell, 'no chord');
    const { ch } = this.lines;
    const [a, b] = ranked[0];
    if (ch.rule[cell] === pl.ruleIx && (ch.D[cell] >> a) & 1 && this.lines.retap(cell, a, b, pl.ruleIx)) {
      return this.accept(owner, cell);
    }
    const blocked = (c: number) => this.terr[c] !== 0 && this.terr[c] !== owner;
    let first: Refusal | null = null;
    for (const [d0, d1] of ranked) {
      const why = this.lines.tap({ cell, d0, d1, rule: pl.ruleIx }, blocked);
      if (why === null) return this.accept(owner, cell);
      if (why !== 'drawn' && why !== 'hot') return this.refuse(owner, cell, why);
      first ??= why;
    }
    return this.refuse(owner, cell, first!);
  }

  private accept(owner: number, cell: number): null {
    this.area.update(this.lines, Int32Array.of(cell));
    this.quiet = 0;
    this.events.push({ t: this.now, kind: 'tap', owner, cell });
    this.read(); // the tapped tile scores at once
    return null;
  }

  private refuse(owner: number, cell: number, why: Refusal): Refusal {
    this.events.push({ t: this.now, kind: 'refused', owner, cell, why });
    return why;
  }

  // ── The clock ────────────────────────────────────────────────────────────

  /**
   * Runs every line step due by game time `nowMs` (at most MAX_STEPS_PER_TICK; a longer gap is dropped), then
   * the flood steps they owe (`floodPerStep` each), then the readouts. `budgetMs`: stop the flood once this much
   * wall time has gone on this tick (a page's frame budget); what is left is owed to the next tick, up to a few
   * ticks' worth — a flood slower than the lines just lags. No budget: every step owed runs (tests, replays).
   */
  tick(nowMs: number, budgetMs?: number): void {
    const t0 = budgetMs === undefined ? 0 : performance.now();
    if (this.last === null) this.last = nowMs;
    this.acc += Math.max(0, nowMs - this.last);
    this.last = nowMs;
    this.now = nowMs;
    let n = 0;
    while (this.acc >= this.knobs.stepMs && n < MAX_STEPS_PER_TICK) {
      this.lineStep();
      this.acc -= this.knobs.stepMs;
      n++;
    }
    if (n === MAX_STEPS_PER_TICK) this.acc = Math.min(this.acc, this.knobs.stepMs);
    const ran = this.flood(budgetMs === undefined ? Infinity : budgetMs - (performance.now() - t0), t0);
    if ((n || ran) && this.sinceRead >= this.knobs.scoreEvery) this.read();
  }

  /** Exactly one line step and the flood steps it owes, then a readout (the page's Step button). */
  step(): void {
    this.lineStep();
    this.flood(Infinity, 0);
    this.read();
  }

  /** Whether the flood is still settling (it has steps owed, or a wall changed within its settle time). */
  settling(): boolean {
    return this.quiet < SETTLE_PER_R * this.floodR;
  }

  private lineStep(): void {
    const { ch } = this.lines;
    this.prevT.set(ch.T.subarray(0, this.board.n));
    this.prevClosed.set(ch.closed.subarray(0, this.board.n));
    this.lines.step();
    this.steps++;
    this.sinceRead++;
    const changed = this.lines.changed();
    if (changed.length) {
      this.area.update(this.lines, changed);
      this.quiet = 0;
    }
    this.floodDue = Math.min(this.floodDue + this.knobs.floodPerStep, this.knobs.floodPerStep * MAX_STEPS_PER_TICK);
    this.scan();
  }

  private flood(budgetMs: number, t0: number): boolean {
    let ran = false;
    while (this.floodDue >= 1) {
      if (!this.settling()) {
        this.floodDue = 0;
        break;
      }
      if (ran && budgetMs !== Infinity && performance.now() - t0 > budgetMs) break;
      this.area.step(1);
      this.floodDue--;
      this.floodSteps++;
      this.quiet++;
      ran = true;
    }
    return ran;
  }

  /** Hits (respawn), tails and newly closed loops in the step just run, as events. */
  private scan(): void {
    const { board } = this;
    const { ch } = this.lines;
    const n = board.n;
    const hitOwners = new Set<number>();
    for (let c = 0; c < n; c++) {
      if (ch.H[c]) {
        let o = ch.owner[c];
        if (o) hitOwners.add(o);
        else {
          // a crash on an empty tile: the owners of the tips that were aimed at it
          for (let d = 0; d < 6; d++) {
            const nb = board.nbr[c * 6 + d];
            if (nb >= 0 && (this.prevT[nb] >> opp(d)) & 1 && ch.owner[nb]) hitOwners.add((o = ch.owner[nb]));
          }
        }
        this.events.push({ t: this.now, kind: 'hit', owner: o, cell: c });
      }
      const gone = this.prevT[c] & ~ch.T[c];
      if (gone && !ch.H[c]) {
        for (let d = 0; d < 6; d++) {
          if (!((gone >> d) & 1) || !((ch.D[c] >> d) & 1)) continue;
          const nb = board.nbr[c * 6 + d];
          const onward = nb >= 0 && ch.rule[nb] === ch.rule[c] && (ch.D[nb] >> opp(d)) & 1;
          if (!onward) this.events.push({ t: this.now, kind: 'tail', owner: ch.owner[c], cell: c });
        }
      }
    }
    for (const o of hitOwners) {
      const p = this.players[o];
      if (p) p.respawnAt = this.now + this.knobs.respawnMs;
    }
    this.scanClosed();
  }

  /** One 'closed' event per loop that newly shows `closed`: a run of newly closed cells with no closed neighbour
   * along its chords from before (those are the flag still spreading round a loop already announced). */
  private scanClosed(): void {
    const { board } = this;
    const { ch } = this.lines;
    const seen = new Uint8Array(board.n);
    const fresh = (c: number) => ch.closed[c] !== 0 && this.prevClosed[c] === 0;
    for (let c0 = 0; c0 < board.n; c0++) {
      if (seen[c0] || !fresh(c0)) continue;
      let continued = false;
      const stack = [c0];
      seen[c0] = 1;
      while (stack.length) {
        const c = stack.pop()!;
        for (let d = 0; d < 6; d++) {
          if (!((ch.closed[c] >> d) & 1)) continue;
          const nb = board.nbr[c * 6 + d];
          if (nb < 0 || ch.rule[nb] !== ch.rule[c] || !((ch.D[nb] >> opp(d)) & 1)) continue;
          if (this.prevClosed[nb]) continued = true;
          else if (!seen[nb] && fresh(nb)) {
            seen[nb] = 1;
            stack.push(nb);
          }
        }
      }
      if (!continued) this.events.push({ t: this.now, kind: 'closed', owner: ch.owner[c0], cell: c0 });
    }
  }

  // ── Readouts ─────────────────────────────────────────────────────────────

  /**
   * Territory and scores. A line tile is its owner's at once; a tile the flood fills (or stops filling) changes
   * hands only when two reads a flood step apart agree (§3.4's hysteresis: a settling flood doesn't flicker the
   * score). Then v1.1's conversions.
   */
  private read(): void {
    this.sinceRead = 0;
    let raw = this.area.territory();
    if (this.knobs.convert && this.convertInside()) raw = this.area.territory();
    if (this.floodSteps !== this.rawAt) {
      [this.rawBefore, this.rawNow] = [this.rawNow, this.rawBefore];
      this.rawAt = this.floodSteps;
    }
    this.rawNow.set(raw);
    const { D, owner } = this.lines.ch;
    for (let c = 0; c < raw.length; c++) {
      const line = D[c] !== 0 && owner[c] > 0;
      if (line || this.wasLine[c] || raw[c] === this.rawBefore[c]) this.terr[c] = raw[c];
      this.wasLine[c] = line ? 1 : 0;
    }
    this.score.fill(0);
    if (this.knobs.scoreFill) {
      for (let c = 0; c < this.terr.length; c++) if (this.terr[c] > 0) this.score[this.terr[c]]++;
    } else {
      for (const p of this.seated()) this.score[p.owner] = this.lines.lineTiles(p.owner);
    }
  }

  /**
   * v1.1 conversion (§3.4, Spectacle's convertPath): every line (a component of drawn chords) whose tiles have all
   * been inside one rival's fill for convertPerR × R flood steps becomes that rival's: its chords go and the
   * rival's rule's chords are laid on its tiles, with no tips. One 'convert' event per tile. Whether any did.
   */
  private convertInside(): boolean {
    const { D, owner } = this.lines.ch;
    const n = this.board.n;
    const fills = this.seated().map((p) => [p.owner, this.area.fill(p.owner)] as const);
    let any = false;
    for (let c = 0; c < n; c++) {
      let p = 0;
      if (D[c] && owner[c]) {
        for (const [o, f] of fills) if (o !== owner[c] && f[c]) p = p ? -1 : o;
      }
      if (p !== this.insideOf[c]) {
        this.insideOf[c] = p;
        this.insideAt[c] = this.floodSteps;
      }
      if (p > 0) any = true;
    }
    if (!any) return false;
    const wait = this.knobs.convertPerR * this.floodR;
    let done = false;
    for (const comp of this.lines.components()) {
      const p = this.insideOf[comp.cells[0]];
      if (p <= 0 || !this.players[p]) continue;
      let ok = true;
      for (const c of comp.cells) {
        if (this.insideOf[c] !== p || this.floodSteps - this.insideAt[c] < wait) { ok = false; break; }
      }
      if (!ok) continue;
      this.lines.convert(comp.cells, this.players[p]!.ruleIx);
      this.area.update(this.lines, comp.cells);
      this.quiet = 0;
      for (const c of comp.cells) {
        this.insideOf[c] = 0;
        this.events.push({ t: this.now, kind: 'convert', owner: p, cell: c });
      }
      done = true;
    }
    return done;
  }

  /** Per board cell, as of the last readout: the owner of a line on it, else the one player whose fill covers
   * it (steadied: see `read`), else 0; -1 contested. */
  territory(): Int8Array {
    return this.terr;
  }

  /** Per owner (index 0 unused), as of the last readout: line tiles + sole fill (`scoreFill`), or line tiles. */
  scores(): Int32Array {
    return this.score;
  }

  /** Game ms the player still has to wait before a tap is taken (0 when free). */
  respawnLeft(owner: number): number {
    return Math.max(0, this.player(owner).respawnAt - this.now);
  }

  /** The time handed to the last tick. */
  get time(): number {
    return this.now;
  }

  /** The events since the last drain, oldest first. */
  drain(): GameEvent[] {
    const out = this.events;
    this.events = [];
    return out;
  }
}
