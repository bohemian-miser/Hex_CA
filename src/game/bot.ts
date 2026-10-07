// A bot tapper for the hybrid game (docs/spectacle-ca-hybrid.md §4: "1-4 players, a bot tapper"): a player who
// taps through the same `Game.tap` a person does, at a board point, now and then. It knows its own rule's strands
// (src/strand.ts's walker, the same chords the line CA grows) and picks the tap whose strand looks worth most: the
// free tiles it would draw plus the area it would wall off (what a loop encloses, the smaller side of a line from
// edge to edge: what the flood fills), and keeps clear of rivals' lines (a hit wipes both) unless it is behind, when
// it sometimes aims at one on purpose. Deterministic given its `rand`.

import { allStrands, type Strand } from '../strand.js';
import { edgeMid, unitCentre } from '../draw.js';
import type { Game, Refusal } from './host.js';

export interface BotOptions {
  /** Game ms between its looks at the board (jittered ±40%). */
  everyMs: number;
  /** Candidate taps it weighs per look. */
  tries: number;
}

export const DEFAULT_BOT: BotOptions = { everyMs: 900, tries: 40 };

interface Plan { strand: number; cell: number; d0: number; d1: number }

export class Bot {
  readonly opts: BotOptions;
  private next = 0;
  private strands: Strand[] = [];
  /** strandAt[cell * 6 + d] = the strand through edge d of the cell (-1 none), for the current rule. */
  private strandAt = new Int32Array(0);
  /** cells[k]: strand k's tiles; area[k]: the tiles it walls off on an empty board (src/nca.ts targets()'s
   * rule: every region the strand's tiles leave but the largest one touching the rim). */
  private cells: Int32Array[] = [];
  private area: number[] = [];
  private ruleKey = '';

  constructor(readonly game: Game, readonly owner: number, private readonly rand: () => number, opts: Partial<BotOptions> = {}) {
    this.opts = { ...DEFAULT_BOT, ...opts };
  }

  /** Called with the game time as the game runs: maybe a tap. What the tap came to, or undefined for none. */
  act(now: number): Refusal | null | undefined {
    if (now < this.next) return undefined;
    this.next = now + this.opts.everyMs * (0.6 + 0.8 * this.rand());
    const g = this.game;
    if (g.respawnLeft(this.owner) > 0) return undefined;
    if (g.lines.countTips(this.owner) + 2 > g.knobs.maxTips) return undefined; // a line still growing
    this.learn();
    const plan = this.choose();
    if (!plan) return undefined;
    // A point on the chord, a little towards d1, so the host's nearest-chord pick lands on it, d1 first.
    const p = g.board.pos[plan.cell];
    const [cx, cy] = unitCentre(p % g.board.w, Math.floor(p / g.board.w));
    const [ax, ay] = edgeMid(cx, cy, plan.d0, 1);
    const [bx, by] = edgeMid(cx, cy, plan.d1, 1);
    return g.tap(this.owner, ax + (bx - ax) * 0.6, ay + (by - ay) * 0.6);
  }

  /** The current rule's strands (again when the rule changes). */
  private learn(): void {
    const g = this.game;
    const pl = g.player(this.owner);
    const key = g.table.describe(pl.rule);
    if (key === this.ruleKey) return;
    this.ruleKey = key;
    const b = g.board;
    this.strands = allStrands(g.table.exits(pl.rule, b), b);
    this.strandAt = new Int32Array(b.n * 6).fill(-1);
    this.cells = [];
    this.area = [];
    const wall = new Uint8Array(b.n);
    this.strands.forEach((st, k) => {
      const cells = st.rows.map((row, j) => b.cellOf[row * b.w + st.cols[j]]);
      cells.forEach((c, j) => {
        this.strandAt[c * 6 + st.ins[j]] = k;
        this.strandAt[c * 6 + st.outs[j]] = k;
      });
      const own = Int32Array.from(new Set(cells));
      this.cells.push(own);
      for (const c of own) wall[c] = 1;
      this.area.push(walledOff(b.n, b.nbr, wall));
      for (const c of own) wall[c] = 0;
    });
  }

  /** The best of `tries` random free taps of its rule, or null. */
  private choose(): Plan | null {
    const g = this.game;
    const { rule, W } = g.lines.ch;
    const terr = g.territory();
    const scores = g.scores();
    let best = -Infinity;
    for (const p of g.seated()) if (p.owner !== this.owner) best = Math.max(best, scores[p.owner]);
    const behind = best > scores[this.owner] + 10;
    const aggressive = behind && this.rand() < 0.35;
    let pick: Plan | null = null;
    let top = 0;
    const n = g.board.n;
    for (let k = 0; k < this.opts.tries; k++) {
      const cell = Math.floor(this.rand() * n);
      if (rule[cell] !== 0 || W[cell] !== 0 || (terr[cell] !== 0 && terr[cell] !== this.owner)) continue;
      const chords = g.chordsOf(this.owner, cell);
      if (!chords.length) continue;
      const [d0, d1] = chords[Math.floor(this.rand() * chords.length)];
      const s = this.strandAt[cell * 6 + d0];
      if (s < 0) continue;
      const v = this.value(s, aggressive) * (0.85 + 0.3 * this.rand());
      if (v > top) {
        top = v;
        pick = { strand: s, cell, d0, d1 };
      }
    }
    return pick;
  }

  /** What a strand would be worth drawn: its free tiles, weighted by what it is, less the cost of rivals on it. */
  private value(s: number, aggressive: boolean): number {
    const g = this.game;
    const { rule, owner } = g.lines.ch;
    let free = 0;
    let rival = 0;
    for (const c of this.cells[s]) {
      if (rule[c] === 0) free++;
      else if (owner[c] !== this.owner) rival++;
    }
    if (!free) return 0;
    const worth = free + 1.5 * this.area[s];
    if (!rival) return worth;
    return aggressive ? worth + 20 * rival : worth / (1 + 3 * rival);
  }
}

/** How many free cells walls cut off: every 6-connected region of non-wall cells but the largest one that touches
 * the rim (a region off the rim is enclosed; of the rim regions, all but the biggest are the smaller sides). */
function walledOff(n: number, nbr: Int32Array, wall: Uint8Array): number {
  const seen = new Uint8Array(n);
  const stack: number[] = [];
  let total = 0;
  let biggestRim = 0;
  for (let c0 = 0; c0 < n; c0++) {
    if (wall[c0] || seen[c0]) continue;
    seen[c0] = 1;
    stack.push(c0);
    let size = 0;
    let rim = false;
    while (stack.length) {
      const c = stack.pop()!;
      size++;
      for (let d = 0; d < 6; d++) {
        const x = nbr[c * 6 + d];
        if (x < 0) rim = true;
        else if (!wall[x] && !seen[x]) {
          seen[x] = 1;
          stack.push(x);
        }
      }
    }
    total += size;
    if (rim && size > biggestRim) biggestRim = size;
  }
  return total - biggestRim;
}
