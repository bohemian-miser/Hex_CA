/**
 * The hybrid line CA (src/game/line-ca.ts, `LineCA`) on scripted taps, for the parity checks of
 * docs/spectacle-ca-hybrid.md §2.10 and §5 (D2):
 *
 *   npx tsx scripts/game-parity.ts [--in data/strand-v2/collide.json] [--out data/strand-v2/ca-collide.json]
 *       [--max-ticks 4000] [--quiet]
 *
 * The input is collide.json's format (`scripts/strand-export.ts --collide`: boards with `geo`, episodes with
 * `players` as [subset, digit_0 .. digit_8] and `taps` as {t, player, row, col, d0, d1}), or any file in it
 * (`nca/strand/ca_parity.py` writes sim.py's draw_taps episodes that way). Each episode is replayed as the
 * engine's was: CA step = engine tick, a tap at t is applied after the t-th step (the CA's generation t), then the
 * chords are read. The output mirrors the engine's record per episode: `taps` [{ok, reason}], `chords` [player,
 * row, col, a, b, on, off] (a < b; every interval a player's lines held the chord, in steps; off -1 = there at
 * rest), `settle` (the last step a chord changed) and `ticks` (stepped to: every tap applied, no tip and no wave
 * left). Knobs: §4.1's defaults but `maxTips` 99 (the engine's maxHeads 12 never binds on the fixture, the sim has
 * no limit). With the engine's chords in the input it prints the at-rest comparison: accepted taps, end state, per
 * tap whether its line survives, and the chords each holds at rest.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_LINE_KNOBS, LineCA, type LineKnobs } from '../src/game/line-ca.js';
import { Board, RuleTable, type BoardData, type StrandData } from '../src/strand.js';

export interface FixtureBoard {
  h: number;
  w: number;
  geo: number[];
  tile?: number[];
}
export interface FixtureTap {
  t: number;
  player: number;
  row: number;
  col: number;
  d0: number;
  d1: number;
  ok?: boolean;
  reason?: string;
}
/** [player, row, col, a, b, on, off]: a < b; off -1 = there at rest. */
export type Interval = [number, number, number, number, number, number, number];
export interface FixtureEpisode {
  board: string;
  players: number[][];
  taps: FixtureTap[];
  chords?: Interval[];
  settle?: number;
  ticks?: number;
}
export interface Fixture {
  boards: Record<string, FixtureBoard>;
  episodes: FixtureEpisode[];
  [k: string]: unknown;
}
export interface CaEpisode {
  board: string;
  taps: { ok: boolean; reason: string }[];
  chords: Interval[];
  settle: number;
  ticks: number;
}

export const PARITY_KNOBS: Partial<LineKnobs> = { maxTips: 99 };

/** A strand Board from a fixture board (level and root from its key "L<level>/<root>"). */
export function boardOf(key: string, b: FixtureBoard): Board {
  const m = /^L(\d+)\/(\d+)$/.exec(key);
  const data: BoardData = {
    level: m ? Number(m[1]) : 0,
    root: m ? m[2] : key,
    h: b.h,
    w: b.w,
    mirror: b.geo.some((g) => g >= 0 && g % 2 === 1) ? -1 : 1,
    tiles: b.geo.filter((g) => g >= 0).length,
    geo: b.geo,
    tile: b.tile,
  };
  return new Board(data);
}

/** The chords a player's lines hold on board cell i: keys "player,row,col,a,b" (a < b). */
export function cellChords(lca: LineCA, board: Board, exits: Int8Array[], playerOfRule: Map<number, number>,
  i: number): string[] {
  const bits = lca.ch.D[i];
  if (!bits) return [];
  const pl = playerOfRule.get(lca.ch.rule[i]);
  if (pl === undefined) throw new Error(`cell ${i}: chords under rule ${lca.ch.rule[i]}, which no player holds`);
  const p = board.pos[i];
  const row = Math.floor(p / board.w);
  const col = p % board.w;
  const HW = board.h * board.w;
  const out: string[] = [];
  for (let d = 0; d < 6; d++) {
    if (!((bits >> d) & 1)) continue;
    const e = exits[pl][d * HW + p];
    if (e < 0 || !((bits >> e) & 1)) throw new Error(`cell (${row}, ${col}): edge ${d} drawn without its chord's other end`);
    if (d < e) out.push(`${pl},${row},${col},${d},${e}`);
  }
  return out;
}

/** Every chord the players' lines hold now (a whole-board scan). */
export function chordsHeld(lca: LineCA, board: Board, exits: Int8Array[], playerOfRule: Map<number, number>): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i < board.n; i++) for (const k of cellChords(lca, board, exits, playerOfRule, i)) out.add(k);
  return out;
}

/** No tip left and nothing changed: every chord is final (a crash's or a wave's W pulse changes D next step). */
function quiet(lca: LineCA, owners: number, n: number): boolean {
  if (lca.changed().length) return false;
  for (let o = 1; o <= owners; o++) if (lca.countTips(o)) return false;
  const { T, W } = lca.ch;
  for (let i = 0; i < n; i++) if (W[i] || T[i]) return false;
  return true;
}

/** Replay one episode through a fresh LineCA. */
export function replay(table: RuleTable, board: Board, ep: FixtureEpisode, knobs: Partial<LineKnobs> = PARITY_KNOBS,
  maxTicks = 4000): CaEpisode {
  const lca = new LineCA(board, table, knobs);
  const ruleIx: number[] = [];
  const playerOfRule = new Map<number, number>();
  const exits: Int8Array[] = [];
  ep.players.forEach((p, pl) => {
    const rule = { s: p[0], digits: p.slice(1) };
    const ix = lca.addRule(rule, pl + 1);
    ruleIx.push(ix);
    playerOfRule.set(ix, pl);
    exits.push(table.exits(rule, board));
  });
  const taps = [...ep.taps].map((x, k) => ({ ...x, k })).sort((a, b) => a.t - b.t || a.k - b.k);
  const tapsOut = ep.taps.map(() => ({ ok: false, reason: '' }));
  const held: string[][] = Array.from({ length: board.n }, () => []);
  const open = new Map<string, number>();
  const intervals: Interval[] = [];
  const close = (k: string, on: number, off: number): void => {
    intervals.push([...(k.split(',').map(Number) as [number, number, number, number, number]), on, off]);
  };
  let settle = 0;
  let ti = 0;
  let tick = 0;
  // changed() reports cells whose D or owner differs from what it last reported, so a tap undone by the next step
  // is not in it: tapped cells are read again after that step
  let tapped: number[] = [];
  for (; tick <= maxTicks; tick++) {
    const dirty: number[] = tapped;
    tapped = [];
    if (tick > 0) {
      lca.step();
      dirty.push(...lca.changed());
    }
    for (; ti < taps.length && taps[ti].t <= tick; ti++) {
      const x = taps[ti];
      const cell = board.cellOf[x.row * board.w + x.col];
      const why = cell < 0 ? 'off the board' : lca.tap({ cell, d0: x.d0, d1: x.d1, rule: ruleIx[x.player] });
      tapsOut[x.k] = { ok: why === null, reason: why ?? '' };
      if (why === null) {
        dirty.push(cell);
        tapped.push(cell);
      }
    }
    for (const i of dirty) {
      const now = cellChords(lca, board, exits, playerOfRule, i);
      for (const k of held[i]) {
        if (now.includes(k)) continue;
        close(k, open.get(k)!, tick);
        open.delete(k);
        settle = tick;
      }
      for (const k of now) {
        if (open.has(k)) continue;
        open.set(k, tick);
        settle = tick;
      }
      held[i] = now;
    }
    if (ti >= taps.length && quiet(lca, ep.players.length, board.n)) break;
  }
  if (tick > maxTicks) throw new Error(`still growing after ${maxTicks} steps`);
  const all = chordsHeld(lca, board, exits, playerOfRule);
  if (all.size !== open.size || [...all].some((k) => !open.has(k))) throw new Error('changed() missed a cell');
  for (const [k, on] of open) close(k, on, -1);
  intervals.sort((x, y) => x[5] - y[5] || x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || x[3] - y[3]);
  return { board: ep.board, taps: tapsOut, chords: intervals, settle, ticks: tick };
}

/** The chords held at rest: keys "player,row,col,a,b". */
export function restOf(chords: readonly Interval[]): Set<string> {
  return new Set(chords.filter((c) => c[6] < 0).map((c) => c.slice(0, 5).join(',')));
}

/** Per tap: accepted, and its tapped chord still its player's at rest. */
export function survivors(taps: readonly FixtureTap[], ok: readonly boolean[], rest: Set<string>): boolean[] {
  return taps.map((x, k) => ok[k] && rest.has(`${x.player},${x.row},${x.col},${Math.min(x.d0, x.d1)},${Math.max(x.d0, x.d1)}`));
}

export function sameSet(a: Set<string>, b: Set<string>): boolean {
  if (a.size !== b.size) return false;
  for (const k of a) if (!b.has(k)) return false;
  return true;
}

export interface Comparison {
  episodes: number;
  taps: number;
  tapsSame: number;
  endSame: number;
  lines: number;
  survive: number;
  surviveSame: number;
  /** Chords held at rest: by the engine, by the CA, by both (summed over episodes). */
  engineChords: number;
  caChords: number;
  bothChords: number;
  /** Episodes whose accepted taps or end state differ from the engine's. */
  differ: number[];
}

/** The CA against the engine at rest (the fixture's own `chords` and `ok`). */
export function compare(fx: Fixture, ca: CaEpisode[]): Comparison {
  const c: Comparison = {
    episodes: 0, taps: 0, tapsSame: 0, endSame: 0, lines: 0, survive: 0, surviveSame: 0,
    engineChords: 0, caChords: 0, bothChords: 0, differ: [],
  };
  fx.episodes.forEach((e, i) => {
    if (!e.chords) return;
    const r = ca[i];
    c.episodes++;
    const engOk = e.taps.map((x) => !!x.ok);
    const caOk = r.taps.map((x) => x.ok);
    const engRest = restOf(e.chords);
    const caRest = restOf(r.chords);
    const tapsSame = engOk.every((v, k) => v === caOk[k]);
    const endSame = sameSet(engRest, caRest);
    c.taps += e.taps.length;
    c.tapsSame += tapsSame ? 1 : 0;
    c.endSame += endSame ? 1 : 0;
    c.engineChords += engRest.size;
    c.caChords += caRest.size;
    for (const k of caRest) if (engRest.has(k)) c.bothChords++;
    const es = survivors(e.taps, engOk, engRest);
    const cs = survivors(e.taps, caOk, caRest);
    engOk.forEach((ok, k) => {
      if (!ok) return;
      c.lines++;
      c.survive += es[k] ? 1 : 0;
      c.surviveSame += es[k] === cs[k] ? 1 : 0;
    });
    if (!tapsSame || !endSame) c.differ.push(i);
  });
  return c;
}

export function loadTable(root: string): RuleTable {
  return new RuleTable(JSON.parse(readFileSync(resolve(root, 'web/strand-data.json'), 'utf8')) as StrandData);
}

export function replayAll(table: RuleTable, fx: Fixture, knobs: Partial<LineKnobs> = PARITY_KNOBS, maxTicks = 4000): CaEpisode[] {
  const boards = new Map<string, Board>();
  return fx.episodes.map((e, i) => {
    let b = boards.get(e.board);
    if (!b) {
      b = boardOf(e.board, fx.boards[e.board]);
      boards.set(e.board, b);
    }
    try {
      return replay(table, b, e, knobs, maxTicks);
    } catch (err) {
      throw new Error(`episode ${i} (${e.board}): ${(err as Error).message}`);
    }
  });
}

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) throw new Error(`unexpected argument ${a}`);
    if (i + 1 >= argv.length || argv[i + 1].startsWith('--')) out[a.slice(2)] = '1';
    else out[a.slice(2)] = argv[++i];
  }
  return out;
}

function main(): void {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const args = parseArgs(process.argv.slice(2));
  const inPath = resolve(root, args.in ?? 'data/strand-v2/collide.json');
  const outPath = resolve(root, args.out ?? resolve(dirname(inPath), 'ca-collide.json'));
  const maxTicks = Number(args['max-ticks'] ?? 4000);
  const t0 = performance.now();
  const table = loadTable(root);
  const fx = JSON.parse(readFileSync(inPath, 'utf8')) as Fixture;
  const ca = replayAll(table, fx, PARITY_KNOBS, maxTicks);
  const knobs = { ...DEFAULT_LINE_KNOBS, ...PARITY_KNOBS };
  writeFileSync(outPath, JSON.stringify({
    version: 1,
    generated: new Date().toISOString(),
    source: inPath.slice(root.length + 1),
    knobs,
    conventions: {
      time: 'CA steps; a tap at t is applied after the t-th step (generation t), then the chords are read',
      chords: '[player, row, col, a, b, on, off]: a < b; off -1 = there at rest',
      taps: '{ok, reason}: per input tap, in its order; reason = LineCA.tap\'s Refusal',
    },
    episodes: ca,
  }));
  const ms = performance.now() - t0;
  if (args.quiet) return;
  const steps = ca.reduce((a, e) => a + e.ticks, 0);
  console.log(`${ca.length} episodes, ${steps} steps, ${(ms / 1000).toFixed(1)} s -> ${outPath}`);
  const c = compare(fx, ca);
  if (c.episodes) {
    console.log(`LineCA = the engine at rest: taps ${c.tapsSame}/${c.episodes}, end state ${c.endSame}/${c.episodes}, ` +
      `survival agrees for ${c.surviveSame}/${c.lines} of the engine's lines (${c.survive} survive); chords at rest: ` +
      `engine ${c.engineChords}, LineCA ${c.caChords}, both ${c.bothChords}; ` +
      `differ: ${c.differ.slice(0, 20).join(' ')}${c.differ.length > 20 ? ' …' : ''}`);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
