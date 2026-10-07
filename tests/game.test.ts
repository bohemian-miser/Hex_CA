// The game host (src/game/host.ts, docs/spectacle-ca-hybrid.md §5 "Game"): a scripted two-player game replays
// byte for byte; respawn blocks taps for respawnMs; score = line tiles + sole fill at every read; taps land on
// the nearest chord and are refused in Spectacle's order.

import { describe, expect, it } from 'vitest';
import dataJson from '../web/strand-data.json';
import weightsJson from '../web/nca-weights.json';
import { Board, RuleTable, walk, type Rule, type StrandData } from '../src/strand.js';
import { loadWeights } from '../src/nca.js';
import { edgeMid, unitCentre } from '../src/draw.js';
import { Game, type GameEvent } from '../src/game/host.js';
import { Bot } from '../src/game/bot.js';

const data = dataJson as unknown as StrandData;
const table = new RuleTable(data);
const weights = loadWeights(weightsJson);
const l2 = new Board(data.boards.l2);
const rule = (text: string) => table.parse(text) as Rule;
const LOOPS = rule('128·010000000');
const OTHER = rule('01346·000001000');

/** A two-player level-2 game: two seats, a 2-seat area (the flood is the slow part). */
function twoPlayers(knobs = {}): { g: Game; a: number; b: number } {
  const g = new Game(l2, table, weights, undefined, { stepMs: 50, ...knobs });
  const a = g.addPlayer('A', LOOPS);
  const b = g.addPlayer('B', OTHER);
  return { g, a, b };
}

/** Every chord of `owner`'s rule on the board, as (cell, d0, d1), in board order. */
function chords(g: Game, owner: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let c = 0; c < g.board.n; c++) for (const [a, b] of g.chordsOf(owner, c)) out.push([c, a, b]);
  return out;
}

/** The tiles a tap of (cell, d0, d1) under `r` would draw, if nothing stopped it. */
function strandTiles(r: Rule, cell: number, d0: number, d1: number): Set<number> {
  const p = l2.pos[cell];
  const st = walk(table.exits(r, l2), l2, Math.floor(p / l2.w), p % l2.w, d0, d1);
  return new Set(st.rows.map((row, k) => l2.cellOf[row * l2.w + st.cols[k]]));
}

function run(g: Game, from: number, to: number, every = 50): number {
  let t = from;
  while (t < to) g.tick((t += every));
  return t;
}

describe('the game host', () => {
  it('seats players with distinct rules and refuses a rule another player holds', () => {
    const { g, a, b } = twoPlayers();
    expect([a, b]).toEqual([1, 2]);
    expect(g.addPlayer('C', LOOPS)).toBe(-1);
    expect(g.setRule(b, LOOPS)).toBe(false);
    expect(g.setRule(a, LOOPS)).toBe(true);
    expect(g.holder(OTHER)).toBe(b);
  });

  it('taps the chord nearest the point, either end, and draws it', () => {
    const { g, a } = twoPlayers();
    const [cell, d0, d1] = chords(g, a).find(([c]) => g.chordsOf(a, c).length >= 2)!;
    const p = l2.pos[cell];
    const [cx, cy] = unitCentre(p % l2.w, Math.floor(p / l2.w));
    // a point a little way along the chord from its d1 end
    const [ax, ay] = edgeMid(cx, cy, d0, 1);
    const [bx, by] = edgeMid(cx, cy, d1, 1);
    expect(g.tap(a, bx + (ax - bx) * 0.2, by + (ay - by) * 0.2)).toBeNull();
    const D = g.lines.ch.D[cell];
    expect((D >> d0) & 1 && (D >> d1) & 1).toBeTruthy();
    expect(g.drain().map((e) => e.kind)).toEqual(['tap']);
  });

  it('refuses taps where Spectacle does: off the board, no chord, a rival tile, too many tips', () => {
    const { g, a, b } = twoPlayers({ fuel: 0 });
    expect(g.tap(a, -100, -100)).toBe('no chord');
    const noChord = Array.from({ length: l2.n }, (_, c) => c).find((c) => !g.chordsOf(a, c).length);
    if (noChord !== undefined) expect(g.tapChord(a, noChord, 0, 1)).toBe('no chord');
    const [c0, d0, d1] = chords(g, a)[0];
    expect(g.tapChord(a, c0, d0, d1)).toBeNull();
    const rival = g.chordsOf(b, c0)[0];
    if (rival) expect(g.tapChord(b, c0, rival[0], rival[1])).toBe('rival tile');
    // a two-way tap is two tips: with maxTips 2 a second line waits for the first to stop
    const far = chords(g, a).find(([c]) => !strandTiles(LOOPS, c0, d0, d1).has(c))!;
    if (g.lines.countTips(a) >= 2) expect(g.tapChord(a, far[0], far[1], far[2])).toBe('heads');
    const kinds = g.drain().map((e) => (e.kind === 'refused' ? e.why : e.kind));
    expect(kinds[0]).toBe('no chord');
    expect(kinds).toContain('tap');
  });

  it('a hit blocks both players\' taps for respawnMs', () => {
    const { g, a, b } = twoPlayers({ respawnMs: 400 });
    // A's line first, grown out; then a tap of B whose strand runs into A's tiles.
    const [c0, d0, d1] = chords(g, a).map((c) => [c, strandTiles(LOOPS, ...c).size] as const).sort((x, y) => y[1] - x[1])[0][0];
    expect(g.tapChord(a, c0, d0, d1)).toBeNull();
    let t = run(g, 0, 4000);
    const held = new Set<number>();
    for (let c = 0; c < l2.n; c++) if (g.lines.ch.owner[c] === a && g.lines.ch.D[c]) held.add(c);
    expect(held.size).toBeGreaterThan(3);
    const shot = chords(g, b).find(([c, e0, e1]) => !held.has(c) && [...strandTiles(OTHER, c, e0, e1)].some((x) => held.has(x)));
    expect(shot).toBeDefined();
    g.drain();
    expect(g.tapChord(b, ...shot!)).toBeNull();
    let hit: GameEvent | undefined;
    for (let k = 0; k < 400 && !hit; k++) {
      g.tick((t += 50));
      hit = g.drain().find((e) => e.kind === 'hit');
    }
    expect(hit).toBeDefined();
    const at = g.time;
    // both are waiting; a tap anywhere is refused for the delay, then taken again
    const free = chords(g, b).find(([c]) => !g.lines.ch.rule[c] && g.territory()[c] === 0)!;
    expect(g.tapChord(b, ...free)).toBe('respawn');
    expect(g.respawnLeft(a)).toBeGreaterThan(0);
    while (g.time < at + 400) g.tick((t += 50));
    expect(g.respawnLeft(b)).toBe(0);
    expect(g.tapChord(b, ...free)).not.toBe('respawn');
  });

  it('scores line tiles plus the tiles the flood fills for one player alone, at every read', () => {
    const { g, a, b } = twoPlayers();
    const taps = [...chords(g, a).filter((_, k) => k % 11 === 3).slice(0, 4).map((c) => [a, ...c]),
      ...chords(g, b).filter((_, k) => k % 13 === 5).slice(0, 4).map((c) => [b, ...c])];
    let t = 0;
    for (let k = 0; k < 300; k++) {
      if (k % 30 === 0 && taps.length) {
        const [o, c, d0, d1] = taps.shift()!;
        g.tapChord(o, c, d0, d1);
      }
      g.tick((t += 50));
      const fills = [a, b].map((o) => g.area.fill(o));
      const want = new Int32Array(3);
      const { owner, D } = g.lines.ch;
      for (let c = 0; c < l2.n; c++) {
        if (D[c] && owner[c]) want[owner[c]]++;
        else if (fills[0][c] && !fills[1][c]) want[a]++;
        else if (fills[1][c] && !fills[0][c]) want[b]++;
      }
      const sc = g.scores();
      expect([sc[a], sc[b]]).toEqual([want[a], want[b]]);
    }
  });

  it('replays a scripted two-player game byte for byte', () => {
    const script = (g: Game, a: number, b: number) => {
      const ca = chords(g, a);
      const cb = chords(g, b);
      let t = 0;
      const log: string[] = [];
      for (let k = 0; k < 500; k++) {
        if (k % 25 === 0) {
          const [c, d0, d1] = ca[(k * 7) % ca.length];
          log.push(`a${k}:${g.tapChord(a, c, d0, d1)}`);
        }
        if (k % 25 === 12) {
          const [c, d0, d1] = cb[(k * 5) % cb.length];
          log.push(`b${k}:${g.tapChord(b, c, d0, d1)}`);
        }
        g.tick((t += 50));
        for (const e of g.drain()) log.push(`${e.t}:${e.kind}:${e.owner}:${e.cell}:${e.why ?? ''}`);
      }
      return { log, state: snapshot(g) };
    };
    const first = (() => { const { g, a, b } = twoPlayers(); return script(g, a, b); })();
    const again = (() => { const { g, a, b } = twoPlayers(); return script(g, a, b); })();
    expect(again.log).toEqual(first.log);
    expect(again.state).toEqual(first.state);
    expect(first.log.some((l) => l.includes(':tap:'))).toBe(true);
  });

  it('bots tap through the host by themselves, and a seeded bot game replays exactly', () => {
    const play = () => {
      const { g, a, b } = twoPlayers();
      let seed = 11;
      const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const bots = [new Bot(g, a, rand), new Bot(g, b, rand)];
      const kinds: Record<string, number> = {};
      let t = 0;
      for (let k = 0; k < 400; k++) {
        g.tick((t += 50));
        for (const bot of bots) bot.act(t);
        for (const e of g.drain()) kinds[e.kind] = (kinds[e.kind] ?? 0) + 1;
      }
      return { kinds, state: snapshot(g), tiles: [g.lines.lineTiles(a), g.lines.lineTiles(b)] };
    };
    const first = play();
    expect(first.kinds.tap).toBeGreaterThan(4);
    expect(first.tiles[0] + first.tiles[1]).toBeGreaterThan(0);
    expect(play()).toEqual(first);
  });
});

/** Everything the game shows: the line channels, the fills, territory and scores. */
function snapshot(g: Game): string {
  const { rule, owner, D, T, W, H } = g.lines.ch;
  const parts = [rule, owner, D, T, W, H].map((a) => Array.from(a.subarray(0, g.board.n)).join(','));
  for (const p of g.seated()) parts.push(Array.from(g.area.fill(p.owner)).join(''));
  parts.push(Array.from(g.territory()).join(','), Array.from(g.scores()).join(','), `${g.steps}/${g.floodSteps}`);
  return parts.join('|');
}
