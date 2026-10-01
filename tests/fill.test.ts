import { describe, expect, it } from 'vitest';
import { Automaton, Foot, Terrain } from '../src/ca.js';
import { makeBoard } from '../src/hex.js';
import { expectedFill, randomLine, rng } from '../src/lines.js';

function sameSet(a: Set<number>, b: Set<number>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

describe('flood fill of the shorter side', () => {
  it('matches the reference on random single lines', () => {
    let ties = 0;
    let cases = 0;
    for (let seed = 1; seed <= 400; seed++) {
      const rand = rng(seed);
      const radius = 3 + Math.floor(rand() * 22);
      const board = makeBoard(radius);
      const line = randomLine(board, rand, { wiggle: rand() * 2 });
      if (!line) continue;
      cases++;
      const onLine = new Set(line);
      const ca = new Automaton(board, (i) => (onLine.has(i) ? Terrain.Line : Terrain.Empty));
      ca.run();
      const want = expectedFill(board, line);
      if (want.shorter < 0) ties++;
      expect(sameSet(ca.flooded(), want.fill), `seed ${seed} radius ${radius}`).toBe(true);
    }
    expect(cases).toBeGreaterThan(380);
    expect(ties).toBeGreaterThan(0); // the tie path is exercised too
  });

  it('decides at step 2 + (cells on the shorter arc), both feet together', () => {
    for (let seed = 1000; seed < 1100; seed++) {
      const rand = rng(seed);
      const board = makeBoard(5 + Math.floor(rand() * 15));
      const line = randomLine(board, rand)!;
      const onLine = new Set(line);
      const ca = new Automaton(board, (i) => (onLine.has(i) ? Terrain.Line : Terrain.Empty));
      const want = expectedFill(board, line);
      const s = Math.min(want.arcs[0].length, want.arcs[1].length);
      const feet = [line[0], line[line.length - 1]];
      for (let t = 0; t < 2 + s; t++) {
        for (const f of feet) expect(ca.cells[f].foot).toBeLessThan(Foot.Tie);
        ca.step();
      }
      for (const f of feet) expect(ca.cells[f].foot).toBeGreaterThanOrEqual(Foot.Tie);
    }
  });

  it('goes quiet: no pulse outlives the round', () => {
    const rand = rng(7);
    const board = makeBoard(20);
    const line = randomLine(board, rand)!;
    const onLine = new Set(line);
    const ca = new Automaton(board, (i) => (onLine.has(i) ? Terrain.Line : Terrain.Empty));
    ca.run();
    for (const i of board.cells) expect(ca.cells[i].pulse).toBe(0);
  });

  it('resolves lines one after another over old lines (walls)', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const rand = rng(seed * 7919);
      const board = makeBoard(8 + Math.floor(rand() * 16));
      const walls = new Set<number>();
      let line = randomLine(board, rand)!;
      const ca = new Automaton(board, () => Terrain.Empty);
      ca.commit(line);
      for (let round = 0; round < 6; round++) {
        ca.run();
        const want = expectedFill(board, line);
        expect(sameSet(ca.flooded(), want.fill), `seed ${seed} round ${round}`).toBe(true);
        for (const c of line) walls.add(c);
        const nextLine = randomLine(board, rand, { blocked: walls });
        if (!nextLine) break;
        line = nextLine;
        ca.commit(line);
      }
    }
  });
});
