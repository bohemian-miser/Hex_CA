import { describe, expect, it } from 'vitest';
import { Automaton, NONE } from '../src/ca.js';
import { Board, borderRing, coordsOf, indexOf, makeBoard } from '../src/hex.js';
import { expectedFill, randomLine, randomLoop, randomWalls, rng } from '../src/lines.js';

function sorted(s: Set<number>): number[] {
  return [...s].sort((a, b) => a - b);
}

/** Settle, then compare with the reference. Returns the steps it took. */
function settleAndCheck(ca: Automaton, label: string): number {
  const steps = ca.run(ca.settleBound * 2);
  expect(ca.changed, `${label}: never settled`).toBe(0);
  expect(sorted(ca.filled()), label).toEqual(sorted(expectedFill(ca.board, ca.walls())));
  return steps;
}

function paint(ca: Automaton, walls: Iterable<number>, on = true): void {
  for (const c of walls) ca.setWall(c, on);
}

describe('the border ring', () => {
  it('numbers itself 0..P-1 and counts the open cells', () => {
    for (const R of [1, 2, 5, 13]) {
      const board = makeBoard(R);
      const ca = new Automaton(board);
      const ring = borderRing(board);
      paint(ca, ring.filter((_, k) => k % 3 === 0));
      ca.run();
      expect(ring.map((c) => ca.cells[c].idx).sort((a, b) => a - b)).toEqual(ring.map((_, k) => k));
      const T = ring.filter((_, k) => k % 3 !== 0).length;
      for (const c of ring) expect(ca.cells[c].tot).toBe(T);
    }
  });
});

describe('static: the fill depends only on the walls there are', () => {
  it('a circuit drawn in two halves, with time between, fills', () => {
    const board = makeBoard(10);
    const loop = randomLoop(board, rng(5))!;
    const ca = new Automaton(board);
    const half = Math.floor(loop.length / 2);
    paint(ca, loop.slice(0, half));
    settleAndCheck(ca, 'first half');
    expect(ca.filled().size).toBe(0);
    paint(ca, loop.slice(half));
    settleAndCheck(ca, 'second half');
    expect(ca.filled().size).toBeGreaterThan(0);
    // Knock one cell out: it leaks, and the fill goes.
    ca.setWall(loop[3], false);
    settleAndCheck(ca, 'reopened');
    expect(ca.filled().size).toBe(0);
  });

  it('a bridge fills its shorter side, whatever order its cells come in', () => {
    for (let seed = 1; seed <= 30; seed++) {
      const rand = rng(seed);
      const board = makeBoard(6 + Math.floor(rand() * 10));
      const line = randomLine(board, rand)!;
      const ca = new Automaton(board);
      const order = [...line].sort(() => rand() - 0.5);
      for (const c of order) {
        ca.setWall(c, true);
        ca.step();
      }
      settleAndCheck(ca, `seed ${seed}`);
    }
  });

  it('matches the reference on random wall pictures', () => {
    for (let seed = 1; seed <= 150; seed++) {
      const rand = rng(seed * 17);
      const board = makeBoard(2 + Math.floor(rand() * 12));
      const ca = new Automaton(board);
      paint(ca, randomWalls(board, rand));
      const steps = settleAndCheck(ca, `seed ${seed}`);
      expect(steps).toBeLessThanOrEqual(ca.settleBound);
    }
  });

  it('absorbs edits made at any moment, many per step, adding and erasing', () => {
    for (let seed = 1; seed <= 60; seed++) {
      const rand = rng(seed * 101);
      const board = makeBoard(3 + Math.floor(rand() * 10));
      const ca = new Automaton(board);
      const target = randomWalls(board, rand);
      const noise = randomWalls(board, rand);
      const edits: Array<[number, boolean]> = [
        ...[...noise].map((c): [number, boolean] => [c, true]),
        ...[...target].map((c): [number, boolean] => [c, true]),
        ...[...noise].filter((c) => !target.has(c)).map((c): [number, boolean] => [c, false]),
      ];
      while (edits.length) {
        for (let k = Math.floor(rand() * 5); k >= 0 && edits.length; k--) {
          const [c, on] = edits.shift()!;
          ca.setWall(c, on);
        }
        for (let k = Math.floor(rand() * 8); k > 0; k--) ca.step();
      }
      expect(sorted(ca.walls())).toEqual(sorted(target));
      settleAndCheck(ca, `seed ${seed}`);
    }
  });

  it('self-stabilises: from random garbage in every field, the same answer', () => {
    for (let seed = 1; seed <= 60; seed++) {
      const rand = rng(seed * 7);
      const board = makeBoard(2 + Math.floor(rand() * 10));
      const ca = new Automaton(board);
      paint(ca, randomWalls(board, rand));
      const int = (n: number) => Math.floor(rand() * n);
      const { N, P } = ca.limits;
      for (const i of board.cells) {
        ca.cells[i] = {
          ...ca.cells[i],
          idx: int(P + 1), cnt: int(P + 1), tot: int(P + 1),
          lab: rand() < 0.2 ? NONE : int(P), d: int(N + 1),
          par: int(7) - 1, s: int(P + 1), v: rand() < 0.5,
        };
      }
      const steps = settleAndCheck(ca, `seed ${seed}`);
      expect(steps).toBeLessThanOrEqual(ca.settleBound);
    }
  });

  it('gives a rotated picture the rotated fill', () => {
    const rot = (board: Board, i: number) => {
      const [q, r] = coordsOf(board, i);
      return indexOf(board, -r, q + r);
    };
    for (let seed = 1; seed <= 30; seed++) {
      const rand = rng(seed * 3);
      const board = makeBoard(3 + Math.floor(rand() * 8));
      const walls = randomWalls(board, rand);
      const a = new Automaton(board);
      const b = new Automaton(board);
      paint(a, walls);
      paint(b, [...walls].map((c) => rot(board, c)));
      a.run();
      b.run();
      expect(sorted(new Set([...a.filled()].map((c) => rot(board, c))))).toEqual(sorted(b.filled()));
    }
  });
});
