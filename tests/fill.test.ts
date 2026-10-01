import { describe, expect, it } from 'vitest';
import { Automaton, Event, Orient, Terrain, drawLine } from '../src/ca.js';
import { makeBoard } from '../src/hex.js';
import { expectedOutcome, randomLine, randomLoop, randomScribble, rng } from '../src/lines.js';

function sameSet(a: Set<number>, b: Set<number>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

/** Draw `cells` on `ca`, check event, length and fill against the reference. */
function check(ca: Automaton, cells: number[], label: string) {
  const want = expectedOutcome(ca.board, cells);
  const got = drawLine(ca, cells);
  expect(got.event, `${label}: event`).toBe(want.event);
  expect(got.drawn, `${label}: cells drawn`).toBe(want.drawn);
  expect(sameSet(ca.flooded(), want.fill), `${label}: fill`).toBe(true);
  expect(ca.board.cells.filter((i) => ca.cells[i].pulse !== 0), `${label}: pulses left over`).toEqual([]);
  return want;
}

describe('bridges: flood the side of the shorter arc', () => {
  it('matches the reference on random edge-to-edge lines', () => {
    let ties = 0;
    let cases = 0;
    for (let seed = 1; seed <= 300; seed++) {
      const rand = rng(seed);
      const board = makeBoard(3 + Math.floor(rand() * 22));
      const line = randomLine(board, rand, { wiggle: rand() * 2 });
      if (!line) continue;
      cases++;
      const want = check(new Automaton(board), line, `seed ${seed}`);
      expect(want.event).toBe(Event.Bridge);
      if (want.arcs![0].length === want.arcs![1].length) ties++;
    }
    expect(cases).toBeGreaterThan(280);
    expect(ties).toBeGreaterThan(0); // the tie path is exercised too
  });

  it('decides 2·(shorter arc) + 2 steps after the last cell lands', () => {
    for (let seed = 1000; seed < 1060; seed++) {
      const rand = rng(seed);
      const board = makeBoard(5 + Math.floor(rand() * 15));
      const line = randomLine(board, rand)!;
      const want = expectedOutcome(board, line);
      const s = Math.min(want.arcs![0].length, want.arcs![1].length);
      const ca = new Automaton(board);
      for (const c of line) {
        ca.place(c);
        ca.step();
      }
      const h = line[line.length - 1];
      // Step 1 armed it (done above); then fire, echo out and back, hear.
      let steps = 1;
      while (ca.cells[h].foot < 3) {
        ca.step();
        steps++;
      }
      expect(steps).toBe(2 * s + 2);
    }
  });
});

describe('circuits: flood the inside of a loop', () => {
  it('matches the reference on random loops, both ways round', () => {
    const seen = new Set<Orient>();
    let cases = 0;
    for (let seed = 1; seed <= 300; seed++) {
      const rand = rng(seed * 31);
      const board = makeBoard(5 + Math.floor(rand() * 20));
      const loop = randomLoop(board, rand, { wiggle: rand() * 2 });
      if (!loop) continue;
      cases++;
      const ca = new Automaton(board);
      const want = check(ca, loop, `seed ${seed}`);
      expect(want.event).toBe(Event.Circuit);
      expect(want.fill.size).toBeGreaterThan(0);
      seen.add(ca.cells[loop[0]].decided);
    }
    expect(cases).toBeGreaterThan(250);
    expect([...seen].sort()).toEqual([Orient.Ccw, Orient.Cw]);
  });
});

describe('scribbles: any drawing', () => {
  it('matches the reference: circuits, bridges, sharp turns, border runs, open lines', () => {
    const events = new Map<Event, number>();
    for (let seed = 1; seed <= 600; seed++) {
      const rand = rng(seed * 7);
      const board = makeBoard(3 + Math.floor(rand() * 16));
      const walk = randomScribble(board, rand);
      const want = check(new Automaton(board), walk, `seed ${seed}`);
      events.set(want.event, (events.get(want.event) ?? 0) + 1);
    }
    expect(events.get(Event.Circuit)).toBeGreaterThan(50);
    expect(events.get(Event.Bridge)).toBeGreaterThan(50);
  });

  it('never decides an impossible orientation', () => {
    for (let seed = 1; seed <= 300; seed++) {
      const rand = rng(seed * 13);
      const board = makeBoard(4 + Math.floor(rand() * 12));
      const ca = new Automaton(board);
      drawLine(ca, randomScribble(board, rand));
      expect(board.cells.filter((i) => ca.cells[i].decided === Orient.Bad)).toEqual([]);
    }
  });
});

describe('rounds', () => {
  it('resolves one line after another over old lines (walls)', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const rand = rng(seed * 7919);
      const board = makeBoard(8 + Math.floor(rand() * 16));
      const ca = new Automaton(board);
      const walls = new Set<number>();
      for (let round = 0; round < 8; round++) {
        const kind = round % 3;
        const blocked = walls;
        const cells =
          kind === 0 ? randomLine(board, rand, { blocked })
          : kind === 1 ? randomLoop(board, rand, { blocked })
          : randomScribble(board, rand, { blocked });
        if (!cells) continue;
        check(ca, cells, `seed ${seed} round ${round}`);
        for (const c of ca.line) walls.add(c);
        ca.commit();
        expect([...walls].filter((c) => ca.cells[c].terrain !== Terrain.Wall)).toEqual([]);
      }
    }
  });

  it('refuses a cell that is not next to the head, and anything after the line closes', () => {
    const board = makeBoard(6);
    const rand = rng(3);
    const ca = new Automaton(board);
    const loop = randomLoop(board, rand)!;
    ca.place(loop[0]);
    ca.step();
    expect(ca.canPlace(loop[2])).toBe(false);
    drawLine(ca, loop.slice(1));
    expect(ca.event).toBe(Event.Circuit);
    expect(ca.canExtend).toBe(false);
  });
});
