import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { HexNCA, boardMask, cellIndex, enclosed, hexDist, loadWeights, side } from '../src/nca.js';

const readJson = (path: string): unknown => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const weightsJson = readJson('../web/nca-weights.json') as Record<string, unknown>;
const weights = loadWeights(weightsJson);
const fixture = readJson('./fixtures/nca-parity.json') as { R: number; walls: number[]; steps: number; state: number[] };

/** Walls at every on-board (q, r) a predicate picks. */
function wallsWhere(R: number, pick: (q: number, r: number) => boolean): Uint8Array {
  const S = side(R);
  const w = new Uint8Array(S * S);
  for (let r = -R; r <= R; r++) {
    for (let q = -R; q <= R; q++) if (hexDist(q, r) <= R && pick(q, r)) w[cellIndex(R, q, r)] = 1;
  }
  return w;
}

/** The on-board (q, r) where `grid` is 1, as "q,r" strings. */
function onCells(R: number, grid: Uint8Array): string[] {
  const out: string[] = [];
  for (let r = -R; r <= R; r++) {
    for (let q = -R; q <= R; q++) if (hexDist(q, r) <= R && grid[cellIndex(R, q, r)]) out.push(`${q},${r}`);
  }
  return out;
}

/** A model on the fixture's or another picture, from the fresh state. */
function modelOn(R: number, walls: ArrayLike<number>): HexNCA {
  const m = new HexNCA(weights, R);
  for (let i = 0; i < walls.length; i++) m.setWall(i, walls[i] ? 1 : 0);
  m.reset();
  return m;
}

describe('the trained NCA in TypeScript', () => {
  it('reproduces the parity fixture to 1e-3 on every value', () => {
    const S = side(fixture.R);
    expect(fixture.walls.length).toBe(S * S);
    expect(fixture.state.length).toBe(weights.channels * S * S);
    const m = modelOn(fixture.R, fixture.walls);
    m.step(fixture.steps);
    let worst = 0;
    let at = -1;
    for (let i = 0; i < fixture.state.length; i++) {
      const e = Math.abs(m.state[i] - fixture.state[i]);
      if (e > worst) [worst, at] = [e, i];
    }
    expect(worst, `largest difference at channel ${Math.floor(at / (S * S))}, slot ${at % (S * S)}`).toBeLessThan(1e-3);
  });

  it('starts from the fresh state: zeros except channel 0 = the walls', () => {
    const R = 5;
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) === 2);
    const m = modelOn(R, walls);
    const N = m.N;
    expect(Array.from(m.state.subarray(0, N))).toEqual(Array.from(walls));
    expect(m.state.subarray(N).every((v) => v === 0)).toBe(true);
  });

  it('keeps every dead (off-board) slot exactly 0 in every channel', () => {
    const R = 6;
    const walls = wallsWhere(R, (q, r) => hexDist(q - 1, r) === 2 || (q === -3 && r <= 2));
    const m = modelOn(R, walls);
    const mask = boardMask(R);
    for (let s = 0; s < 40; s++) {
      m.step();
      for (let c = 0; c < m.channels; c++) {
        for (let i = 0; i < m.N; i++) if (!mask[i]) expect(m.state[c * m.N + i]).toBe(0);
      }
    }
  });

  it('has channel 0 equal to the walls after every step, edits included', () => {
    const R = 6;
    const m = modelOn(R, wallsWhere(R, (q, r) => hexDist(q, r) === 3));
    const ch0 = () => Array.from(m.state.subarray(0, m.N));
    m.step();
    expect(ch0()).toEqual(Array.from(m.walls));
    // Open the ring, and wall a cell inside it, mid-run.
    expect(m.setWall(cellIndex(R, 3, 0), 0)).toBe(true);
    expect(m.setWall(cellIndex(R, 0, 0), 1)).toBe(true);
    expect(m.setWall(cellIndex(R, 0, 0), 1)).toBe(false);
    expect(m.setWall(0, 1)).toBe(false); // slot 0 is off the board
    for (let s = 0; s < 5; s++) {
      m.step();
      expect(ch0()).toEqual(Array.from(m.walls));
    }
    expect(m.steps).toBe(6);
  });

  it('clamps every value to the export’s range', () => {
    const clamp = weights.clamp;
    if (!clamp) return;
    const m = modelOn(fixture.R, fixture.walls);
    m.step(30);
    for (const v of m.state) {
      expect(v).toBeGreaterThanOrEqual(clamp[0]);
      expect(v).toBeLessThanOrEqual(clamp[1]);
    }
  });
});

describe('loadWeights', () => {
  it('reads the shipped file and its meta', () => {
    expect(weights.w1.length).toBe(weights.hidden * (weights.channels + 1) * 9);
    expect(typeof weights.meta).toBe('object');
  });

  it('rejects files that are not the spec’s format', () => {
    const bad = (patch: Record<string, unknown>) => () => loadWeights({ ...weightsJson, ...patch });
    expect(bad({ version: 2 })).toThrow(/version/);
    expect(bad({ b1: (weightsJson.b1 as number[]).slice(1) })).toThrow(/b1/);
    expect(bad({ hidden: 63 })).toThrow(/w1/);
    expect(bad({ clamp: [1, -1] })).toThrow(/clamp/);
    const w1 = (weightsJson.w1 as number[]).slice();
    w1[9] = 0.5; // k = 0 of the second (h, c) kernel
    expect(bad({ w1 })).toThrow(/corner/);
    expect(() => loadWeights(null)).toThrow();
  });
});

describe('enclosed (the oracle)', () => {
  const R = 6;

  it('fills nothing on an empty board', () => {
    expect(onCells(R, enclosed(new Uint8Array(side(R) ** 2), R))).toEqual([]);
  });

  it('fills the inside of a closed ring, and nothing else', () => {
    const fill = enclosed(wallsWhere(R, (q, r) => hexDist(q, r) === 2), R);
    expect(onCells(R, fill).sort()).toEqual(onCells(R, wallsWhere(R, (q, r) => hexDist(q, r) < 2)).sort());
    expect(onCells(R, fill)).toHaveLength(7);
  });

  it('fills nothing when the ring has a gap', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) === 2 && !(q === 2 && r === 0));
    expect(onCells(R, enclosed(walls, R))).toEqual([]);
  });

  it('fills a ring that runs along the rim', () => {
    // A radius-2 ring round (q, r) = (4, 0): five of its cells are rim cells.
    const walls = wallsWhere(R, (q, r) => hexDist(q - 4, r) === 2);
    expect(onCells(R, enclosed(walls, R)).sort()).toEqual(
      onCells(R, wallsWhere(R, (q, r) => hexDist(q - 4, r) < 2)).sort(),
    );
    // The whole rim walled: everything inside it fills.
    const rim = wallsWhere(R, (q, r) => hexDist(q, r) === R);
    expect(onCells(R, enclosed(rim, R))).toHaveLength(3 * (R - 1) * R + 1);
    // ...and with one rim cell open, that cell is outside and so is all the rest.
    rim[cellIndex(R, R, 0)] = 0;
    expect(onCells(R, enclosed(rim, R))).toEqual([]);
  });

  it('never fills a wall or an off-board slot', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) <= 3 && (q + r) % 2 === 0);
    const fill = enclosed(walls, R);
    const mask = boardMask(R);
    for (let i = 0; i < fill.length; i++) if (walls[i] || !mask[i]) expect(fill[i]).toBe(0);
  });
});
