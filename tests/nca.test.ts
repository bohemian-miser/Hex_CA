import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { HexNCA, NEIGHBOURS, boardMask, buildConst, cellCoords, cellIndex, hexDist, loadWeights, randomBridge, side, targets } from '../src/nca.js';
import { rng } from '../src/lines.js';

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

/**
 * A wall that hugs just inside the rim (hexDist = R − 1) for a long stretch, then cuts
 * across to the true rim at both ends — a single curved bridge, rim to rim, for v4 §2's
 * "measures disagree" case (spec: "a wall that hugs the rim for a long stretch then cuts
 * across, so the larger-area side has the shorter span"). Built by walking the
 * hexDist = R − 1 ring in angle order (`buildConst`'s theta1), leaving a `gapLen`-cell gap
 * starting at `gapStart` open, and capping each side of the gap with one outward step to
 * a true rim cell. The huge interior (reachable through the gap) ends up with by far the
 * most cells but the narrowest span (just that one short arc of true rim); the thin
 * rim-hugging band on the far side of the wall has far fewer cells but spans almost the
 * whole rim.
 */
function hugThenCutWalls(R: number, gapStart: number, gapLen: number): Uint8Array {
  const S = side(R);
  const theta1 = buildConst('theta1', R);
  const ring: Array<{ q: number; r: number; i: number }> = [];
  for (let r = -R; r <= R; r++) {
    for (let q = -R; q <= R; q++) if (hexDist(q, r) === R - 1) ring.push({ q, r, i: cellIndex(R, q, r) });
  }
  ring.sort((a, b) => theta1[a.i] - theta1[b.i]);
  const gap = new Set(Array.from({ length: gapLen }, (_, k) => (gapStart + k) % ring.length));
  const walls = new Uint8Array(S * S);
  for (let k = 0; k < ring.length; k++) if (!gap.has(k)) walls[ring[k].i] = 1;
  // Axial (dq, dr) neighbour offsets — NOT src/nca.ts's NEIGHBOURS, which are (drow, dcol).
  const AXIAL: ReadonlyArray<readonly [number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];
  const capOutward = (cell: { q: number; r: number }): void => {
    for (const [dq, dr] of AXIAL) {
      const q = cell.q + dq;
      const r = cell.r + dr;
      if (hexDist(q, r) === R) {
        walls[cellIndex(R, q, r)] = 1;
        return;
      }
    }
    throw new Error('hugThenCutWalls: no outward rim neighbour found');
  };
  capOutward(ring[(gapStart - 1 + ring.length) % ring.length]);
  capOutward(ring[(gapStart + gapLen) % ring.length]);
  return walls;
}

/** A model on the fixture's or another picture, from the fresh state. */
function modelOn(R: number, walls: ArrayLike<number>): HexNCA {
  const m = new HexNCA(weights, R);
  for (let i = 0; i < walls.length; i++) m.setWall(i, walls[i] ? 1 : 0);
  m.reset();
  return m;
}

// ── Synthetic weights, so version-specific tests don't depend on whichever version
// web/nca-weights.json happens to be right now (the Python side may have already
// replaced it with a version-2 file). ──

/** Deterministic small values in (-0.3, 0.3), corners (k = 0, 8) forced to 0. */
function fillW1(rand: () => number, H: number, CK: number): number[] {
  const w1 = new Array(H * CK * 9).fill(0);
  for (let h = 0; h < H; h++) {
    for (let c = 0; c < CK; c++) {
      for (let k = 1; k < 8; k++) w1[(h * CK + c) * 9 + k] = (rand() - 0.5) * 0.6;
    }
  }
  return w1;
}

/** Deterministic small values in (-0.3, 0.3) for w1pool (H × 2C: max then min per channel). */
function fillW1Pool(rand: () => number, H: number, C: number): number[] {
  return Array.from({ length: H * 2 * C }, () => (rand() - 0.5) * 0.6);
}

/** All seven v6 §1 consts, in the file order the spec fixes. */
const ALL_CONSTS = ['mask', 'theta1', 'theta2', 'src1', 'src1c', 'src2', 'src2c'];

/** A tiny, internally-consistent weights object of the given version. `constNames` overrides the default consts list (version >= 2 only). */
function syntheticWeights(version: 1 | 2 | 3, C: number, H: number, seed: number, constNames?: string[]): Record<string, unknown> {
  const consts = constNames ?? (version === 1 ? ['mask'] : ['mask', 'theta1', 'theta2']);
  const rand = rng(seed);
  const CK = C + consts.length;
  const w: Record<string, unknown> = {
    version, channels: C, hidden: H, clamp: null, fireRate: 1,
    w1: fillW1(rand, H, CK),
    b1: Array.from({ length: H }, () => (rand() - 0.5) * 0.2),
    w2: Array.from({ length: C * H }, () => (rand() - 0.5) * 0.6),
    b2: Array.from({ length: C }, () => (rand() - 0.5) * 0.2),
    meta: { note: 'synthetic, for tests' },
  };
  if (version >= 2) w.consts = consts;
  if (version === 3) {
    w.perception = 'taps+pool';
    w.w1pool = fillW1Pool(rand, H, C);
  }
  return w;
}

/**
 * The spec's forward pass at one cell, computed directly from the raw (un-rounded)
 * exported arrays and the spec's own tap/kernel-index rule — independent of HexNCA's
 * bias-folding optimisation. Used to check that optimisation is faithful, consts included.
 */
function referenceDelta(
  R: number, C: number, H: number, w1: number[], b1: number[], w2: number[], b2: number[],
  consts: Float32Array[], state: Float32Array, N: number, q: number, r: number,
): number[] {
  const S = side(R);
  const mask = boardMask(R);
  const CK = C + consts.length;
  const taps: ReadonlyArray<readonly [number, number]> = [[0, 0], ...NEIGHBOURS];
  const row = r + R;
  const col = q + R;
  const d = b2.slice();
  for (let h = 0; h < H; h++) {
    let z = b1[h];
    for (const [dr, dc] of taps) {
      const r2 = row + dr;
      const c2 = col + dc;
      if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S || !mask[r2 * S + c2]) continue;
      const j = r2 * S + c2;
      const k = (dr + 1) * 3 + (dc + 1);
      for (let c = 0; c < C; c++) z += w1[(h * CK + c) * 9 + k] * state[c * N + j];
      for (let c = 0; c < consts.length; c++) z += w1[(h * CK + (C + c)) * 9 + k] * consts[c][j];
    }
    const a = Math.max(0, z);
    for (let o = 0; o < C; o++) d[o] += w2[o * H + h] * a;
  }
  return d;
}

/**
 * `referenceDelta`, plus the version-3 pool features (spec v4 §1), computed independently:
 * per state channel, the max and min over the on-board taps among self and the six
 * `NEIGHBOURS` — never a corner (NEIGHBOURS has none) and never an off-board tap (skipped
 * outright, not filled with 0).
 */
function referenceDeltaPooled(
  R: number, C: number, H: number, w1: number[], w1pool: number[], b1: number[], w2: number[], b2: number[],
  consts: Float32Array[], state: Float32Array, N: number, q: number, r: number,
): number[] {
  const S = side(R);
  const mask = boardMask(R);
  const CK = C + consts.length;
  const taps: ReadonlyArray<readonly [number, number]> = [[0, 0], ...NEIGHBOURS];
  const row = r + R;
  const col = q + R;
  const pmax = new Array(C).fill(-Infinity);
  const pmin = new Array(C).fill(Infinity);
  for (const [dr, dc] of taps) {
    const r2 = row + dr;
    const c2 = col + dc;
    if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S || !mask[r2 * S + c2]) continue;
    const j = r2 * S + c2;
    for (let c = 0; c < C; c++) {
      const v = state[c * N + j];
      if (v > pmax[c]) pmax[c] = v;
      if (v < pmin[c]) pmin[c] = v;
    }
  }
  const d = b2.slice();
  for (let h = 0; h < H; h++) {
    let z = b1[h];
    for (const [dr, dc] of taps) {
      const r2 = row + dr;
      const c2 = col + dc;
      if (r2 < 0 || r2 >= S || c2 < 0 || c2 >= S || !mask[r2 * S + c2]) continue;
      const j = r2 * S + c2;
      const k = (dr + 1) * 3 + (dc + 1);
      for (let c = 0; c < C; c++) z += w1[(h * CK + c) * 9 + k] * state[c * N + j];
      for (let c = 0; c < consts.length; c++) z += w1[(h * CK + (C + c)) * 9 + k] * consts[c][j];
    }
    for (let c = 0; c < C; c++) z += w1pool[h * 2 * C + c] * pmax[c];
    for (let c = 0; c < C; c++) z += w1pool[h * 2 * C + C + c] * pmin[c];
    const a = Math.max(0, z);
    for (let o = 0; o < C; o++) d[o] += w2[o * H + h] * a;
  }
  return d;
}

describe('loadWeights: version 1 and version 2', () => {
  it('loads a synthetic version-1 file (consts = ["mask"]) and runs', () => {
    const R = 5;
    const w = loadWeights(syntheticWeights(1, 3, 6, 1));
    expect(w.version).toBe(1);
    expect(w.consts).toEqual(['mask']);
    const m = new HexNCA(w, R);
    for (const i of m.cells) if (hexDist(...cellCoords(R, i)) === 2) m.setWall(i, 1);
    m.reset();
    for (let s = 0; s < 10; s++) {
      m.step();
      for (const i of m.cells) expect(m.state[i]).toBe(m.walls[i]); // channel 0 is always the walls
      for (const v of m.state) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('loads a synthetic version-2 file (consts = mask, theta1, theta2), exposes them, and matches a manual forward pass', () => {
    const R = 3;
    const C = 3;
    const H = 5;
    const json = syntheticWeights(2, C, H, 2);
    const w = loadWeights(json);
    expect(w.version).toBe(2);
    expect(w.consts).toEqual(['mask', 'theta1', 'theta2']);

    const m = new HexNCA(w, R);
    expect(m.consts).toHaveLength(3);
    for (const a of m.consts) expect(a).toHaveLength(m.N);
    expect(Array.from(m.consts[0])).toEqual(Array.from(boardMask(R)));
    expect(Array.from(m.consts[1])).toEqual(Array.from(buildConst('theta1', R)));
    expect(Array.from(m.consts[2])).toEqual(Array.from(buildConst('theta2', R)));

    for (const i of m.cells) if (hexDist(...cellCoords(R, i)) === 1) m.setWall(i, 1);
    m.reset(); // the centre (0, 0) is not a wall, so its state is all zeros here

    const before = m.state.slice();
    m.step();
    const expected = referenceDelta(
      R, C, H, json.w1 as number[], json.b1 as number[], json.w2 as number[], json.b2 as number[],
      m.consts, before, m.N, 0, 0,
    );
    const centre = cellIndex(R, 0, 0);
    for (let o = 1; o < C; o++) expect(m.state[o * m.N + centre]).toBeCloseTo(expected[o], 4);
  });

  it('rejects an unsupported version, missing consts, or an unknown const', () => {
    expect(() => loadWeights({ ...syntheticWeights(1, 2, 2, 3), version: 4 })).toThrow(/version/);
    const v2 = syntheticWeights(2, 2, 2, 4);
    expect(() => loadWeights({ ...v2, consts: undefined })).toThrow(/consts/);
    expect(() => loadWeights({ ...v2, consts: [] })).toThrow(/consts/);
    expect(() => loadWeights({ ...v2, consts: ['mask', 'theta3'] })).toThrow(/theta3/);
  });
});

describe('loadWeights: version 3 (pooled perception)', () => {
  it('loads a synthetic version-3 file ("taps+pool"), exposes w1pool, and matches an independent forward pass with pooling — excluding an off-board neighbour and an on-board corner', () => {
    const R = 3;
    const C = 3;
    const H = 4;
    const json = syntheticWeights(3, C, H, 7);
    const w = loadWeights(json);
    expect(w.version).toBe(3);
    expect(w.perception).toBe('taps+pool');
    expect(w.consts).toEqual(['mask', 'theta1', 'theta2']);
    expect(w.w1pool).toBeInstanceOf(Float32Array);
    expect(w.w1pool).toHaveLength(H * 2 * C);

    const m = new HexNCA(w, R);
    const idx = (q: number, r: number) => cellIndex(R, q, r);
    const before = new Float32Array(C * m.N);
    const setCell = (q: number, r: number, values: number[]) => {
      const i = idx(q, r);
      for (let c = 0; c < C; c++) before[c * m.N + i] = values[c];
    };

    // Target cell (R, 0), an east rim cell. Of its 7 taps: self and 3 neighbours are
    // on-board (kept away from 0, so an erroneous phantom-zero pool entry would show
    // up as a wrong min); [0, 1] and [-1, 1] are off the S×S array entirely; [1, 0]
    // lands in-array but off the hex mask (hexDist R+1) — poisoned, to catch code that
    // reads it directly instead of checking validity; and (-1, -1), an on-board cell
    // that is never a tap at all (a corner, k = 0), is poisoned even more extremely, to
    // catch code that pools over the full 3×3 neighbourhood instead of the 7 taps.
    setCell(R, 0, [10, 15, 20]); // self
    setCell(R, -1, [12, 18, 24]); // tap (drow, dcol) = (-1, 0)
    setCell(R - 1, 0, [8, 13, 19]); // tap (0, -1)
    setCell(R - 1, 1, [11, 16, 22]); // tap (1, -1)
    setCell(R, 1, [-777, -777, -777]); // off-mask, in-array: tap (1, 0) — must be excluded
    setCell(R - 1, -1, [999, 999, 999]); // on-board corner, never a tap — must be excluded

    m.state.set(before);
    m.step();
    const expected = referenceDeltaPooled(
      R, C, H, json.w1 as number[], json.w1pool as number[], json.b1 as number[], json.w2 as number[], json.b2 as number[],
      m.consts, before, m.N, R, 0,
    );
    const target = idx(R, 0);
    for (let o = 1; o < C; o++) {
      expect(m.state[o * m.N + target]).toBeCloseTo(before[o * m.N + target] + expected[o], 4);
    }
  });

  it('loads a synthetic version-3 file with the full 7-name consts list (v6 §1) and matches an independent forward pass', () => {
    const R = 3;
    const C = 3;
    const H = 4;
    const json = syntheticWeights(3, C, H, 13, ALL_CONSTS);
    const w = loadWeights(json);
    expect(w.version).toBe(3);
    expect(w.consts).toEqual(ALL_CONSTS); // file order preserved, not reordered to some registry order
    expect(w.w1.length).toBe(H * (C + ALL_CONSTS.length) * 9); // right perception width: 9 taps x (C state + 7 consts) per hidden unit

    const m = new HexNCA(w, R);
    expect(m.consts).toHaveLength(ALL_CONSTS.length);
    for (let k = 0; k < ALL_CONSTS.length; k++) {
      expect(Array.from(m.consts[k])).toEqual(Array.from(buildConst(ALL_CONSTS[k], R)));
    }

    const rand = rng(21);
    const before = new Float32Array(C * m.N);
    for (const i of m.cells) for (let c = 0; c < C; c++) before[c * m.N + i] = (rand() - 0.5) * 2;
    m.state.set(before);
    m.step();
    const expected = referenceDeltaPooled(
      R, C, H, json.w1 as number[], json.w1pool as number[], json.b1 as number[], json.w2 as number[], json.b2 as number[],
      m.consts, before, m.N, 0, 0,
    );
    const centre = cellIndex(R, 0, 0);
    for (let o = 1; o < C; o++) { // o = 0 (wall) is forced to the wall picture, not the forward pass
      expect(m.state[o * m.N + centre]).toBeCloseTo(before[o * m.N + centre] + expected[o], 4);
    }
  });

  it('rejects version 3 without "taps+pool" perception, or a wrong-length w1pool', () => {
    const v3 = syntheticWeights(3, 2, 2, 8);
    expect(() => loadWeights({ ...v3, perception: undefined })).toThrow(/perception/);
    expect(() => loadWeights({ ...v3, perception: 'taps' })).toThrow(/perception/);
    expect(() => loadWeights({ ...v3, w1pool: (v3.w1pool as number[]).slice(1) })).toThrow(/w1pool/);
  });
});

describe('buildConst (the per-board constant inputs)', () => {
  const R = 6;
  const mask = boardMask(R);
  const theta1 = buildConst('theta1', R);
  const theta2 = buildConst('theta2', R);

  it('matches boardMask for "mask"', () => {
    expect(Array.from(buildConst('mask', R))).toEqual(Array.from(mask));
  });

  it('is 0 off the board and in [0, 1) on it, for both angles', () => {
    for (let i = 0; i < theta1.length; i++) {
      if (mask[i]) {
        expect(theta1[i]).toBeGreaterThanOrEqual(0);
        expect(theta1[i]).toBeLessThan(1);
        expect(theta2[i]).toBeGreaterThanOrEqual(0);
        expect(theta2[i]).toBeLessThan(1);
      } else {
        expect(theta1[i]).toBe(0);
        expect(theta2[i]).toBe(0);
      }
    }
  });

  it('gives the centre cell (0, 0) theta1 = 0', () => {
    expect(theta1[cellIndex(R, 0, 0)]).toBe(0);
  });

  it('has theta2 = theta1 + 0.5 mod 1 everywhere on the board', () => {
    for (const i of [...Array(theta1.length).keys()].filter((i) => mask[i])) {
      expect(theta2[i]).toBeCloseTo((theta1[i] + 0.5) % 1, 5);
    }
  });

  it('differs by about 0.5 in theta1 between two opposite rim cells', () => {
    const a = theta1[cellIndex(R, R, 0)];
    const b = theta1[cellIndex(R, -R, 0)];
    const diff = Math.abs(a - b);
    expect(Math.min(diff, 1 - diff)).toBeCloseTo(0.5, 4);
  });
});

describe('buildConst: src1/src1c/src2/src2c (v6 §1, theta gated by the rim)', () => {
  const R = 6;
  const mask = boardMask(R);
  const theta1 = buildConst('theta1', R);
  const theta2 = buildConst('theta2', R);
  const src1 = buildConst('src1', R);
  const src1c = buildConst('src1c', R);
  const src2 = buildConst('src2', R);
  const src2c = buildConst('src2c', R);
  const isRim = (i: number) => {
    const [q, r] = cellCoords(R, i);
    return mask[i] === 1 && hexDist(q, r) === R;
  };

  it('is 0 on every non-rim cell (on-board interior and off-board alike)', () => {
    for (let i = 0; i < src1.length; i++) {
      if (isRim(i)) continue;
      expect(src1[i]).toBe(0);
      expect(src1c[i]).toBe(0);
      expect(src2[i]).toBe(0);
      expect(src2c[i]).toBe(0);
    }
  });

  it('has src1 + src1c = 1 and src2 + src2c = 1 on every rim cell', () => {
    let rimCells = 0;
    for (let i = 0; i < src1.length; i++) {
      if (!isRim(i)) continue;
      rimCells++;
      expect(src1[i] + src1c[i]).toBeCloseTo(1, 6);
      expect(src2[i] + src2c[i]).toBeCloseTo(1, 6);
    }
    expect(rimCells).toBe(6 * R); // sanity: the test actually covered rim cells
  });

  it('matches theta1/theta2 (and 1 minus each) at a couple of rim cells', () => {
    for (const [q, r] of [[R, 0], [0, R], [-R, 0], [0, -R], [R, -R]] as const) {
      const i = cellIndex(R, q, r);
      expect(isRim(i)).toBe(true);
      expect(src1[i]).toBeCloseTo(theta1[i], 6);
      expect(src1c[i]).toBeCloseTo(1 - theta1[i], 6);
      expect(src2[i]).toBeCloseTo(theta2[i], 6);
      expect(src2c[i]).toBeCloseTo(1 - theta2[i], 6);
    }
  });
});

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
    expect(weights.w1.length).toBe(weights.hidden * (weights.channels + weights.consts.length) * 9);
    expect(typeof weights.meta).toBe('object');
  });

  it('rejects files that are not the spec’s format', () => {
    const bad = (patch: Record<string, unknown>) => () => loadWeights({ ...weightsJson, ...patch });
    expect(bad({ version: 4 })).toThrow(/version/); // not 3: the shipped file may already be version 3
    expect(bad({ b1: (weightsJson.b1 as number[]).slice(1) })).toThrow(/b1/);
    expect(bad({ hidden: 63 })).toThrow(/w1/);
    expect(bad({ clamp: [1, -1] })).toThrow(/clamp/);
    const w1 = (weightsJson.w1 as number[]).slice();
    w1[9] = 0.5; // k = 0 of the second (h, c) kernel
    expect(bad({ w1 })).toThrow(/corner/);
    expect(() => loadWeights(null)).toThrow();
  });
});

describe('targets (the oracle)', () => {
  const R = 6;
  /** Each target's filled cells, as sorted "q,r" lists. */
  const fills = (walls: Uint8Array) => targets(walls, R).map((t) => onCells(R, t).sort());
  /** The on-board non-wall cells a predicate picks, as a sorted "q,r" list. */
  const cellsWhere = (walls: Uint8Array, pick: (q: number, r: number) => boolean) =>
    onCells(R, wallsWhere(R, (q, r) => pick(q, r) && !walls[cellIndex(R, q, r)])).sort();

  it('fills nothing on an empty board', () => {
    expect(fills(new Uint8Array(side(R) ** 2))).toEqual([[]]);
  });

  it('fills the inside of a closed ring, and nothing else', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) === 2);
    expect(fills(walls)).toEqual([cellsWhere(walls, (q, r) => hexDist(q, r) < 2)]);
    expect(fills(walls)[0]).toHaveLength(7);
  });

  it('fills nothing when the ring has a gap', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) === 2 && !(q === 2 && r === 0));
    expect(fills(walls)).toEqual([[]]);
  });

  it('fills a ring that runs along the rim, and the whole board inside a walled rim', () => {
    // A radius-2 ring round (q, r) = (4, 0): five of its cells are rim cells, and the rest of the rim is one region.
    const walls = wallsWhere(R, (q, r) => hexDist(q - 4, r) === 2);
    expect(fills(walls)).toEqual([cellsWhere(walls, (q, r) => hexDist(q - 4, r) < 2)]);
    // The whole rim walled: no rim region, everything inside it fills.
    const rim = wallsWhere(R, (q, r) => hexDist(q, r) === R);
    expect(fills(rim).map((f) => f.length)).toEqual([3 * (R - 1) * R + 1]);
    // ...and with one rim cell open, that cell is a rim region and so is all the rest.
    rim[cellIndex(R, R, 0)] = 0;
    expect(fills(rim)).toEqual([[]]);
  });

  it('fills the smaller side of a straight wall across the board, off centre', () => {
    const walls = wallsWhere(R, (_q, r) => r === 2);
    expect(fills(walls)).toEqual([cellsWhere(walls, (_q, r) => r > 2)]);
  });

  it('gives two targets for a wall through the centre, one side filled in each', () => {
    const walls = wallsWhere(R, (_q, r) => r === 0);
    const top = cellsWhere(walls, (_q, r) => r < 0);
    const bottom = cellsWhere(walls, (_q, r) => r > 0);
    // The primary leaves the side with the lowest cell index (the top rows) unfilled.
    expect(fills(walls)).toEqual([bottom, top]);
  });

  it('gives a target per measure when area and span disagree: a wall that hugs the rim then cuts across', () => {
    const walls = hugThenCutWalls(R, 13, 5);
    const t = targets(walls, R);
    const filledCount = (f: Uint8Array) => f.reduce((n, v) => n + v, 0);
    expect(t).toHaveLength(2);
    expect(filledCount(t[0])).toBe(29); // primary: leaves out the big interior (max area, narrow span)
    expect(filledCount(t[1])).toBe(71); // secondary: leaves out the thin band (max span, small area)
    // A true-rim cell reached only through the gap (part of the big, narrow-span interior)...
    const gapRim = cellIndex(R, -6, 0);
    // ...versus one reached only along the hugged band (part of the thin, wide-span band):
    const bandRim = cellIndex(R, 6, 0);
    expect([walls[gapRim], walls[bandRim]]).toEqual([0, 0]);
    expect([t[0][gapRim], t[1][gapRim]]).toEqual([0, 1]);
    expect([t[0][bandRim], t[1][bandRim]]).toEqual([1, 0]);
  });

  it('fills a corner cut off by a wall', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q - R, r) === 2);
    expect(fills(walls)).toEqual([cellsWhere(walls, (q, r) => hexDist(q - R, r) < 2)]);
    expect(fills(walls)[0]).toHaveLength(4);
  });

  it('fills a loop on the larger side as well as the smaller side', () => {
    const walls = wallsWhere(R, (q, r) => r === 2 || hexDist(q + 1, r + 2) === 2);
    expect(fills(walls)).toEqual([cellsWhere(walls, (q, r) => r > 2 || hexDist(q + 1, r + 2) < 2)]);
  });

  it('fills all but the largest of three rim regions', () => {
    const walls = wallsWhere(R, (_q, r) => Math.abs(r) === 4);
    expect(fills(walls)).toEqual([cellsWhere(walls, (_q, r) => Math.abs(r) > 4)]);
  });

  it('never fills a wall or an off-board slot', () => {
    const walls = wallsWhere(R, (q, r) => hexDist(q, r) <= 3 && (q + r) % 2 === 0);
    const mask = boardMask(R);
    for (const fill of targets(walls, R)) {
      for (let i = 0; i < fill.length; i++) if (walls[i] || !mask[i]) expect(fill[i]).toBe(0);
    }
  });
});

describe('randomBridge', () => {
  it('runs rim to rim through the interior and splits the board', () => {
    for (const R of [4, 8, 16]) {
      for (let seed = 1; seed <= 10; seed++) {
        const walls = new Uint8Array(side(R) ** 2);
        const bridge = randomBridge(walls, R, rng(seed));
        expect(bridge).not.toBeNull();
        const onRim = bridge!.map((i) => hexDist(...cellCoords(R, i)) === R);
        expect(onRim[0] && onRim[onRim.length - 1] && !onRim.slice(1, -1).some(Boolean)).toBe(true);
        for (const i of bridge!) walls[i] = 1;
        expect(onCells(R, targets(walls, R)[0]).length).toBeGreaterThan(0);
      }
    }
  });
});
