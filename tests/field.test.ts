import { describe, expect, it } from 'vitest';
import { hexField, maskedField, presetWalls } from '../src/field.js';
import { coordsOf, hexDistance, makeBoard } from '../src/hex.js';

describe('hexField', () => {
  it('has N = 3R(R+1)+1 field cells and K = R', () => {
    for (const R of [1, 3, 5, 8]) {
      const field = hexField(R);
      expect(field.N).toBe(3 * R * (R + 1) + 1);
      expect(field.K).toBe(R);
      expect(field.CAP).toBe(field.N + 1);
      expect(field.topo.cells.length).toBe(field.N);
    }
  });

  it("every non-root field slot's gpar names a field neighbour one step closer", () => {
    const field = hexField(8);
    const { board, topo, root, ids, gpar } = field;

    // Independent BFS distance from the root, over field slots only.
    const dist = new Int32Array(board.size).fill(-1);
    const queue = [root];
    dist[root] = 0;
    for (let i = 0; i < queue.length; i++) {
      const slot = queue[i];
      for (let k = 0; k < 6; k++) {
        const j = board.neighbours[slot * 6 + k];
        if (j >= 0 && topo.isField[j] && dist[j] < 0) {
          dist[j] = dist[slot] + 1;
          queue.push(j);
        }
      }
    }

    for (const slot of topo.cells) {
      if (slot === root) {
        expect(gpar[slot]).toBe(0);
        continue;
      }
      expect(gpar[slot]).not.toBe(0);
      let found = false;
      for (let k = 0; k < 6; k++) {
        const j = board.neighbours[slot * 6 + k];
        if (j >= 0 && topo.isField[j] && dist[j] === dist[slot] - 1 && ids[j] === gpar[slot]) found = true;
      }
      expect(found, `slot ${slot}: gpar ${gpar[slot]} names a neighbour one step closer`).toBe(true);
    }
  });

  it('ids are a permutation of 1..N over field slots, 0 elsewhere', () => {
    const field = hexField(6);
    const seen = new Array(field.N + 1).fill(false);
    for (const slot of field.topo.cells) {
      const id = field.ids[slot];
      expect(id).toBeGreaterThanOrEqual(1);
      expect(id).toBeLessThanOrEqual(field.N);
      expect(seen[id]).toBe(false);
      seen[id] = true;
    }
    expect(seen.slice(1).every(Boolean)).toBe(true);
    for (let i = 0; i < field.board.size; i++) {
      if (!field.topo.isField[i]) expect(field.ids[i]).toBe(0);
    }
  });

  it('different idSeed values give different permutations', () => {
    const a = hexField(5, { idSeed: 1 });
    const b = hexField(5, { idSeed: 2 });
    let differs = false;
    for (const slot of a.topo.cells) if (a.ids[slot] !== b.ids[slot]) differs = true;
    expect(differs).toBe(true);
  });
});

describe('maskedField', () => {
  it('accepts a connected mask', () => {
    const board = makeBoard(8);
    const mask = new Uint8Array(board.size);
    for (const i of board.cells) {
      const [, r] = coordsOf(board, i);
      if (r >= 0) mask[i] = 1; // the lower half: still 6-connected
    }
    const field = maskedField(board, mask);
    expect(field.N).toBeGreaterThan(0);
    expect(field.N).toBe(mask.reduce((s, v) => s + v, 0));
  });

  it('throws on a disconnected mask', () => {
    const board = makeBoard(8);
    const mask = new Uint8Array(board.size);
    // Two islands far apart, nothing between them.
    for (const i of board.cells) {
      const [q, r] = coordsOf(board, i);
      if (hexDistance(q - 5, r) <= 1 || hexDistance(q + 5, r) <= 1) mask[i] = 1;
    }
    expect(() => maskedField(board, mask)).toThrow();
  });
});

describe('presetWalls', () => {
  it('produces only field slots, for every preset', () => {
    const field = hexField(10);
    for (const preset of ['hexagon', 'blob', 'lobes', 'ring'] as const) {
      const walls = presetWalls(field, preset, 3);
      for (const slot of walls) expect(field.topo.isField[slot]).toBe(1);
    }
    expect(presetWalls(field, 'hexagon')).toEqual([]);
  });

  it('also stays within field slots on a masked (ragged) field', () => {
    const board = makeBoard(10);
    const mask = new Uint8Array(board.size);
    for (const i of board.cells) {
      const [q, r] = coordsOf(board, i);
      if (hexDistance(q, r) <= 9) mask[i] = 1; // a slightly smaller, still-full hexagon
    }
    const field = maskedField(board, mask);
    for (const preset of ['blob', 'lobes', 'ring'] as const) {
      const walls = presetWalls(field, preset, 4);
      for (const slot of walls) expect(field.topo.isField[slot]).toBe(1);
    }
  });
});
