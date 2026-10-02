import { describe, expect, it } from 'vitest';
import { hexField, Field } from '../src/field.js';
import { FILLED, OFF, ON, WALL } from '../src/fill.js';
import { coordsOf, hexDistance } from '../src/hex.js';
import { oracle, settleBound } from '../src/oracle.js';

/** A picture as a lookup: everything not named is OFF. */
function makePaint(field: Field, marks: { on?: Iterable<number>; wall?: Iterable<number> }): (slot: number) => number {
  const p = new Uint8Array(field.board.size).fill(OFF);
  for (const s of marks.on ?? []) p[s] = ON;
  for (const s of marks.wall ?? []) p[s] = WALL;
  return (slot) => p[slot];
}

/** Every field slot satisfying a coordinate predicate. */
function slotsWhere(field: Field, pred: (q: number, r: number, d: number) => boolean): number[] {
  const out: number[] = [];
  for (const slot of field.topo.cells) {
    const [q, r] = coordsOf(field.board, slot);
    if (pred(q, r, hexDistance(q, r))) out.push(slot);
  }
  return out;
}

/** Assert every slot matching a predicate has the given oracle state. */
function expectRegion(
  field: Field,
  state: Uint8Array,
  pred: (q: number, r: number, d: number) => boolean,
  expected: number,
  label: string,
): void {
  const slots = slotsWhere(field, pred);
  expect(slots.length, `${label}: no slots matched`).toBeGreaterThan(0);
  for (const slot of slots) {
    expect(state[slot], `${label}: slot ${slot} (${coordsOf(field.board, slot)})`).toBe(expected);
  }
}

describe('oracle: hand-counted pictures (DESIGN.md §4)', () => {
  it('a tie: a full row through the centre fills neither half', () => {
    // R = 8: row r=0 has 2R+1 = 17 cells; the two halves (r<0, r>0) are
    // 100 cells each by the hexagon's reflection symmetry — an exact tie,
    // so DESIGN.md's tie rule keeps both OFF.
    const R = 8;
    const field = hexField(R);
    const bridge = slotsWhere(field, (_q, r) => r === 0);
    expect(bridge.length).toBe(2 * R + 1);
    const upper = slotsWhere(field, (_q, r) => r < 0);
    const lower = slotsWhere(field, (_q, r) => r > 0);
    expect(upper.length).toBe(100);
    expect(lower.length).toBe(100);

    const paint = makePaint(field, { on: bridge });
    const state = oracle(field, paint);
    expectRegion(field, state, (_q, r) => r === 0, ON, 'bridge');
    expectRegion(field, state, (_q, r) => r < 0, OFF, 'upper (tied)');
    expectRegion(field, state, (_q, r) => r > 0, OFF, 'lower (tied)');
  });

  it('a near-tie: a row one off centre fills the smaller side', () => {
    // R = 8: row(r) = 2R+1-|r|. Row r=1 is a full edge-to-edge bridge
    // (2R = 16 cells); it splits the field into r<=0 (117 cells, the
    // larger side) and r>1 (84 cells, the smaller side).
    const R = 8;
    const field = hexField(R);
    const bridge = slotsWhere(field, (_q, r) => r === 1);
    expect(bridge.length).toBe(2 * R);
    const upper = slotsWhere(field, (_q, r) => r <= 0);
    const lower = slotsWhere(field, (_q, r) => r > 1);
    expect(upper.length).toBe(117);
    expect(lower.length).toBe(84);

    const state = oracle(field, makePaint(field, { on: bridge }));
    expectRegion(field, state, (_q, r) => r <= 0, OFF, 'larger side');
    expectRegion(field, state, (_q, r) => r > 1, FILLED, 'smaller side');
  });

  it('nested loops fill both interiors, not the outer ring', () => {
    // R = 10: loops at d=2 and d=6. The inner disc (d<2) and the middle
    // annulus (3<=d<=5) never touch the dead ring, so both fill regardless
    // of size; the outer ring (d>=7) is the field's only edge region, so it
    // alone stays OFF.
    const R = 10;
    const field = hexField(R);
    const innerLoop = slotsWhere(field, (_q, _r, d) => d === 2);
    const outerLoop = slotsWhere(field, (_q, _r, d) => d === 6);
    expect(innerLoop.length).toBe(6 * 2);
    expect(outerLoop.length).toBe(6 * 6);

    const state = oracle(field, makePaint(field, { on: [...innerLoop, ...outerLoop] }));
    expectRegion(field, state, (_q, _r, d) => d < 2, FILLED, 'inner disc');
    expectRegion(field, state, (_q, _r, d) => d === 2, ON, 'inner loop');
    expectRegion(field, state, (_q, _r, d) => d >= 3 && d <= 5, FILLED, 'middle annulus');
    expectRegion(field, state, (_q, _r, d) => d === 6, ON, 'outer loop');
    expectRegion(field, state, (_q, _r, d) => d >= 7, OFF, 'outer ring (sole edge region)');
  });

  it('a loop one in from the border fills everything inside', () => {
    const R = 8;
    const field = hexField(R);
    const loop = slotsWhere(field, (_q, _r, d) => d === R - 1);
    expect(loop.length).toBe(6 * (R - 1));

    const state = oracle(field, makePaint(field, { on: loop }));
    expectRegion(field, state, (_q, _r, d) => d < R - 1, FILLED, 'everything inside');
    expectRegion(field, state, (_q, _r, d) => d === R, OFF, 'outer ring (sole edge region)');
  });

  it('a loop round a wall island fills the annulus between them', () => {
    // A WALL disc (d<=1) in the centre is an island, not connected to the
    // dead ring: by default it is not "exterior", so the annulus it shares
    // a border with does not touch X and fills regardless of size. The
    // outer region (d>=5) is the field's only edge region and stays OFF.
    const R = 8;
    const field = hexField(R);
    const island = slotsWhere(field, (_q, _r, d) => d <= 1);
    const loop = slotsWhere(field, (_q, _r, d) => d === 4);
    expect(island.length).toBe(7);

    const paint = makePaint(field, { wall: island, on: loop });
    const state = oracle(field, paint);
    expectRegion(field, state, (_q, _r, d) => d <= 1, WALL, 'wall island');
    expectRegion(field, state, (_q, _r, d) => d === 2 || d === 3, FILLED, 'annulus (not edge)');
    expectRegion(field, state, (_q, _r, d) => d === 4, ON, 'loop');
    expectRegion(field, state, (_q, _r, d) => d >= 5, OFF, 'outer (sole edge region)');

    // With wallsBound every WALL cell counts as exterior, so the wall
    // island itself becomes part of X and the annulus becomes an edge
    // region too — but it is still the *smaller* one (30 cells against the
    // outer region's 156), so it still fills, just via the other clause of
    // the fill rule (size < M rather than !edge).
    const annulus = slotsWhere(field, (_q, _r, d) => d === 2 || d === 3);
    const outer = slotsWhere(field, (_q, _r, d) => d >= 5);
    expect(annulus.length).toBeLessThan(outer.length);
    const bound = oracle(field, paint, true);
    expectRegion(field, bound, (_q, _r, d) => d <= 1, WALL, 'wall island (bound)');
    expectRegion(field, bound, (_q, _r, d) => d === 2 || d === 3, FILLED, 'annulus still fills (bound)');
    expectRegion(field, bound, (_q, _r, d) => d >= 5, OFF, 'outer still the larger edge region (bound)');
  });

  it('two parallel bridges fill the two smaller strips, not the middle', () => {
    // R = 8: rows r=-3 and r=5 are both full edge-to-edge bridges. They cut
    // the field into a top strip (r<=-4, 55 cells), a middle strip
    // (-2<=r<=4, 106 cells) and a bottom strip (r>=6, 30 cells); the middle
    // is the largest, so it alone stays OFF.
    const R = 8;
    const field = hexField(R);
    const bridgeA = slotsWhere(field, (_q, r) => r === -3);
    const bridgeB = slotsWhere(field, (_q, r) => r === 5);
    const top = slotsWhere(field, (_q, r) => r <= -4);
    const middle = slotsWhere(field, (_q, r) => r >= -2 && r <= 4);
    const bottom = slotsWhere(field, (_q, r) => r >= 6);
    expect(top.length).toBe(55);
    expect(middle.length).toBe(106);
    expect(bottom.length).toBe(30);

    const state = oracle(field, makePaint(field, { on: [...bridgeA, ...bridgeB] }));
    expectRegion(field, state, (_q, r) => r <= -4, FILLED, 'top strip (smaller)');
    expectRegion(field, state, (_q, r) => r >= -2 && r <= 4, OFF, 'middle strip (largest)');
    expectRegion(field, state, (_q, r) => r >= 6, FILLED, 'bottom strip (smaller)');
  });

  it('a wall ring near the border with an inner loop fills only the loop', () => {
    // R = 9: a wall ring occupies d in [R-2, R], reaching the dead ring
    // directly, so it is exterior with or without wallsBound. A loop at
    // d=3 separates the interior (d<3, never touching X) from the annulus
    // (4<=d<=6), which touches the wall ring and is the field's only edge
    // region, so it alone stays OFF.
    const R = 9;
    const field = hexField(R);
    const wallRing = slotsWhere(field, (_q, _r, d) => d >= R - 2);
    const loop = slotsWhere(field, (_q, _r, d) => d === 3);

    const state = oracle(field, makePaint(field, { wall: wallRing, on: loop }));
    expectRegion(field, state, (_q, _r, d) => d >= R - 2, WALL, 'wall ring');
    expectRegion(field, state, (_q, _r, d) => d < 3, FILLED, "the loop's interior");
    expectRegion(field, state, (_q, _r, d) => d === 3, ON, 'loop');
    expectRegion(field, state, (_q, _r, d) => d >= 4 && d <= R - 3, OFF, 'annulus (sole edge region)');
  });

  it('a bridge onto that ring fills its smaller side', () => {
    // Same wall ring (R=9, d in [7,9]); a spoke at q=2 runs from the
    // interior out to the ring on both ends (its own endpoints land on
    // d=6, next to the wall), splitting the open disc (d<=6) into a q<2
    // part and a q>2 part. Count them by brute force, independent of the
    // oracle's own region search, to know which is smaller.
    const R = 9;
    const field = hexField(R);
    const wallRing = slotsWhere(field, (_q, _r, d) => d >= R - 2);
    const spoke = slotsWhere(field, (q, _r, d) => q === 2 && d <= R - 3);
    const left = slotsWhere(field, (q, _r, d) => q < 2 && d <= R - 3);
    const right = slotsWhere(field, (q, _r, d) => q > 2 && d <= R - 3);
    expect(left.length + right.length + spoke.length).toBe(
      slotsWhere(field, (_q, _r, d) => d <= R - 3).length,
    );
    expect(left.length).not.toBe(right.length);
    const [smaller, larger] = left.length < right.length ? [left, right] : [right, left];
    const smallerPred = left.length < right.length ? (q: number, _r: number, d: number) => q < 2 && d <= R - 3
      : (q: number, _r: number, d: number) => q > 2 && d <= R - 3;
    const largerPred = left.length < right.length ? (q: number, _r: number, d: number) => q > 2 && d <= R - 3
      : (q: number, _r: number, d: number) => q < 2 && d <= R - 3;

    const state = oracle(field, makePaint(field, { wall: wallRing, on: spoke }));
    expectRegion(field, state, smallerPred, FILLED, `smaller side (${smaller.length} cells)`);
    expectRegion(field, state, largerPred, OFF, `larger side (${larger.length} cells)`);
  });

  it('an empty board fills nothing', () => {
    const field = hexField(8);
    const state = oracle(field, makePaint(field, {}));
    for (const slot of field.topo.cells) expect(state[slot]).toBe(OFF);
  });
});

describe('settleBound (DESIGN.md §5)', () => {
  // A filled hexagon is a metric "ball": for a cell at hex distance d from
  // the centre, its eccentricity over the whole field is exactly R + d
  // (straight line to the opposite side) — independent of the oracle/engine
  // code, so it is a clean cross-check for Dw.
  it('matches the hand formula on an empty board', () => {
    const R = 6;
    const field = hexField(R);
    const paint = makePaint(field, {});
    const K = field.K;
    expect(K).toBe(R);

    // Dw: the edited cell's eccentricity. Edit the root (centre, d=0): R+0.
    const atRoot = settleBound(field, paint, [field.root]);
    const DwRoot = R + 0;
    // Er: the sole region is the whole (empty) field; its leader is the
    // max-id cell, and (same ball fact) its eccentricity is R + its own
    // distance from the centre.
    let leader = field.topo.cells[0];
    for (const s of field.topo.cells) if (field.ids[s] > field.ids[leader]) leader = s;
    const [lq, lr] = coordsOf(field.board, leader);
    const Er = R + hexDistance(lq, lr);
    // No walls: Wd = 0, so the rim branch is 0 + 2·Er and 3·Er wins.
    const Wd = 0;
    const expected = DwRoot + Math.max(3 * Er, Wd + 2 * Er) + 3 * K + 3;
    expect(atRoot).toBe(expected);

    // Editing a border cell instead only changes Dw (R + R = 2R).
    let border = field.root;
    for (const s of field.topo.cells) {
      const [q, r] = coordsOf(field.board, s);
      if (hexDistance(q, r) === R) {
        border = s;
        break;
      }
    }
    const atBorder = settleBound(field, paint, [border]);
    expect(atBorder).toBe(2 * R + Math.max(3 * Er, Wd + 2 * Er) + 3 * K + 3);
  });

  it('grows with K and is unaffected by edits outside the field', () => {
    const field = hexField(5);
    const paint = makePaint(field, {});
    const withNoEdit = settleBound(field, paint, []);
    const withOffBoard = settleBound(field, paint, [-1, field.board.size + 5]);
    expect(withOffBoard).toBe(withNoEdit); // ignored: not field slots
    expect(settleBound(hexField(9), makePaint(hexField(9), {}), [])).toBeGreaterThan(withNoEdit);
  });
});
