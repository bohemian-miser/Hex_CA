// Hexagon field: axial coordinates (q, r), pointy-top hexagons.
//
// The board is every cell within hex distance `radius` of the centre. It is
// stored in a square array with one extra ring of `outside` cells round it,
// so every inside cell has all six neighbours in the array.

/** Neighbour offsets, indexed by direction. Direction d and (d + 3) % 6 are opposite. */
export const DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0], //  0 E
  [1, -1], // 1 NE
  [0, -1], // 2 NW
  [-1, 0], // 3 W
  [-1, 1], // 4 SW
  [0, 1], //  5 SE
];

export const opp = (d: number): number => (d + 3) % 6;

export function hexDistance(q: number, r: number): number {
  return Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r));
}

export interface Board {
  radius: number;
  /** Side of the square backing array: 2 * (radius + 1) + 1. */
  side: number;
  /** Number of slots in the backing array (side²). */
  size: number;
  /** Whether a slot is on the board. */
  inside: Uint8Array;
  /** neighbours[i * 6 + d] = slot index of i's neighbour in direction d, or -1 off the array. */
  neighbours: Int32Array;
  /** Every inside slot, row by row. */
  cells: number[];
}

export function makeBoard(radius: number): Board {
  const off = radius + 1;
  const side = 2 * off + 1;
  const size = side * side;
  const inside = new Uint8Array(size);
  const neighbours = new Int32Array(size * 6).fill(-1);
  const cells: number[] = [];
  for (let r = -off; r <= off; r++) {
    for (let q = -off; q <= off; q++) {
      const i = (r + off) * side + (q + off);
      if (hexDistance(q, r) <= radius) {
        inside[i] = 1;
        cells.push(i);
      }
      for (let d = 0; d < 6; d++) {
        const nq = q + DIRS[d][0] + off;
        const nr = r + DIRS[d][1] + off;
        if (nq >= 0 && nq < side && nr >= 0 && nr < side) neighbours[i * 6 + d] = nr * side + nq;
      }
    }
  }
  return { radius, side, size, inside, neighbours, cells };
}

export function coordsOf(board: Board, i: number): [number, number] {
  const off = board.radius + 1;
  return [(i % board.side) - off, Math.floor(i / board.side) - off];
}

export function indexOf(board: Board, q: number, r: number): number {
  const off = board.radius + 1;
  return (r + off) * board.side + (q + off);
}

/** Whether an inside cell touches the outside (the border ring). */
export function isBorder(board: Board, i: number): boolean {
  if (!board.inside[i]) return false;
  for (let d = 0; d < 6; d++) {
    const j = board.neighbours[i * 6 + d];
    if (j < 0 || !board.inside[j]) return true;
  }
  return false;
}

/**
 * The border ring in order, as a cycle of slot indices (counter-clockwise on
 * screen). Used by the oracle and the line generator, never by the CA.
 */
export function borderRing(board: Board): number[] {
  const R = board.radius;
  if (R === 0) return [indexOf(board, 0, 0)];
  const ring: number[] = [];
  // Walk the six sides of the hexagon: start at the E corner, step NW-wards...
  let q = R;
  let r = 0;
  const walk = [2, 3, 4, 5, 0, 1]; // NW, W, SW, SE, E, NE
  for (const d of walk) {
    for (let k = 0; k < R; k++) {
      ring.push(indexOf(board, q, r));
      q += DIRS[d][0];
      r += DIRS[d][1];
    }
  }
  return ring;
}
