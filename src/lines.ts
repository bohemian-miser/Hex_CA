// Random lines across the board, and the reference answer the CA must match.
// Neither is part of the automaton: they set it up and check it.

import { Board, borderRing, coordsOf, hexDistance, indexOf, isBorder } from './hex.js';

/** mulberry32: small, seedable, good enough for test fields. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface LineOptions {
  /** Cells the line may not use (old lines). */
  blocked?: ReadonlySet<number>;
  /** How much the line wanders: 0 = straight-ish, larger = curlier. Default 1. */
  wiggle?: number;
}

/** Smooth random cell costs (a few random bumps): cheapest paths through it wander. */
function costField(board: Board, rand: () => number, wiggle: number): Float64Array {
  const cost = new Float64Array(board.size);
  const R = board.radius;
  const bumps = Array.from({ length: 6 + Math.floor(rand() * 6) }, () => ({
    q: (rand() * 2 - 1) * R,
    r: (rand() * 2 - 1) * R,
    w: 0.15 + rand() * 0.5,
    h: rand() * 2 - 0.6,
  }));
  for (const i of board.cells) {
    const [q, r] = coordsOf(board, i);
    let v = 0;
    for (const b of bumps) {
      const dist = hexDistance(q - b.q, r - b.r) / (R * b.w + 1);
      v += b.h * Math.exp(-dist * dist);
    }
    cost[i] = 1 + wiggle * 12 * Math.max(0, v + 0.3) ** 2 + rand() * 0.5;
  }
  return cost;
}

/**
 * A random bridge: a line from one border cell to another, through the
 * interior, in drawing order.
 *
 * It touches the border at its two ends and nowhere else, and it is an
 * *induced* path (no two cells are neighbours unless consecutive), so drawn
 * cell by cell it never closes a circuit on the way. Built as a cheapest path
 * under random, smoothly varying costs: with positive costs a cheapest path is
 * always induced (any shortcut would be cheaper). Returns null if the blocked
 * cells leave no way across.
 */
export function randomLine(board: Board, rand: () => number, opts: LineOptions = {}): number[] | null {
  const blocked = opts.blocked ?? new Set<number>();
  const ring = borderRing(board);
  const n = ring.length;
  if (n < 4) return null;
  const cost = costField(board, rand, opts.wiggle ?? 1);
  for (let attempt = 0; attempt < 40; attempt++) {
    const a = Math.floor(rand() * n);
    // The other end at least a sixth of the way round either way.
    const gap = Math.floor(n / 6) + Math.floor(rand() * (n - 2 * Math.floor(n / 6)));
    const b = (a + Math.max(2, Math.min(n - 2, gap))) % n;
    const A = ring[a];
    const B = ring[b];
    if (blocked.has(A) || blocked.has(B)) continue;
    const path = cheapestPath(board, cost, A, B, (i) => i === B || (!isBorder(board, i) && !blocked.has(i)));
    if (path && path.length >= 3) return path;
  }
  return null;
}

/**
 * A random circuit: a loop in drawing order, closing when its last cell
 * lands next to its first. It goes round a random centre: the first cell sits
 * on a ray cut from the centre to the edge, and the rest is a cheapest path
 * from one side of the cut round to the other, so it can't touch itself
 * early. Clockwise or counter-clockwise at random. Stays off the border.
 */
export function randomLoop(board: Board, rand: () => number, opts: LineOptions = {}): number[] | null {
  const blocked = opts.blocked ?? new Set<number>();
  const R = board.radius;
  if (R < 4) return null;
  const cost = costField(board, rand, opts.wiggle ?? 1);
  for (let attempt = 0; attempt < 40; attempt++) {
    const span = Math.max(1, Math.floor(R * 0.6));
    const qc = Math.floor((rand() * 2 - 1) * span);
    const rc = Math.floor((rand() * 2 - 1) * span);
    if (hexDistance(qc, rc) > R - 3) continue;
    const core = Math.floor(rand() * Math.min(3, R / 4));
    // The cut: the row through the centre, from the centre east to the edge.
    const cut = new Set<number>();
    for (let q = qc; hexDistance(q, rc) <= R; q++) cut.add(indexOf(board, q, rc));
    const start = indexOf(board, qc + core + 1, rc); // on the cut, just outside the core
    const [sq, sr] = coordsOf(board, start);
    if (hexDistance(sq, sr) > R - 2) continue;
    const up = [indexOf(board, sq, sr - 1), indexOf(board, sq + 1, sr - 1)];
    const down = [indexOf(board, sq - 1, sr + 1), indexOf(board, sq, sr + 1)];
    const a = up[Math.floor(rand() * 2)];
    const b = down[Math.floor(rand() * 2)];
    // Keep the start's other neighbours free, so the loop only meets its start at the two ends.
    const shut = new Set([...up, ...down].filter((c) => c !== a && c !== b));
    const coreCells = new Set<number>();
    for (const i of board.cells) {
      const [q, r] = coordsOf(board, i);
      if (hexDistance(q - qc, r - rc) <= core) coreCells.add(i);
    }
    const ok = (i: number) =>
      !cut.has(i) && !shut.has(i) && !coreCells.has(i) && !blocked.has(i) && !isBorder(board, i);
    if (blocked.has(start) || !ok(a) || !ok(b)) continue;
    const path = cheapestPath(board, cost, a, b, ok);
    if (!path || path.length < 4) continue;
    const loop = [start, ...path];
    return rand() < 0.5 ? loop : [start, ...path.reverse()];
  }
  return null;
}

/**
 * A random scribble, the way a person might drag: a wandering walk with
 * momentum that may hug the border, turn sharply, run into itself (a circuit),
 * reach the border again (a bridge) or just stop (an open line).
 */
export function randomScribble(board: Board, rand: () => number, opts: LineOptions & { maxLength?: number } = {}): number[] {
  const blocked = opts.blocked ?? new Set<number>();
  const free = board.cells.filter((i) => !blocked.has(i));
  const start = free[Math.floor(rand() * free.length)];
  const walk = [start];
  const used = new Set(walk);
  let dir = Math.floor(rand() * 6);
  const max = opts.maxLength ?? board.radius * 6;
  while (walk.length < max) {
    const h = walk[walk.length - 1];
    const options: Array<[number, number]> = [];
    for (const t of [0, 1, -1, 2, -2]) {
      const d = (dir + t + 6) % 6;
      const j = board.neighbours[h * 6 + d];
      if (j < 0 || !board.inside[j] || used.has(j) || blocked.has(j)) continue;
      options.push([d, t === 0 ? 6 : Math.abs(t) === 1 ? 2.5 : 0.6]);
    }
    if (!options.length) break;
    let x = rand() * options.reduce((s, o) => s + o[1], 0);
    let pick = options[0][0];
    for (const [d, w] of options) {
      if ((x -= w) <= 0) {
        pick = d;
        break;
      }
    }
    dir = pick;
    const j = board.neighbours[h * 6 + pick];
    walk.push(j);
    used.add(j);
  }
  return walk;
}

function cheapestPath(
  board: Board,
  cost: Float64Array,
  from: number,
  to: number,
  allowed: (i: number) => boolean,
): number[] | null {
  const dist = new Float64Array(board.size).fill(Infinity);
  const prev = new Int32Array(board.size).fill(-1);
  // Binary heap of [dist, index].
  const heap: Array<[number, number]> = [];
  const push = (e: [number, number]) => {
    heap.push(e);
    let k = heap.length - 1;
    while (k > 0) {
      const p = (k - 1) >> 1;
      if (heap[p][0] <= heap[k][0]) break;
      [heap[p], heap[k]] = [heap[k], heap[p]];
      k = p;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let k = 0;
      for (;;) {
        const l = 2 * k + 1;
        const r = l + 1;
        let m = k;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
        if (m === k) break;
        [heap[m], heap[k]] = [heap[k], heap[m]];
        k = m;
      }
    }
    return top;
  };
  dist[from] = cost[from];
  push([dist[from], from]);
  while (heap.length) {
    const [dv, v] = pop();
    if (dv > dist[v]) continue;
    if (v === to) break;
    for (let d = 0; d < 6; d++) {
      const w = board.neighbours[v * 6 + d];
      if (w < 0 || !board.inside[w] || !allowed(w)) continue;
      const nd = dv + cost[w];
      if (nd < dist[w]) {
        dist[w] = nd;
        prev[w] = v;
        push([nd, w]);
      }
    }
  }
  if (dist[to] === Infinity) return null;
  const path: number[] = [];
  for (let v = to; v !== -1; v = prev[v]) path.push(v);
  return path.reverse();
}

/**
 * A random wall picture: a few bridges, loops and scribbles, some broken or
 * crossing, plus scattered single cells.
 */
export function randomWalls(board: Board, rand: () => number): Set<number> {
  const walls = new Set<number>();
  const shapes = 1 + Math.floor(rand() * 4);
  for (let k = 0; k < shapes; k++) {
    const kind = rand();
    const cells =
      kind < 0.35 ? randomLine(board, rand, { wiggle: rand() * 2 })
      : kind < 0.7 ? randomLoop(board, rand, { wiggle: rand() * 2 })
      : randomScribble(board, rand, { maxLength: 4 + Math.floor(rand() * board.radius * 4) });
    if (!cells) continue;
    // Sometimes leave a gap.
    const gap = rand() < 0.25 ? Math.floor(rand() * cells.length) : -1;
    cells.forEach((c, i) => {
      if (i !== gap) walls.add(c);
    });
  }
  const specks = Math.floor(rand() * board.radius);
  for (let k = 0; k < specks; k++) walls.add(board.cells[Math.floor(rand() * board.cells.length)]);
  return walls;
}
