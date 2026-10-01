// Random lines across the board, and the reference answer the CA must match.
// Neither is part of the automaton: they set it up and check it.

import { Board, borderRing, coordsOf, hexDistance, isBorder } from './hex.js';

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

/**
 * A random line from one border cell to another, through the interior.
 *
 * Guarantees what the automaton assumes of a line:
 *   - it touches the border at its two ends (the feet) and nowhere else;
 *   - it is an *induced* path: no two cells of it are neighbours unless they
 *     are consecutive, so it never pinches off a pocket;
 *   - both arcs of the ring between the feet have at least one cell.
 *
 * Built as a cheapest path under random, smoothly varying cell costs: a
 * cheapest path with positive node costs is always induced (any shortcut
 * would be cheaper). Returns null if the blocked cells leave no way across.
 */
export function randomLine(board: Board, rand: () => number, opts: LineOptions = {}): number[] | null {
  const blocked = opts.blocked ?? new Set<number>();
  const wiggle = opts.wiggle ?? 1;
  const ring = borderRing(board);
  const n = ring.length;
  if (n < 4) return null;

  // Smooth random cost field: a few random bumps.
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

  for (let attempt = 0; attempt < 40; attempt++) {
    const a = Math.floor(rand() * n);
    // The other foot at least a sixth of the way round either way.
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

export interface Expected {
  /** Cells on each arc between the feet (border cells not on the line). */
  arcs: [number[], number[]];
  /** Index into `arcs` of the shorter one, or -1 on a tie. */
  shorter: number;
  /** The cells the flood should cover. */
  fill: Set<number>;
}

/**
 * The reference answer, computed globally: walk the ring, split it at the
 * line's feet, take the shorter arc, breadth-first fill from it without
 * crossing the line.
 */
export function expectedFill(board: Board, line: readonly number[]): Expected {
  const onLine = new Set(line);
  const ring = borderRing(board);
  const feet = ring.map((c, k) => (onLine.has(c) ? k : -1)).filter((k) => k >= 0);
  if (feet.length !== 2) throw new Error(`line touches the border ${feet.length} times, expected 2`);
  const [a, b] = feet;
  const arc0 = ring.slice(a + 1, b);
  const arc1 = [...ring.slice(b + 1), ...ring.slice(0, a)];
  const shorter = arc0.length < arc1.length ? 0 : arc1.length < arc0.length ? 1 : -1;
  const fill = new Set<number>();
  if (shorter >= 0) {
    const queue = [...(shorter === 0 ? arc0 : arc1)];
    for (const c of queue) fill.add(c);
    for (let k = 0; k < queue.length; k++) {
      const v = queue[k];
      for (let d = 0; d < 6; d++) {
        const w = board.neighbours[v * 6 + d];
        if (w < 0 || !board.inside[w] || onLine.has(w) || fill.has(w)) continue;
        fill.add(w);
        queue.push(w);
      }
    }
  }
  return { arcs: [arc0, arc1], shorter, fill };
}
