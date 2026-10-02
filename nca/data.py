"""Random wall pictures and the flood-fill oracle (ground truth for training).

Everything here is deterministic given the numpy Generator passed in: same
rng state in, same picture out.

Training pictures come from random_walls / batch / edit_walls. page_loops is
a port of the demo page's "Random loop" button, kept apart for evaluation (an
out-of-distribution check): nothing in training draws from it.
"""

import math
import heapq

import numpy as np

from .hexgrid import NEIGHBOURS, mask, rim, side

# The six neighbours in turning order round a cell, (drow, dcol): E, NE, NW, W, SW, SE.
_ROUND = ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0))
_SQ3 = math.sqrt(3.0)


# --------------------------------------------------------------------------
# Small array helpers.
# --------------------------------------------------------------------------

def _look(arr: np.ndarray, dr: int, dc: int, fill=False) -> np.ndarray:
    """result[row,col] = arr[row+dr, col+dc]; `fill` where that's out of bounds."""
    out = np.full_like(arr, fill)
    rows, cols = arr.shape
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(cols, cols - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = arr[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def _dilate(arr: np.ndarray) -> np.ndarray:
    """Bool array, true where `arr` itself is true or a hex neighbour is."""
    out = arr.copy()
    for dr, dc in NEIGHBOURS:
        out |= _look(arr, dr, dc)
    return out


def _axial_dist(R: int, cq: float, cr: float) -> np.ndarray:
    """Hex distance of every array cell's (q, r) from centre (cq, cr)."""
    S = side(R)
    r = (np.arange(S) - R).reshape(S, 1) - cr
    q = (np.arange(S) - R).reshape(1, S) - cq
    return np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))


_NB = {}


def _nbrs(R: int):
    """Per flat index row*S+col: list of the flat indices of its on-board neighbours (cached per R)."""
    if R not in _NB:
        S, on = side(R), mask(R) == 1
        _NB[R] = [[(r + dr) * S + c + dc for dr, dc in NEIGHBOURS
                   if on[r, c] and 0 <= r + dr < S and 0 <= c + dc < S and on[r + dr, c + dc]]
                  for r in range(S) for c in range(S)]
    return _NB[R]


def _offset(rng: np.random.Generator, d: int):
    """A uniformly random (dq, dr) within hex distance d of (0, 0)."""
    while True:
        q, r = (int(x) for x in rng.integers(-d, d + 1, 2))
        if abs(q + r) <= d:
            return q, r


# --------------------------------------------------------------------------
# The oracle.
# --------------------------------------------------------------------------

def oracle(walls: np.ndarray, R: int):
    """(fill uint8 [S,S], depth int16 [S,S]).

    Outside = on-board non-wall cells 6-connected, through on-board non-wall
    cells, to a non-wall rim cell (a non-wall rim cell is itself outside).
    fill = 1 on on-board non-wall cells that are not outside; 0 on walls,
    outside cells and off-board cells. depth = BFS distance of each outside
    cell from the rim (rim itself = 0); -1 where not outside.
    """
    S = side(R)
    open_cells = (mask(R) == 1) & (walls == 0)
    free = open_cells.ravel().tolist()
    nb = _nbrs(R)
    depth = [-1] * (S * S)
    queue = np.flatnonzero(open_cells & rim(R)).tolist()
    for i in queue:
        depth[i] = 0
    for i in queue:  # grows as it goes: a BFS
        d = depth[i] + 1
        for j in nb[i]:
            if free[j] and depth[j] < 0:
                depth[j] = d
                queue.append(j)
    depth = np.array(depth, dtype=np.int16).reshape(S, S)
    fill = (open_cells & (depth < 0)).astype(np.uint8)
    return fill, depth


# --------------------------------------------------------------------------
# Closed shapes. Each takes a centre (cq, cr) and a rough radius s and returns
# bool [S,S] walls (clipped to the board later).
# --------------------------------------------------------------------------

def _walk(rng, R, on_board, row, col, length, wiggle):
    """A stroke from (row, col): `length` cells, straight (wiggle 0) or turning now and then; stops at the edge."""
    out = np.zeros_like(on_board)
    S = side(R)
    d = int(rng.integers(6))
    turns = rng.choice((0, 1, -1, 2, -2), size=length, p=(0.5, 0.2, 0.2, 0.05, 0.05)) if wiggle else [0] * length
    for t in turns:
        if not (0 <= row < S and 0 <= col < S and on_board[row, col]):
            break
        out[row, col] = True
        d = (d + int(t)) % 6
        row, col = row + _ROUND[d][0], col + _ROUND[d][1]
    return out


def _grow(rng, R, seeds, n):
    """bool [S,S]: a region grown from the seed cells (flat indices) one random frontier cell at a time to n cells."""
    nb = _nbrs(R)
    cells = set(seeds)
    front = [j for i in seeds for j in nb[i]]
    for x in rng.random(8 * n + 8).tolist():
        if len(cells) >= n or not front:
            break
        i = int(x * len(front))
        c = front[i]
        front[i] = front[-1]
        front.pop()
        if c not in cells:
            cells.add(c)
            front.extend(nb[c])
    g = np.zeros(side(R) ** 2, dtype=bool)
    g[list(cells)] = True
    return g.reshape(side(R), side(R))


def _blob(rng, R, on_board, cq, cr, s):
    """The outline of a random region: its cells that have a non-region neighbour.

    The region is grown cell by cell (compact, sometimes from a straight seed
    so it comes out elongated) or is a thickened winding stroke (a snake: a
    corridor inside). Either way the inside is enclosed by construction --
    unless the board edge cuts it open: then the region carries on past the
    rim, and its rim cells are no wall.
    """
    row, col = cr + R, cq + R
    if rng.random() < 0.3:  # snake
        g = _walk(rng, R, on_board, row, col, int(s * rng.uniform(2, 7)) + 1, wiggle=True)
        b = _dilate(g) & on_board
    else:
        thick = rng.random() < 0.5  # grow smaller, then add a ring of cells round it
        sg = max(0.0, s - 1) if thick else s
        n = int(3 * sg * (sg + 1) * rng.uniform(0.3, 1.0)) + 1
        seeds = _walk(rng, R, on_board, row, col, 1 if rng.random() < 0.6 else int(rng.integers(1, int(s) + 2)), False)
        g = _grow(rng, R, np.flatnonzero(seeds).tolist(), n)
        b = _dilate(g) & on_board if thick else g
    beyond = rng.random() < 0.35
    ext = b | ~on_board if beyond else b
    inner = b.copy()
    for dr, dc in NEIGHBOURS:
        inner &= _look(ext, dr, dc, fill=beyond)
    return b & ~inner


def _polyline(R, pts, closed):
    """bool [S,S]: the 6-connected hex line through float axial points (q, r), closed or not."""
    S = side(R)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    qs, rs = [], []
    for (q0, r0), (q1, r1) in zip(pts[:-1], pts[1:]):
        dq, dr = q1 - q0, r1 - r0
        # Samples under 0.4 apart: a segment only ever skips a cell it clips at a
        # corner, and the cells either side of such a corner are neighbours.
        t = np.linspace(0, 1, int(2.5 * max(abs(dq), abs(dr), abs(dq + dr))) + 2)
        qs.append(q0 + dq * t)
        rs.append(r0 + dr * t)
    q, r = np.concatenate(qs), np.concatenate(rs)
    # Cube rounding: round all three coordinates, then fix the one that moved most.
    s = -q - r
    rq, rr, rs_ = np.rint(q), np.rint(r), np.rint(s)
    eq, er, es = np.abs(rq - q), np.abs(rr - r), np.abs(rs_ - s)
    fix_q = (eq > er) & (eq > es)
    fix_r = ~fix_q & (er > es)
    rq = np.where(fix_q, -rr - rs_, rq)
    rr = np.where(fix_r, -rq - rs_, rr)
    row, col = rr.astype(int) + R, rq.astype(int) + R
    ok = (row >= 0) & (row < S) & (col >= 0) & (col < S)
    out = np.zeros((S, S), dtype=bool)
    out[row[ok], col[ok]] = True
    return out


def _polar(cq, cr, ang, rad):
    """Float axial (q, r) of the points at angles `ang`, distances `rad` from (cq, cr)."""
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    return np.stack([cq + x - y / _SQ3, cr + 2 * y / _SQ3], axis=1)


def _poly_points(rng, cq, cr, s):
    """3-8 points round (cq, cr) in angle order at random distances up to s."""
    k = int(rng.integers(3, 9))
    ang = (np.arange(k) + rng.uniform(-0.4, 0.4, k)) * (2 * np.pi / k) + rng.uniform(0, 2 * np.pi)
    return _polar(cq, cr, ang, s * rng.uniform(0.3, 1.0, k))


def _spiral_points(rng, cq, cr, s):
    """A spiral from distance s in towards (cq, cr), 1-2 turns: a long winding way in for the outside."""
    turns = rng.uniform(1.0, 2.0)
    t = np.linspace(0, 1, int(8 * turns) + 2)
    ang = rng.choice((-1, 1)) * t * turns * 2 * np.pi + rng.uniform(0, 2 * np.pi)
    return _polar(cq, cr, ang, s * (1 - 0.8 * t))


def _poly(rng, R, on_board, cq, cr, s):
    """A closed polygon round (cq, cr), drawn as a thin hex line: a loop the way a person might draw one."""
    return _polyline(R, _poly_points(rng, cq, cr, s), closed=True)


def _ring(rng, R, on_board, cq, cr, s):
    """A hexagon ring of radius ~s, its corners often nudged off true."""
    corners = np.array([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)], float) * max(1, round(s)) + (cq, cr)
    if rng.random() < 0.6:
        corners += rng.uniform(-0.7, 0.7, corners.shape)
    return _polyline(R, corners, closed=True)


_SHAPES = {"blob": _blob, "poly": _poly, "ring": _ring}
_SHAPE_P = (0.45, 0.4, 0.15)


def _closed(rng, R, on_board, first=None):
    """1-3 closed shapes: side by side, overlapping (sharing walls), nested, or running over the rim."""
    walls = np.zeros_like(on_board)
    prev = None
    for i in range(int(rng.choice((1, 2, 3), p=(0.5, 0.3, 0.2)))):
        shape = first if i == 0 and first else rng.choice(list(_SHAPES), p=_SHAPE_P)
        if prev and rng.random() < 0.35:  # inside (or across) the last one
            pq, pr, ps = prev
            s = max(1.0, ps * rng.uniform(0.25, 0.6))
            dq, dr = _offset(rng, max(0, int(ps - s - 1)))
            cq, cr = (pq + dq, pr + dr) if max(abs(pq + dq), abs(pr + dr), abs(pq + dq + pr + dr)) <= R else (pq, pr)
        else:
            s = 1 + max(0, R - 1) * rng.random()
            cq, cr = _offset(rng, R if rng.random() < 0.25 else max(0, int(R - s - 1)))
        walls |= _SHAPES[shape](rng, R, on_board, cq, cr, s)
        prev = (cq, cr, s)
    return walls


# --------------------------------------------------------------------------
# Wall picture kinds.
# --------------------------------------------------------------------------

def _knock_out(rng, walls, n):
    """Remove n random wall cells (in place)."""
    idx = np.argwhere(walls)
    for i in rng.choice(len(idx), size=min(n, len(idx)), replace=False):
        walls[tuple(idx[i])] = False
    return walls


def _strokes(rng, R, on_board, n):
    """n open strokes: straight lines, wandering walks, open polylines (a loop that doesn't close), spirals."""
    walls = np.zeros_like(on_board)
    cells = np.argwhere(on_board)
    for _ in range(n):
        row, col = (int(x) for x in cells[rng.integers(len(cells))])
        u = rng.random()
        if u < 0.3:
            walls |= _walk(rng, R, on_board, row, col, int(rng.integers(1, 2 * R + 3)), wiggle=False)
        elif u < 0.6:
            walls |= _walk(rng, R, on_board, row, col, int(rng.integers(2, 3 * R + 4)), wiggle=True)
        else:
            pts = _poly_points if u < 0.85 else _spiral_points
            walls |= _polyline(R, pts(rng, col - R, row - R, 1 + max(0, R - 1) * rng.random()), closed=False)
    return walls


def _walls_leaky(rng, R, on_board):
    """Closed shapes with 1-3 wall cells knocked out: some drain, some (behind another wall) still fill."""
    return _knock_out(rng, _closed(rng, R, on_board), int(rng.integers(1, 4)))


def _walls_messy(rng, R, on_board):
    """Closed shapes plus clutter: stray strokes, a sprinkle of single cells, stubs off the walls, doubled walls."""
    walls = _closed(rng, R, on_board)
    if rng.random() < 0.5:
        walls |= _strokes(rng, R, on_board, int(rng.integers(1, 3)))
    if rng.random() < 0.4:
        walls |= rng.random(walls.shape) < rng.uniform(0.01, 0.06)
    if rng.random() < 0.4 and walls.any():  # stubs: short straight bits sticking out of a wall
        idx = np.argwhere(walls)
        for i in rng.integers(len(idx), size=int(rng.integers(1, 5))):
            walls |= _walk(rng, R, on_board, int(idx[i][0]), int(idx[i][1]), int(rng.integers(2, 5)), False)
    if rng.random() < 0.3:  # doubled walls: the walls shifted one cell, near a random centre
        dr, dc = _ROUND[int(rng.integers(6))]
        cq, cr = _offset(rng, R)
        walls |= _look(walls, dr, dc) & (_axial_dist(R, cq, cr) <= rng.uniform(1, 2 * R + 2))
    if rng.random() < 0.2:
        _knock_out(rng, walls, 1)
    return walls


def _walls_open(rng, R, on_board):
    return _strokes(rng, R, on_board, int(rng.integers(1, 4)))


def _walls_noise(rng, R, on_board):
    return rng.random(on_board.shape) < rng.uniform(0.03, 0.45)


_KINDS = {  # each (rng, R, on_board) -> bool [S,S] walls
    "blob": lambda rng, R, on_board: _closed(rng, R, on_board, "blob"),
    "poly": lambda rng, R, on_board: _closed(rng, R, on_board, "poly"),
    "ring": lambda rng, R, on_board: _closed(rng, R, on_board, "ring"),
    "leaky": _walls_leaky,
    "messy": _walls_messy,
    "open": _walls_open,
    "noise": _walls_noise,
}

# Weights for kind=None, tuned (see nca/selftest.py) so roughly half of all
# boards end up with at least one filled cell. "blob"/"poly"/"ring" name the
# first shape of 1-3; the others are drawn from all three.
_MIX = {"blob": 0.18, "poly": 0.15, "ring": 0.04, "leaky": 0.17, "messy": 0.17, "open": 0.14, "noise": 0.15}


def random_walls(rng: np.random.Generator, R: int, kind: str = None) -> np.ndarray:
    """uint8 [S,S] wall picture (0/1), zero off board. Deterministic given rng."""
    on_board = mask(R) == 1
    if kind is None:
        kind = rng.choice(list(_MIX.keys()), p=list(_MIX.values()))
    walls = _KINDS[kind](rng, R, on_board)
    return (walls & on_board).astype(np.uint8)


def batch(rng: np.random.Generator, R: int, n: int):
    """(walls [n,S,S] uint8, fill [n,S,S] uint8, depth [n,S,S] int16)."""
    S = side(R)
    walls = np.zeros((n, S, S), dtype=np.uint8)
    fill = np.zeros((n, S, S), dtype=np.uint8)
    depth = np.zeros((n, S, S), dtype=np.int16)
    for i in range(n):
        w = random_walls(rng, R)
        f, d = oracle(w, R)
        walls[i], fill[i], depth[i] = w, f, d
    return walls, fill, depth


# --------------------------------------------------------------------------
# Edits: what a user drawing on a settled board does.
# --------------------------------------------------------------------------

def _open_runs(free: np.ndarray) -> np.ndarray:
    """int [S,S]: how many separate runs of free cells there are round each cell (off the array counts as free)."""
    ring = [_look(free, dr, dc, fill=True) for dr, dc in _ROUND]
    return sum((ring[k] & ~ring[k - 1]).astype(int) for k in range(6))


def _drain(rng, w, R, filled, outside):
    """Knock out 1-2 wall cells that each sit between a filled cell and the outside (or on the rim): the region drains."""
    gaps = np.argwhere(w & _dilate(filled) & (_dilate(outside) | rim(R)))
    if not len(gaps):
        return None
    w = w.copy()
    for i in rng.choice(len(gaps), size=min(len(gaps), 1 + int(rng.random() < 0.3)), replace=False):
        w[tuple(gaps[i])] = False
    return w


def _seal(rng, w, R, filled, outside, tries=8):
    """Wall one outside cell that cuts some cells off from the rim (they fill), if one of a few candidates does."""
    # Walling a cell can only cut something off if the free cells round it are in two or more separate runs.
    cand = np.argwhere(outside & (_open_runs(~w) >= 2))
    for i in rng.permutation(len(cand))[:tries]:
        w2 = w.copy()
        w2[tuple(cand[i])] = True
        if oracle(w2, R)[0].sum() > filled.sum():
            return w2
    return None


def _pocket(rng, w, R, filled, outside):
    """Wall in an outside cell that is already walled on 2+ sides (adds its 1-4 free neighbours): it fills."""
    n_free = sum(_look(~w, dr, dc) for dr, dc in NEIGHBOURS)
    n_free = np.where(outside & ~rim(R), n_free, 9)  # a rim cell stays outside whatever is round it
    if n_free.min() > 4:
        return None
    cells = np.argwhere(n_free == n_free.min())  # the most walled-in first
    row, col = cells[rng.integers(len(cells))]
    w = w.copy()
    for dr, dc in NEIGHBOURS:
        w[row + dr, col + dc] = True
    return w


# Share of edits that open or seal something (the rest are random toggles):
# tuned so ~2/3 of all edits change the answer (nca/selftest.py prints it).
P_MOVE = 0.45


def edit_walls(rng: np.random.Generator, walls: np.ndarray, R: int) -> np.ndarray:
    """A small edit of a wall picture, mostly one that changes the answer.

    With probability P_MOVE it opens or seals something: knocks out a wall
    cell between a filled region and the outside (the region drains), or walls
    a gap cell that cuts cells off from the rim (they fill) -- whichever
    applies, in random order -- else walls in a half-enclosed pocket.
    Otherwise, or if none applies (an empty board), it toggles 1-4 cells among
    the walls and the cells next to them (often harmless). Always changes at
    least one on-board cell.
    """
    on_board = mask(R) == 1
    w = walls.astype(bool) & on_board
    fill, depth = oracle(w, R)
    filled, outside = fill.astype(bool), depth >= 0

    if rng.random() < P_MOVE:
        moves = (_drain, _seal) if rng.random() < 0.5 else (_seal, _drain)
        for move in moves + (_pocket,):
            out = move(rng, w, R, filled, outside)
            if out is not None:
                return out.astype(np.uint8)

    near_wall = _dilate(w) & on_board & ~w
    pools = [p for p in (np.argwhere(w), np.argwhere(near_wall)) if len(p)]
    pool = np.concatenate(pools, axis=0) if pools else np.argwhere(on_board)
    n_edits = min(int(rng.integers(1, 5)), len(pool))  # 1-4 cells
    for i in rng.choice(len(pool), size=n_edits, replace=False):
        row, col = pool[i]
        w[row, col] = not w[row, col]
    return (w & on_board).astype(np.uint8)


# --------------------------------------------------------------------------
# Held out: the demo page's "Random loop" (src/lines.ts randomLoop), for
# evaluation only. A port of the algorithm, not of its random stream.
# --------------------------------------------------------------------------

def _cost_field(rng, R, wiggle):
    """costField: smooth random cell costs (a few random bumps), so cheapest paths through it wander."""
    S = side(R)
    nb = 6 + int(rng.random() * 6)
    b = rng.random((nb, 4))  # per bump: q, r, width, height
    bq, br = (b[:, 0] * 2 - 1) * R, (b[:, 1] * 2 - 1) * R
    bw, bh = 0.15 + b[:, 2] * 0.5, b[:, 3] * 2 - 0.6
    r = (np.arange(S) - R).reshape(S, 1, 1) - br
    q = (np.arange(S) - R).reshape(1, S, 1) - bq
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r)) / (R * bw + 1)
    v = (bh * np.exp(-dist * dist)).sum(axis=2)
    return 1 + wiggle * 12 * np.maximum(0, v + 0.3) ** 2 + rng.random((S, S)) * 0.5


def _cheapest_path(cost, a, b, allowed):
    """Dijkstra from cell a to cell b (flat indices) through allowed cells; the path as a list, or None."""
    nb = _nbrs((cost.shape[0] - 1) // 2)
    cost = cost.ravel().tolist()
    dist, prev = {a: cost[a]}, {a: -1}
    heap = [(cost[a], a)]
    while heap:
        dv, v = heapq.heappop(heap)
        if dv > dist[v]:
            continue
        if v == b:
            break
        for u in nb[v]:
            nd = dv + cost[u]
            if allowed[u] and nd < dist.get(u, math.inf):
                dist[u], prev[u] = nd, v
                heapq.heappush(heap, (nd, u))
    if b not in dist:
        return None
    path = [b]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    return path[::-1]


def _page_loop(rng, R, blocked, wiggle=1.0):
    """randomLoop: flat indices of a loop round a random centre, avoiding `blocked` and the rim; None if no room."""
    if R < 4:
        return None
    S = side(R)
    on_board, is_rim = mask(R) == 1, rim(R)
    cost = _cost_field(rng, R, wiggle)
    hexd = lambda q, r: max(abs(q), abs(r), abs(q + r))
    at = lambda q, r: (r + R) * S + q + R
    for _ in range(40):
        span = max(1, math.floor(R * 0.6))
        qc = math.floor((rng.random() * 2 - 1) * span)
        rc = math.floor((rng.random() * 2 - 1) * span)
        if hexd(qc, rc) > R - 3:
            continue
        core = math.floor(rng.random() * min(3, R / 4))
        # The cut: the row through the centre, from the centre east to the edge.
        no = (_axial_dist(R, qc, rc) <= core) | blocked | is_rim | ~on_board
        q = qc
        while hexd(q, rc) <= R:
            no[rc + R, q + R] = True
            q += 1
        sq, sr = qc + core + 1, rc  # on the cut, just outside the core
        if hexd(sq, sr) > R - 2:
            continue
        up, down = [at(sq, sr - 1), at(sq + 1, sr - 1)], [at(sq - 1, sr + 1), at(sq, sr + 1)]
        a, b = up[math.floor(rng.random() * 2)], down[math.floor(rng.random() * 2)]
        allowed = ~no.ravel()
        allowed[[c for c in up + down if c not in (a, b)]] = False  # the start's other neighbours stay free
        if blocked[sr + R, sq + R] or not allowed[a] or not allowed[b]:
            continue
        path = _cheapest_path(cost, a, b, allowed.tolist())
        if path is None or len(path) < 4:
            continue
        return [at(sq, sr)] + path
    return None


def page_loops(rng: np.random.Generator, R: int) -> np.ndarray:
    """uint8 [S,S]: 1-3 loops from the demo page's "Random loop" button, each drawn round the walls so far.

    Evaluation only (never in batch() or random_walls). Empty for R < 4,
    where the page draws none.
    """
    S = side(R)
    walls = np.zeros(S * S, dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        loop = _page_loop(rng, R, walls.reshape(S, S))
        if loop is not None:
            walls[loop] = True
    return walls.reshape(S, S).astype(np.uint8)
