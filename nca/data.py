"""Random wall pictures and the targets (ground truth for training).

Everything here is deterministic given the numpy Generator passed in: same
rng state in, same picture out.

targets(walls, R) gives every acceptable answer (spec v2: enclosed regions
fill, and of two or more rim regions all but the "largest" fill; spec v4: the
second measure of "largest" is the angular span of a region's rim cells);
oracle() is its primary one. Training pictures come from random_walls / batch /
edit_walls. page_loops is a port of the demo page's "Random loop" button, kept
apart for evaluation (an out-of-distribution check): nothing in training draws
from it.
"""

import math
import heapq

import numpy as np

from .hexgrid import NEIGHBOURS, consts, mask, rim, side

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
# The targets (ground truth).
# --------------------------------------------------------------------------

def _labels(open_cells: np.ndarray, R: int) -> np.ndarray:
    """int32 [S,S]: region number of each open cell (6-connected components), -1 elsewhere.

    Regions are numbered in order of their lowest flat index (row*S+col), so a
    lower number means a lower first cell.
    """
    S = side(R)
    free = open_cells.ravel().tolist()
    nb = _nbrs(R)
    lab = [-1] * (S * S)
    n = 0
    for s in range(S * S):
        if free[s] and lab[s] < 0:
            lab[s] = n
            stack = [s]
            for i in stack:  # grows as it goes: a BFS of one region
                for j in nb[i]:
                    if free[j] and lab[j] < 0:
                        lab[j] = n
                        stack.append(j)
            n += 1
    return np.array(lab, dtype=np.int32).reshape(S, S)


def _rim_depth(open_cells: np.ndarray, R: int) -> np.ndarray:
    """int16 [S,S]: BFS distance from the rim through open cells (rim = 0) in rim regions, -1 elsewhere."""
    S = side(R)
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
    return np.array(depth, dtype=np.int16).reshape(S, S)


# Spans within EPS of the largest are a near-tie (spec v4): either side is acceptable.
EPS = 0.02
_THETA = {}  # R -> float64 [2,S,S]: theta1, theta2 (the float32 consts, widened)


def _theta_extremes(lab: np.ndarray, R: int):
    """(has a rim cell bool [n], max theta1, min theta1, max theta2, min theta2 float64 [n]) per region number
    of `lab`, over the region's rim cells (0 for a region with none). float64 copies of the float32 consts,
    so differences of them are exact (the same on the TS side, which does its arithmetic in doubles)."""
    n = int(lab.max()) + 1
    on_rim = (lab >= 0) & rim(R)
    rl = lab[on_rim]
    has = np.bincount(rl, minlength=n) > 0
    out = [has]
    if R not in _THETA:
        _THETA[R] = consts(R)[1:3].astype(np.float64)
    for theta in _THETA[R]:  # theta1, theta2
        v = theta[on_rim]
        hi, lo = np.full(n, -np.inf), np.full(n, np.inf)
        np.maximum.at(hi, rl, v)
        np.minimum.at(lo, rl, v)
        out += [np.where(has, hi, 0.0), np.where(has, lo, 0.0)]
    return out


def _spans(lab: np.ndarray, R: int) -> np.ndarray:
    """float64 [n]: angular span of each region's rim cells, min(max1 - min1, max2 - min2) (0 with none).

    theta2 = theta1 + 0.5 (mod 1) catches the arcs that straddle theta1 = 0, so
    the min of the two is the angular extent of the shorter way round that
    covers the region's rim cells: the side of a bridge whose ends are closer
    along the rim has the smaller span.
    """
    _, hi1, lo1, hi2, lo2 = _theta_extremes(lab, R)
    return np.minimum(hi1 - lo1, hi2 - lo2)


def _rim_regions(lab: np.ndarray, R: int):
    """(rim region numbers, their areas, their spans), ordered best "largest" first.

    Order: area descending, then span descending, then lowest first cell
    (= region number) -- so element 0 is the primary target's unfilled region.
    """
    n = int(lab.max()) + 1
    on = lab >= 0
    area = np.bincount(lab[on], minlength=n)
    rimc = np.bincount(lab[on & rim(R)], minlength=n)
    span = _spans(lab, R) if n else np.zeros(0)
    ids = np.flatnonzero(rimc > 0)
    ids = ids[np.lexsort((ids, -span[ids], -area[ids]))]  # last key is the primary sort key
    return ids, area[ids], span[ids]


def targets(walls: np.ndarray, R: int):
    """(fills uint8 [K,S,S], depth int16 [S,S]): every acceptable answer, fills[0] the primary.

    Regions = 6-connected components of on-board non-wall cells; a rim region
    has a rim cell, the others are enclosed. Every enclosed region fills. Of
    the rim regions all fill but one, the "largest" X: the candidates for X
    are the regions of maximal area and those whose span (_spans: the angle of
    rim their rim cells cover, the shorter way round) is within EPS of the
    largest span, and each candidate gives one acceptable target (fill every
    open cell outside X). With one rim region that is the only candidate, so
    it stays empty (as in v1); with none, every open cell fills.
    The primary target leaves the max-area region empty, ties broken by max
    span, then by lowest cell index; the others follow in the same order.
    Walls and off-board cells are 0 everywhere. depth = BFS distance from the
    rim through open cells, for cells in rim regions (rim = 0); -1 elsewhere.
    """
    open_cells = (mask(R) == 1) & (walls == 0)
    lab = _labels(open_cells, R)
    depth = _rim_depth(open_cells, R)
    ids, area, span = _rim_regions(lab, R)
    if len(ids) == 0:
        return open_cells.astype(np.uint8)[None], depth
    keep = (area == area[0]) | (span >= span.max() - EPS)  # a candidate: max area or (near) max span
    fills = np.stack([open_cells & (lab != x) for x in ids[keep]]).astype(np.uint8)
    return fills, depth


def oracle(walls: np.ndarray, R: int):
    """(primary fill uint8 [S,S], depth int16 [S,S]): targets() for call sites that want one answer."""
    fills, depth = targets(walls, R)
    return fills[0], depth


N_AUX = 5  # aux planes, for state channels 2..6
_SRC = {}  # R -> float32 [4,S,S]: what a rim cell feeds into planes 0..3 (0 off the rim)


def _rim_sources(R: int) -> np.ndarray:
    """float32 [4,S,S]: theta1, 1 - theta1, theta2, 1 - theta2 on rim cells, 0 elsewhere (spec v5 §1): the
    const planes src1, src1c, src2, src2c (spec v6 §1), so the flood, the steady state and the model's
    hand-written floods (model.install_floods) all carry identical numbers."""
    if R not in _SRC:
        _SRC[R] = consts(R)[3:7]
    return _SRC[R]


def aux_span(aux: np.ndarray) -> np.ndarray:
    """float32 [..., S, S]: the span read off aux planes [..., 4+, S, S] (spec v5 §1):
    clip(min(p0 + p1 - 1, p2 + p3 - 1), 0, 1) -- max - min of theta1 (and of theta2) over the rim cells that
    have reached a cell, the smaller of the two; 0 where nothing has arrived (or the region has no rim cell)."""
    p = [aux[..., c, :, :] for c in range(4)]
    return np.clip(np.minimum(p[0] + p[1] - 1, p[2] + p[3] - 1), 0, 1)


def _hexmax(x: np.ndarray) -> np.ndarray:
    """Max over each cell's 7 hex taps (self + 6 neighbours) of x [..., S, S], 0 off the array. Every plane
    the flood keeps is >= 0 and 0 off board, so this is the max over the on-board taps."""
    S = x.shape[-1]
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)])
    out = x.copy()
    for dr, dc in NEIGHBOURS:
        np.maximum(out, p[..., 1 + dr:1 + dr + S, 1 + dc:1 + dc + S], out=out)
    return out


def aux_flood(walls: np.ndarray, R: int, T: int) -> np.ndarray:
    """float32 [T, 5, S, S] (walls [S,S]) or [T, B, 5, S, S] (walls [B,S,S]): the reference flood F_1..F_T of
    spec v5 §2, one hop per step from F_0 = 0, the targets of channels 2..6 at each step of a fresh rollout:

        planes 0..3:  F' = open * max(hexmax(F), rim source)   (sources theta1, 1 - theta1, theta2, 1 - theta2)
        plane 4:      F' = mask * max(hexmax(F), aux_span(F'[0..3]))   (ungated: walls carry it too)

    open = on board and not a wall. Planes 0..3 never cross a wall (walls hold 0 and pass nothing on); plane
    4 takes the span of the NEW planes 0..3 and floods the whole board. After enough steps it is aux_targets.
    """
    on = mask(R).astype(np.float32)
    opn = (on * (walls == 0))[..., None, :, :]  # [..., 1, S, S]
    src = _rim_sources(R) * opn
    f = np.zeros(walls.shape[:-2] + (N_AUX,) + walls.shape[-2:], dtype=np.float32)
    out = np.empty((T,) + f.shape, dtype=np.float32)
    for t in range(T):
        g = np.maximum(_hexmax(f[..., :4, :, :]), src) * opn
        f = np.concatenate([g, (np.maximum(_hexmax(f[..., 4, :, :]), aux_span(g)) * on)[..., None, :, :]], -3)
        out[t] = f
    return out


def aux_targets(walls: np.ndarray, R: int) -> np.ndarray:
    """float32 [5,S,S]: the steady state of aux_flood (spec v5 §1), straight from the regions.

    Planes 0..3 (channels 2..5): max theta1, max (1 - theta1), max theta2, max (1 - theta2) over the rim cells
    of each cell's region -- 0 on walls, off board and in regions with no rim cell. Their aux_span is the
    region's span (_spans, up to float32 rounding). Plane 4 (channel 6): the largest span of any region on
    the board, the same value on every on-board cell, walls included (0 off board, and with no rim region).
    The comparative rule read off them (arc_fill): fill iff enclosed or span < plane 4 - EPS.
    """
    on_board = mask(R) == 1
    open_cells = on_board & (walls == 0)
    lab = _labels(open_cells, R)
    out = np.zeros((N_AUX,) + lab.shape, dtype=np.float32)
    feed = open_cells & rim(R)
    if not feed.any():
        return out
    n = int(lab.max()) + 1
    for c, v in enumerate(_rim_sources(R)):
        top = np.zeros(n, dtype=np.float32)  # every source is >= 0, so 0 is "no rim cell" too
        np.maximum.at(top, lab[feed], v[feed])
        out[c][open_cells] = top[lab[open_cells]]
    out[4][on_board] = aux_span(out).max()
    return out


def arc_fill(aux: np.ndarray, walls: np.ndarray, R: int) -> np.ndarray:
    """bool [S,S]: the comparative rule read off aux planes [5,S,S] (targets, or a model's channels 2..6;
    spec v5 §2): an open cell fills iff its region is enclosed (max(plane 0, plane 2) < 0.25: a rim region
    has max theta1 or max theta2 >= 0.5, an enclosed one 0) or its aux_span < plane 4 - EPS."""
    enclosed = np.maximum(aux[0], aux[2]) < 0.25
    return (mask(R) == 1) & (walls == 0) & (enclosed | (aux_span(aux) < aux[4] - EPS))


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


def _closed(rng, R, on_board, first=None, count=None):
    """1-3 closed shapes (or `count` of them): side by side, overlapping (sharing walls), nested, or running
    over the rim."""
    walls = np.zeros_like(on_board)
    prev = None
    for i in range(count or int(rng.choice((1, 2, 3), p=(0.5, 0.3, 0.2)))):
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


# --------------------------------------------------------------------------
# Bridges: wall paths from rim to rim, which split the board into rim regions.
# --------------------------------------------------------------------------

_RING = {}


def _rim_ring(R: int) -> np.ndarray:
    """int [6R, 2]: (row, col) of the rim cells in order round the board (cached per R)."""
    if R not in _RING:
        cells = np.argwhere(rim(R))
        q, r = cells[:, 1] - R, cells[:, 0] - R
        _RING[R] = cells[np.argsort(np.arctan2(r * _SQ3 / 2, q + r / 2))]
    return _RING[R]


def _qr(cell, R):
    """Float axial (q, r) of an array cell (row, col)."""
    return np.array([cell[1] - R, cell[0] - R], dtype=float)


def _path(rng, R, on_board, a, b):
    """bool [S,S]: a thin wall path from cell a to rim cell b ((row, col) each), 6-connected.

    A straight line, or one bent in towards the centre (always when straight
    would run along the rim), a curve (a quadratic Bezier through a random point), or a cheapest
    path through random cell costs that keeps off the rim between its ends
    (smooth or wandering; the cost field and Dijkstra are the ones the
    held-out page_loops port uses, but what they draw here is an open path
    to the rim, never a loop).
    """
    S = side(R)
    pa, pb = _qr(a, R), _qr(b, R)
    u = rng.random()
    if u < 0.35:
        line = _polyline(R, np.stack([pa, pb]), closed=False)
        if (line & on_board & ~rim(R)).any() and rng.random() < 0.6:
            return line
        mid = (pa + pb) / 2 * rng.uniform(0.1, 0.9)  # the midpoint pulled in towards the centre
        return _polyline(R, np.stack([pa, mid, pb]), closed=False)
    if u < 0.6:
        c = np.array(_offset(rng, max(0, R - 1)), dtype=float)
        s = np.linspace(0, 1, 12)[:, None]
        return _polyline(R, (1 - s) ** 2 * pa + 2 * s * (1 - s) * c + s ** 2 * pb, closed=False)
    cost = _cost_field(rng, R, rng.choice((0.0, 0.3, 1.0)))
    ia, ib = int(a[0] * S + a[1]), int(b[0] * S + b[1])
    allowed = (on_board & ~rim(R)).ravel()
    allowed[[ia, ib]] = True
    out = np.zeros(S * S, dtype=bool)
    p = _cheapest_path(cost, ia, ib, allowed.tolist())
    if p is not None:
        out[p] = True
    return out.reshape(S, S)


def _thicken(rng, walls, on_board):
    """Mostly as is; else 2 wide (the path shifted one cell) or 3 wide (dilated), whole or near one spot."""
    u = rng.random()
    if u < 0.6:
        return walls
    if u < 0.85:
        dr, dc = _ROUND[int(rng.integers(6))]
        more = _look(walls, dr, dc)
    else:
        more = _dilate(walls)
    if rng.random() < 0.4:  # thick only near one spot
        R = (walls.shape[0] - 1) // 2
        cq, cr = _offset(rng, R)
        more &= _axial_dist(R, cq, cr) <= rng.uniform(1, R + 1)
    return (walls | more) & on_board


def _bridge(rng, R, on_board, from_centre=False):
    """One bridge: a wall path from a rim cell to another a random way round the rim.

    The way round sets how lopsided the two sides are: a few cells round a
    corner cuts off a corner (as little as the corner cell), half way round
    splits the board near evenly. from_centre: just a path from the centre to
    the rim (the caller mirrors it through the centre).
    """
    ring = _rim_ring(R)
    n = len(ring)
    if from_centre:
        return _thicken(rng, _path(rng, R, on_board, (R, R), ring[rng.integers(n)]), on_board)
    if rng.random() < 0.2:  # round a corner, 1 to R/2+1 cells each side of it
        q, r = ring[:, 1] - R, ring[:, 0] - R
        corners = np.flatnonzero((np.abs(q) == R).astype(int) + (np.abs(r) == R) + (np.abs(q + r) == R) >= 2)
        c = int(rng.choice(corners))
        i, j = c - int(rng.integers(1, R // 2 + 2)), c + int(rng.integers(1, R // 2 + 2))
    else:
        i = int(rng.integers(n))
        j = i + 2 + int((n // 2 - 2) * math.sqrt(rng.random()))  # sqrt: the cut-off area grows ~ arc^2
    return _thicken(rng, _path(rng, R, on_board, ring[i % n], ring[j % n]), on_board)


def _walls_bridge(rng, R, on_board):
    """1-3 bridges; sometimes leaky (1-2 of their cells knocked out), sometimes with loops or clutter.

    One board in six is point-symmetric: its bridges run from the centre to
    the rim and everything is mirrored through the centre, (q, r) -> (-q, -r),
    so the two sides of a bridge tie exactly.
    """
    walls = np.zeros_like(on_board)
    if R < 1:
        return walls
    sym = rng.random() < 1 / 6
    for _ in range(int(rng.choice((1, 2, 3), p=(0.6, 0.28, 0.12)))):
        walls |= _bridge(rng, R, on_board, from_centre=sym)
    if rng.random() < 0.2:
        _knock_out(rng, walls, int(rng.integers(1, 3)))
    if rng.random() < 0.3:
        walls |= _closed(rng, R, on_board)
    if rng.random() < 0.2:
        walls |= _strokes(rng, R, on_board, int(rng.integers(1, 3)))
    if rng.random() < 0.1:
        walls |= rng.random(walls.shape) < rng.uniform(0.01, 0.05)
    return walls | walls[::-1, ::-1] if sym else walls


_KINDS = {  # each (rng, R, on_board) -> bool [S,S] walls
    "blob": lambda rng, R, on_board: _closed(rng, R, on_board, "blob"),
    "poly": lambda rng, R, on_board: _closed(rng, R, on_board, "poly"),
    "ring": lambda rng, R, on_board: _closed(rng, R, on_board, "ring"),
    "leaky": _walls_leaky,
    "messy": _walls_messy,
    "open": _walls_open,
    "noise": _walls_noise,
    "bridge": _walls_bridge,
}

# Weights for kind=None, tuned (see nca/selftest.py) so 25-30 % of boards have
# two or more rim regions and about 0.6 have at least one filled cell.
# "blob"/"poly"/"ring" name the first shape of 1-3; the others are drawn from
# all three.
_MIX = {"blob": 0.13, "poly": 0.12, "ring": 0.04, "leaky": 0.16, "messy": 0.13, "open": 0.14, "noise": 0.13,
        "bridge": 0.15}

# Share of boards of the other kinds drawn with the rim ring left clear of walls. Every
# kind but "bridge" walls rim cells now and then (shapes over the rim, strokes running
# off the board, noise), and two walled spots on the rim are a bridge under the new
# target -- too many of them without this. With the rim ring clear there is exactly one
# rim region: a shape that crossed the rim is then open to the board edge.
P_OFF_RIM = 0.5


def random_walls(rng: np.random.Generator, R: int, kind: str = None) -> np.ndarray:
    """uint8 [S,S] wall picture (0/1), zero off board. Deterministic given rng."""
    on_board = mask(R) == 1
    if kind is None:
        kind = rng.choice(list(_MIX.keys()), p=list(_MIX.values()))
    walls = _KINDS[kind](rng, R, on_board)
    if kind != "bridge" and rng.random() < P_OFF_RIM and (walls & ~rim(R)).any():  # (not if that leaves nothing)
        walls &= ~rim(R)
    return (walls & on_board).astype(np.uint8)


def pad_targets(fills, K: int = None) -> np.ndarray:
    """uint8 [n,K,S,S] from n targets() fills ([k_i,S,S] each), each padded to K rows by repeating its primary.

    K defaults to the largest k_i. A repeated primary changes neither "equals
    any target" nor a min over targets, so padded rows need no masking. A
    board with more than K targets keeps its first K (primary first): fewer
    choices, never a wrong one -- for a fixed-size pool.
    """
    K = K or max(len(f) for f in fills)
    return np.stack([np.concatenate([f[:K], np.repeat(f[:1], max(0, K - len(f)), axis=0)]) for f in fills])


def batch(rng: np.random.Generator, R: int, n: int):
    """(walls [n,S,S] uint8, fills [n,K,S,S] uint8, depth [n,S,S] int16).

    fills[i] = every acceptable target of board i (targets()), fills[i, 0] its
    primary; K is the batch's largest number of targets, the boards with fewer
    padded by repeating their primary (pad_targets).
    """
    walls = np.stack([random_walls(rng, R) for _ in range(n)])
    fills, depth = zip(*(targets(w, R) for w in walls))
    return walls, pad_targets(fills), np.stack(depth)


# --------------------------------------------------------------------------
# Edits: what a user drawing on a settled board does.
# --------------------------------------------------------------------------

def _splitters(w: np.ndarray, R: int) -> np.ndarray:
    """bool [S,S]: open cells that would split their region if walled.

    That is when two separate runs of walls round the cell are already joined:
    through other walls, or through the board edge (everything off the board
    counts as one wall, joined to every wall on the rim). Walling the cell
    then closes a ring of walls -- a loop, or a bridge with the edge -- round
    part of its region. On a hex grid this is exact.
    """
    on_board = mask(R) == 1
    lab = _labels(w & on_board, R)  # wall components
    edge = int(lab.max()) + 1
    lab = np.where(np.isin(lab, lab[w & rim(R)]), edge, lab)  # those on the rim join the edge
    lab[~on_board] = edge
    ring = [_look(lab, dr, dc, fill=edge) for dr, dc in _ROUND]
    starts = [(ring[k] >= 0) & (ring[k - 1] < 0) for k in range(6)]  # a run of walls begins at k
    out = np.zeros_like(on_board)
    for k1 in range(6):
        for k2 in range(k1 + 1, 6):
            out |= starts[k1] & starts[k2] & (ring[k1] == ring[k2])
    return out & on_board & ~w


def _drain(rng, w, R, filled, keep):
    """Knock out 1-2 wall cells that each sit between a filled cell and the unfilled region:
    a loop drains, or a bridge breaks (with no rim region: walls on the rim next to a filled cell)."""
    gaps = np.argwhere(w & _dilate(filled) & (_dilate(keep) if keep.any() else rim(R)))
    if not len(gaps):
        return None
    w = w.copy()
    for i in rng.choice(len(gaps), size=min(len(gaps), 1 + int(rng.random() < 0.3)), replace=False):
        w[tuple(gaps[i])] = False
    return w


def _seal(rng, w, R, filled, keep, tries=4):
    """Wall a cell that splits the unfilled region -- the last gap of a loop or of a bridge: part of it fills.

    Of a few such cells, the one that changes the most of the answer (a real
    gap rather than a nook by the rim).
    """
    cand = np.argwhere(_splitters(w, R) & keep)
    best, most = None, 0
    for i in rng.permutation(len(cand))[:tries]:
        w2 = w.copy()
        w2[tuple(cand[i])] = True
        n = int((oracle(w2, R)[0] != filled).sum())
        if n > most:
            best, most = w2, n
    return best


def _shift(rng, w, R, filled, keep, layers=3):
    """Move the bridge between the unfilled region and the biggest filled rim region into the
    unfilled one, a layer at a time (up to `layers`), until that side is the smaller: the sides swap.

    The bridge = the wall components touching both. A layer: wall the cells
    of the unfilled side next to it, open its cells next to the other side.
    None if there is no such bridge or the sides don't swap.
    """
    on_board = mask(R) == 1
    w = w.copy()
    for _ in range(layers):
        lab = _labels(on_board & ~w, R)
        ids = _rim_regions(lab, R)[0]
        if len(ids) < 2:
            return None
        x, y = lab == ids[0], lab == ids[1]
        wl = _labels(w & on_board, R)
        bridge = w & np.isin(wl, np.intersect1d(wl[w & _dilate(x)], wl[w & _dilate(y)]))
        if not bridge.any():
            return None
        w |= x & _dilate(bridge)
        w &= ~(bridge & _dilate(y))
        if not oracle(w, R)[0][y].any():  # y is the unfilled side now
            return w
    return None


def _pocket(rng, w, R, filled, keep):
    """Wall in a cell of the unfilled region that is already walled on 2+ sides (adds its 1-4 free neighbours): it fills."""
    n_free = sum(_look(~w, dr, dc) for dr, dc in NEIGHBOURS)
    n_free = np.where(keep & ~rim(R), n_free, 9)  # a rim cell stays outside whatever is round it
    if n_free.min() > 4:
        return None
    cells = np.argwhere(n_free == n_free.min())  # the most walled-in first
    row, col = cells[rng.integers(len(cells))]
    w = w.copy()
    for dr, dc in NEIGHBOURS:
        w[row + dr, col + dc] = True
    return w


# Share of edits that open, seal or shift something (the rest are random toggles):
# tuned so ~2/3 of all edits change the primary target (nca/selftest.py prints it).
P_MOVE = 0.4


def edit_walls(rng: np.random.Generator, walls: np.ndarray, R: int) -> np.ndarray:
    """A small edit of a wall picture, mostly one that changes the answer (the primary target).

    With probability P_MOVE, in random order, whichever applies first of:
    knock out a wall cell between a filled region and the unfilled one (a
    loop drains or a bridge breaks); wall a cell that splits the unfilled
    region (closes a loop or plugs a bridge's last gap); shift a bridge into
    the bigger side until the other side is the bigger -- else wall in a
    half-enclosed pocket. Otherwise, or if none applies (an empty board), it
    toggles 1-4 cells among the walls and the cells next to them (often
    harmless). Always changes at least one on-board cell.
    """
    on_board = mask(R) == 1
    w = walls.astype(bool) & on_board
    filled = oracle(w, R)[0].astype(bool)
    keep = on_board & ~w & ~filled  # the rim region the primary target leaves unfilled (none: empty)

    if rng.random() < P_MOVE:
        moves = [(_drain, _seal, _shift)[i] for i in rng.permutation(3)]
        for move in moves + [_pocket]:
            out = move(rng, w, R, filled, keep)
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
# Damage: what the training pool does to a board between visits (train.py --damage), after which its
# targets are recomputed. Edits pile up on the pool board; nothing toggles back.
# --------------------------------------------------------------------------

def disc(R: int, row: int, col: int, rad: float) -> np.ndarray:
    """bool [S,S]: the array cells within hex distance rad of cell (row, col) (off-board ones included)."""
    return _axial_dist(R, col - R, row - R) <= rad


def _burst(rng, w, R, on_board):
    """n ~ U[3, 20] random cell changes, each an addition (a random open cell walled) or a deletion (a random
    wall opened), 50/50 -- so the wall density doesn't drift on average."""
    w = w.copy()
    for _ in range(int(rng.integers(3, 21))):
        pick = on_board & (w != (rng.random() < 0.5))  # True: the open cells (an addition), False: the walls
        if not pick.any():  # an empty board can't lose a wall, a full one can't gain one
            pick = on_board
        row, col = np.argwhere(pick)[rng.integers(int(pick.sum()))]
        w[row, col] = not w[row, col]
    return w


def _erase(rng, w, R, on_board):
    """Every wall within hex distance 1-3 of a random wall cell opened (nothing to erase on an empty board)."""
    cells = np.argwhere(w if w.any() else on_board)
    row, col = cells[rng.integers(len(cells))]
    return w & ~disc(R, row, col, int(rng.integers(1, 4)))


def _stamp(rng, w, R, on_board, p_bridge=0.5):
    """w OR a freshly drawn circuit: a rim-to-rim bridge (_bridge) with probability p_bridge, else one closed
    loop (_closed: a blob outline, a polygon or a ring)."""
    new = _bridge(rng, R, on_board) if rng.random() < p_bridge else _closed(rng, R, on_board, count=1)
    return w | new


WALL_DAMAGE = ("edit", "burst", "erase", "stamp")


def damage_walls(rng: np.random.Generator, walls: np.ndarray, R: int, kind: str, p_bridge: float = 0.5) -> np.ndarray:
    """uint8 [S,S]: walls after one damage of `kind` (train.py --damage kinds a-d):
      edit   a small wall edit (edit_walls: mostly one that opens, seals or shifts something)
      burst  n ~ U[3, 20] random additions and deletions of wall cells, mixed (_burst)
      erase  every wall inside a random disc of radius 1-3 opened (_erase)
      stamp  a fresh closed loop or (with probability p_bridge) a rim-to-rim bridge ORed in (_stamp)
    Stays on the board. Deterministic given rng."""
    on_board = mask(R) == 1
    w = walls.astype(bool) & on_board
    if kind == "edit":
        return edit_walls(rng, w.astype(np.uint8), R)
    if kind == "stamp":
        out = _stamp(rng, w, R, on_board, p_bridge)
    else:
        out = {"burst": _burst, "erase": _erase}[kind](rng, w, R, on_board)
    return (out & on_board).astype(np.uint8)


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
