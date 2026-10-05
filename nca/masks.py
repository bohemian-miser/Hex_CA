"""Ragged board masks: the board shapes other than the hexagon (review item 1b; docs/nca-review.md 2.4).

Spectacle's hex field is an exact axial lattice but a ragged blob, and a model that has only seen the
hexagon's convex rim fails on it. Every mask here is a uint8 [S,S] array (S = 2R+1, the layout of
nca/hexgrid.py) inside the hexagon of radius R: connected, without holes, at least MIN_SHARE of the hexagon's
cells. Its rim is data.edge(mask): the on-board cells with an off-board neighbour. Two sources:

  blob    a union of 2-5 hex discs, minus 1-4 notches (discs centred on its rim: bays, concave corners);
          some are two lobes joined by a 1- or 2-cell isthmus
  field   a crop of Spectacle's own hex field (nca/fields/hex-l3.json, hex-l4.json: the tile centres of
          buildField({family: 'hex', level, rootTile: 'Delta'}) as axial (q, r)): the cells within hex
          distance R of a random field cell near the field's outline, turned by a random multiple of 60
          degrees and maybe mirrored (both lattice symmetries)

ragged_mask(rng, R) draws one (blob half the time, a level-4 crop 0.35, a level-3 crop 0.15). Deterministic
given rng. Numpy only (the training data producer imports it).
"""

import json
import os

import numpy as np

from .data import _axial_dist, _labels, _look, _offset, _polyline, edge
from .hexgrid import mask, rim, side

FIELDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fields")
MIN_SHARE = 0.3      # a mask keeps at least this share of the hexagon's cells
SOURCES = {"blob": 0.5, "l4": 0.35, "l3": 0.15}
_FIELD = {}          # level -> (qr int [n,2], depth int [n]: hex steps to the nearest cell off the field)


def tidy(on: np.ndarray, R: int) -> np.ndarray:
    """bool [S,S]: `on` inside the hexagon, its largest 6-connected component, holes filled (an off-board
    component that doesn't reach the hexagon's rim is enclosed: it joins the board)."""
    disc = mask(R) == 1
    on = on.astype(bool) & disc
    if not on.any():
        return on
    lab = _labels(on, R)
    on = lab == np.bincount(lab[lab >= 0]).argmax()
    off = _labels(disc & ~on, R)
    outside = np.unique(off[(off >= 0) & rim(R)])
    return on | ((off >= 0) & ~np.isin(off, outside))


def blob_mask(rng: np.random.Generator, R: int) -> np.ndarray:
    """uint8 [S,S]: a random blob (module docstring)."""
    disc = mask(R) == 1
    n_disc = int(disc.sum())
    on = disc
    for _ in range(8):
        on = np.zeros_like(disc)
        if rng.random() < 0.3 and R >= 4:  # two lobes and an isthmus between them
            ra, rb = rng.uniform(R / 4, R / 2.2, 2)
            ang = rng.uniform(0, 2 * np.pi)
            far = lambda rad, a: np.array([np.cos(a), np.sin(a)]) * (R - rad - 0.5)
            pa, pb = far(ra, ang), far(rb, ang + np.pi + rng.uniform(-0.5, 0.5))
            to_qr = lambda p: np.array([p[0] - p[1] / np.sqrt(3), 2 * p[1] / np.sqrt(3)])  # x, y -> axial
            qa, qb = to_qr(pa), to_qr(pb)
            on |= _axial_dist(R, *qa) <= ra
            on |= _axial_dist(R, *qb) <= rb
            neck = _polyline(R, np.stack([qa, qb]), closed=False)
            if rng.random() < 0.4:  # 2 cells wide
                neck |= _look(neck, 0, 1)
            on |= neck
        else:
            for _ in range(int(rng.integers(2, 6))):
                cq, cr = _offset(rng, max(1, R // 2))
                on |= _axial_dist(R, cq, cr) <= rng.uniform(R / 3, 0.9 * R)
        on &= disc
        for _ in range(int(rng.integers(1, 5))):  # notches: bays cut in from the rim
            cells = np.argwhere(edge(on))
            if not len(cells):
                break
            row, col = cells[rng.integers(len(cells))]
            on &= ~(_axial_dist(R, col - R, row - R) <= rng.uniform(1, max(1.5, R / 3)))
        on = tidy(on, R)
        if on.sum() >= MIN_SHARE * n_disc:
            break
    if on.sum() < MIN_SHARE * n_disc:  # (rare) the hexagon with one notch
        on = disc & ~(_axial_dist(R, R, 0) <= R / 2)
    return on.astype(np.uint8)


def field(level: int):
    """(qr int [n,2], depth int [n]) of Spectacle's level-`level` hex field (cached): axial (q, r) of each tile
    and its hex distance to the nearest axial cell that is not a tile (1 on the field's outline)."""
    if level not in _FIELD:
        with open(os.path.join(FIELDS, f"hex-l{level}.json")) as f:
            qr = np.array(json.load(f)["qr"], dtype=np.int64)
        lo = qr.min(0) - 1
        sz = qr.max(0) - lo + 2
        grid = np.zeros((sz[1], sz[0]), dtype=bool)  # [r, q]
        grid[qr[:, 1] - lo[1], qr[:, 0] - lo[0]] = True
        front = list(zip(*np.nonzero(~grid)))
        d = np.where(grid, -1, 0)
        i = 0
        while i < len(front):  # BFS from every off-field cell over the field
            r, q = front[i]
            i += 1
            for dr, dq in ((0, 1), (0, -1), (1, 0), (-1, 0), (-1, 1), (1, -1)):
                rr, qq = r + dr, q + dq
                if 0 <= rr < grid.shape[0] and 0 <= qq < grid.shape[1] and d[rr, qq] < 0:
                    d[rr, qq] = d[r, q] + 1
                    front.append((rr, qq))
        _FIELD[level] = (qr, d[qr[:, 1] - lo[1], qr[:, 0] - lo[0]])
    return _FIELD[level]


def _turn(qr: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    """Axial (q, r) turned k times by 60 degrees ((q, r) -> (-r, q + r)), first mirrored ((q, r) -> (r, q))."""
    q, r = (qr[:, 1], qr[:, 0]) if mirror else (qr[:, 0], qr[:, 1])
    for _ in range(k % 6):
        q, r = -r, q + r
    return np.stack([q, r], axis=1)


def field_mask(rng: np.random.Generator, R: int, level: int = 4) -> np.ndarray:
    """uint8 [S,S]: a crop of Spectacle's level-`level` hex field (module docstring). The centre is a field
    cell within max(1, 0.6 R) of the outline, so the crop always has some of the field's real rim."""
    S = side(R)
    disc = mask(R) == 1
    qr, depth = field(level)
    near = np.flatnonzero(depth <= max(1, int(0.6 * R)))
    on = disc
    for _ in range(8):
        c = qr[near[rng.integers(len(near))]]
        d = _turn(qr - c, int(rng.integers(6)), rng.random() < 0.5)
        keep = np.maximum(np.maximum(np.abs(d[:, 0]), np.abs(d[:, 1])), np.abs(d[:, 0] + d[:, 1])) <= R
        on = np.zeros((S, S), dtype=bool)
        on[d[keep, 1] + R, d[keep, 0] + R] = True
        on = tidy(on, R)
        if on.sum() >= MIN_SHARE * disc.sum():
            break
    return on.astype(np.uint8)


def ragged_mask(rng: np.random.Generator, R: int) -> np.ndarray:
    """uint8 [S,S]: a ragged board at radius R from SOURCES (a blob, or a crop of the level-4 or level-3 field)."""
    src = rng.choice(list(SOURCES), p=list(SOURCES.values()))
    return blob_mask(rng, R) if src == "blob" else field_mask(rng, R, int(src[1]))
