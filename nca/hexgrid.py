"""Hex board geometry: axial coordinates (q, r) stored in a square array.

Board = hexagon of radius R in axial coords: on board iff
max(|q|, |r|, |q+r|) <= R. Stored in a square array S x S, S = 2R+1, cell
(q, r) at [row = r+R][col = q+R]. Off-board cells exist in the array but are
always dead (every state channel is 0 there).

This file (and nca/data.py) must agree with the TS side, src/nca.ts: same
array layout, same six neighbour offsets, same kernel mask, same constant
inputs (consts).
"""

import numpy as np

# The six neighbours as (drow, dcol), exactly as the spec lists them. In
# (dq, dr) terms these are the standard axial directions; keeping them in
# (drow, dcol) form here since that's what indexes the array and the conv
# kernel.
NEIGHBOURS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (-1, 1),
    (1, -1),
)

# The 3x3 kernel mask for conv3x3_hexmasked: a hex neighbourhood is a 3x3
# box with the two corners that aren't hex neighbours, (-1,-1) and (+1,+1),
# forced to zero. k = (drow+1)*3 + (dcol+1); k=0 and k=8 are those corners.
KERNEL_MASK = np.ones((3, 3), dtype=np.float32)
KERNEL_MASK[0, 0] = 0.0  # k=0, (drow,dcol) = (-1,-1)
KERNEL_MASK[2, 2] = 0.0  # k=8, (drow,dcol) = (+1,+1)


def side(R: int) -> int:
    """Side of the square backing array for a board of radius R."""
    return 2 * R + 1


def _qr_grid(R: int):
    """q[row,col], r[row,col] for the SxS array of a board of radius R."""
    S = side(R)
    r = (np.arange(S) - R).reshape(S, 1)  # row -> r, broadcasts over columns
    q = (np.arange(S) - R).reshape(1, S)  # col -> q, broadcasts over rows
    return q, r


def mask(R: int) -> np.ndarray:
    """uint8 [S,S]: 1 where (q,r) is on the board, 0 off board."""
    q, r = _qr_grid(R)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))
    return (dist <= R).astype(np.uint8)


def rim(R: int) -> np.ndarray:
    """bool [S,S]: True on the outermost ring of the board (dist == R)."""
    q, r = _qr_grid(R)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))
    return dist == R


# The model's constant input planes, in the order they follow the state channels
# (spec v3: mask, theta1, theta2; spec v6 adds the rim sources of the four hand-written
# floods). A model takes a prefix: a v1 model only the mask, a v3-v5 one the first 3.
CONST_NAMES = ("mask", "theta1", "theta2", "src1", "src1c", "src2", "src2c")


def consts(R: int) -> np.ndarray:
    """float32 [7,S,S] = (mask, theta1, theta2, src1, src1c, src2, src2c), spec v3 + v6 §1.

    Cell centre of axial (q, r), pointy-top: x = sqrt(3) * (q + r/2), y = 1.5 * r.
    theta1 = ((atan2(y, x) / (2 pi)) + 1) mod 1, in [0, 1) (the centre cell: atan2(0, 0) = 0, so 0);
    theta2 = (theta1 + 0.5) mod 1 (so 0.5 at the centre). With rim = 1 on the cells at hex distance R:
    src1 = rim * theta1, src1c = rim * (1 - theta1), src2 = rim * theta2, src2c = rim * (1 - theta2).
    Double precision, stored as float32; off-board cells are 0 in every plane.
    """
    q, r = _qr_grid(R)
    q, r = np.broadcast_arrays(q.astype(np.float64), r.astype(np.float64))
    x = np.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    t1 = np.mod(np.arctan2(y, x) / (2.0 * np.pi) + 1.0, 1.0)
    t2 = np.mod(t1 + 0.5, 1.0)
    m = mask(R).astype(np.float64)
    e = rim(R).astype(np.float64)
    return np.stack([m, t1 * m, t2 * m, e * t1, e * (1.0 - t1), e * t2, e * (1.0 - t2)]).astype(np.float32)
