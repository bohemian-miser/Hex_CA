"""Hex board geometry: axial coordinates (q, r) stored in a square array.

Board = hexagon of radius R in axial coords: on board iff
max(|q|, |r|, |q+r|) <= R. Stored in a square array S x S, S = 2R+1, cell
(q, r) at [row = r+R][col = q+R]. Off-board cells exist in the array but are
always dead (every state channel is 0 there).

This file (and nca/data.py) must agree with the TS side, src/nca.ts: same
array layout, same six neighbour offsets, same kernel mask.
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
