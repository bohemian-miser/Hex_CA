"""Lattice conventions and the strand walker, from the 15 chord-bit planes alone.

Conventions (the same as nca/hexgrid.py, src/hex.ts and scripts/strand-export.ts):

- A cell is axial (q, r); a board array has row = r - r0, col = q - q0 (r0, q0 per board; hexgrid.py's
  radius-R boards are the case r0 = q0 = -R).
- Direction d in 0..5 is src/hex.ts DIRS[d] as (dq, dr): 0 (1,0), 1 (1,-1), 2 (0,-1), 3 (-1,0), 4 (-1,1),
  5 (0,1); as (drow, dcol): 0 (0,1), 1 (-1,1), 2 (-1,0), 3 (0,-1), 4 (1,-1), 5 (1,0). Opposite = (d + 3) % 6.
  These are exactly hexgrid.NEIGHBOURS (in another order), so the masked 3x3 kernel sees all six.
- Chord bit p in 0..14 joins the unordered direction pair PAIRS[p] = (0,1), (0,2), .., (4,5): the line
  enters the cell across edge a and leaves across edge b (or the other way round).

The walk: leave a cell across edge d; the next cell is the neighbour in direction d (off the board or off
the array = a tail); it is entered across edge (d + 3) % 6; the chord there that uses that edge gives the
exit (none = a tail). The strand is closed when the walk comes back to its first chord.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))  # (dq, dr)
DROW = tuple(dr for _, dr in DIRS)
DCOL = tuple(dq for dq, _ in DIRS)
PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
PAIR_INDEX = np.full((6, 6), -1, dtype=np.int64)
for _p, (_a, _b) in enumerate(PAIRS):
    PAIR_INDEX[_a, _b] = PAIR_INDEX[_b, _a] = _p


def exit_table(chords: np.ndarray) -> np.ndarray:
    """int8 [6,H,W] from uint8 [15,H,W]: the exit direction for a line entering across edge d, -1 if none.

    Raises if two chords of one cell use the same edge (then the bits would not define a walk).
    """
    _, H, W = chords.shape
    ex = np.full((6, H, W), -1, dtype=np.int8)
    for p, (a, b) in enumerate(PAIRS):
        on = chords[p] > 0
        if (on & ((ex[a] >= 0) | (ex[b] >= 0))).any():
            raise ValueError(f"two chords use one edge (pair {PAIRS[p]})")
        ex[a][on] = b
        ex[b][on] = a
    return ex


@dataclass
class Strand:
    """One strand in walking order, oriented from the tap's d0 end to its d1 end.

    rows, cols, ins, outs: per step (one chord), the cell and the entry / exit edge.
    index: signed steps from the tap (0 at the tap; > 0 ahead through d1, < 0 behind through d0;
    a loop is walked ahead only, 0..n-1). closed: the strand is a circuit (else tail to tail).
    """

    rows: np.ndarray
    cols: np.ndarray
    ins: np.ndarray
    outs: np.ndarray
    index: np.ndarray
    closed: bool

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def tap_pos(self) -> int:
        """Position of the tap in the sequence (= number of steps behind it)."""
        return int((self.index < 0).sum())

    def grow_steps(self) -> int:
        """CA steps for a line growing one chord per step both ways from the tap to cover the strand."""
        n = len(self)
        if self.closed:
            return n // 2  # two fronts: 1 + 2t chords after t steps (an even loop's last chord in t = n/2)
        return max(self.tap_pos, n - 1 - self.tap_pos)


def _one_way(ex, mask, row, col, din, dout, limit):
    """Steps from (row, col, din -> dout) onwards, and whether the walk came back to the start."""
    H, W = mask.shape
    steps = [(row, col, din, dout)]
    start = (row, col)
    r, c, d = row, col, dout
    seen = {(row, col, din), (row, col, dout)}
    while len(steps) < limit:
        nr, nc = r + DROW[d], c + DCOL[d]
        if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
            return steps, False
        ein = (d + 3) % 6
        eout = int(ex[ein, nr, nc])
        if eout < 0:
            return steps, False
        if (nr, nc) == start and ein in (din, dout):
            return steps, True
        if (nr, nc, ein) in seen:
            raise RuntimeError("walk re-entered a chord other than its first (the bits branch)")
        seen.add((nr, nc, ein))
        seen.add((nr, nc, eout))
        steps.append((nr, nc, ein, eout))
        r, c, d = nr, nc, eout
    raise RuntimeError(f"walk longer than {limit} steps")


def walk(ex: np.ndarray, mask: np.ndarray, row: int, col: int, d0: int, d1: int, limit: int = 1 << 22) -> Strand:
    """The strand through the chord (d0, d1) of cell (row, col), both ways (exit table from `exit_table`)."""
    if int(ex[d0, row, col]) != d1:
        raise ValueError(f"no chord ({d0}, {d1}) at ({row}, {col})")
    ahead, closed = _one_way(ex, mask, row, col, d0, d1, limit)
    behind = []
    if not closed:
        back, closed_back = _one_way(ex, mask, row, col, d1, d0, limit)
        if closed_back:
            raise RuntimeError("closed one way, open the other")
        behind = [(r, c, dout, din) for r, c, din, dout in reversed(back[1:])]  # forward orientation
    seq = behind + ahead
    a = np.array(seq, dtype=np.int64).reshape(-1, 4)
    index = np.arange(len(seq), dtype=np.int64) - len(behind)
    return Strand(a[:, 0], a[:, 1], a[:, 2], a[:, 3], index, closed)


def decompose(ex: np.ndarray, mask: np.ndarray):
    """Every strand of a board: (strand_id int32 [15,H,W] (-1 = no chord), lengths [S], closed [S] bool).

    Strands are numbered in array order of their lowest (row, col, pair) chord.
    """
    _, H, W = ex.shape
    sid = np.full((15, H, W), -1, dtype=np.int32)
    lengths, closed = [], []
    for row in range(H):
        for col in range(W):
            if not mask[row, col]:
                continue
            for a in range(6):
                b = int(ex[a, row, col])
                if b <= a:
                    continue
                p = PAIR_INDEX[a, b]
                if sid[p, row, col] >= 0:
                    continue
                s = walk(ex, mask, row, col, a, b)
                k = len(lengths)
                sid[PAIR_INDEX[s.ins, s.outs], s.rows, s.cols] = k
                lengths.append(len(s))
                closed.append(s.closed)
    return sid, np.array(lengths, dtype=np.int64), np.array(closed, dtype=bool)


def render_chords(tile_type: np.ndarray, tile_rot: np.ndarray, mirror: int, local_pairs) -> np.ndarray:
    """uint8 [15,H,W]: a rule's chord planes from tile type and rotation (the "purer" inputs).

    local_pairs[type] = [[k0, k1], ...] chords as local edge indices (meta.json `local_pairs`); local edge k
    faces board direction (mirror * k + tile_rot) mod 6.
    """
    H, W = tile_type.shape
    out = np.zeros((15, H, W), dtype=np.uint8)
    on = tile_type >= 0
    for t, pairs in enumerate(local_pairs):
        sel = on & (tile_type == t)
        if not sel.any():
            continue
        rot = tile_rot[sel].astype(np.int64)
        rows, cols = np.nonzero(sel)
        for k0, k1 in pairs:
            a = (mirror * k0 + rot) % 6
            b = (mirror * k1 + rot) % 6
            out[PAIR_INDEX[a, b], rows, cols] = 1
    return out
