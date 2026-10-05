"""Read the strand dataset (scripts/strand-export.ts) and make M1a / M1b batches.

Layout under the data dir (default data/strand): meta.json, and one .npz per (split, level, rule):
train/L{2,3,4}/rNNN.npz for the TRAIN rules, eval/L{2,3,4}/rNNN.npz for the HELD-OUT rules (fixed seed).
The arrays of one file (B boards padded to the largest, H x W; T taps; N steps; S strands):

  mask [B,H,W] u1            board cells                   chords [B,15,H,W] u1    the rule's chord bits
  tile_index/type/rot [B,H,W] Spectacle tile, leaf type (HEX_LEAF_ORDER), rotation (-1 off board)
  board_root/kind/orient/mirror/q0/r0/h/w/tiles [B]      kind 0 = full patch, 1 = crop of level 4
  chord_strand [B,15,H,W] i4 strand id of each chord (-1 none); strand_len/closed/board [S]
  tap_board/row/col/d0/d1/closed/strand [T]; tap_ptr [T+1]: tap i's strand is steps tap_ptr[i]:tap_ptr[i+1]
  step_row/col/in/out/index [N]: the strand as Spectacle's walkStrand walks it, oriented d0 -> d1, `index`
                             signed from the tap (see walker.Strand)

Batch shapes are the plan's M1a model (docs/spectacle-nca-plan.md §6): 22 input planes, 6 + 1 targets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .walker import PAIRS, PAIR_INDEX, Strand, exit_table, walk

N_INPUTS = 22  # chords 15, tap 6, mask 1
CHORD_PLANES = slice(0, 15)
TAP_PLANES = slice(15, 21)
MASK_PLANE = 21


def default_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "strand")


def load_meta(data_dir: str | None = None) -> dict:
    with open(os.path.join(data_dir or default_dir(), "meta.json")) as f:
        return json.load(f)


def file_paths(data_dir: str | None = None, split: str = "train", levels=(2, 3, 4), subsets=None) -> list[str]:
    """Paths of one split ('train' or 'eval'), some levels, optionally only rules of some subsets ('15', ...)."""
    d = data_dir or default_dir()
    meta = load_meta(d)
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    out = []
    for f in meta["files"]:
        if f["split"] == split and f["level"] in levels and (subsets is None or subset_of[f["rule"]] in subsets):
            out.append(os.path.join(d, f["path"]))
    return out


class StrandFile:
    """One (split, level, rule) file, fully in memory."""

    def __init__(self, path: str):
        self.path = path
        with np.load(path) as z:
            self.a = {k: z[k] for k in z.files}
        self.rule_id = int(self.a["rule_id"][0])
        self.level = int(self.a["level"][0])
        self._ex: dict[int, np.ndarray] = {}

    def __getitem__(self, k: str) -> np.ndarray:
        return self.a[k]

    @property
    def n_boards(self) -> int:
        return len(self.a["board_h"])

    @property
    def n_taps(self) -> int:
        return len(self.a["tap_board"])

    def hw(self, b: int) -> tuple[int, int]:
        return int(self.a["board_h"][b]), int(self.a["board_w"][b])

    def mask(self, b: int) -> np.ndarray:
        h, w = self.hw(b)
        return self.a["mask"][b, :h, :w]

    def chords(self, b: int) -> np.ndarray:
        h, w = self.hw(b)
        return self.a["chords"][b, :, :h, :w]

    def exits(self, b: int) -> np.ndarray:
        if b not in self._ex:
            self._ex[b] = exit_table(self.chords(b))
        return self._ex[b]

    def tap(self, i: int) -> Strand:
        """The exported (walkStrand) strand of tap i."""
        s, e = self.a["tap_ptr"][i], self.a["tap_ptr"][i + 1]
        g = lambda k: self.a[k][s:e].astype(np.int64)  # noqa: E731
        return Strand(g("step_row"), g("step_col"), g("step_in"), g("step_out"), g("step_index"),
                      bool(self.a["tap_closed"][i]))

    def tap_args(self, i: int) -> tuple[int, int, int, int, int]:
        """(board, row, col, d0, d1) of tap i."""
        a = self.a
        return (int(a["tap_board"][i]), int(a["tap_row"][i]), int(a["tap_col"][i]), int(a["tap_d0"][i]),
                int(a["tap_d1"][i]))

    def walk(self, b: int, row: int, col: int, d0: int, d1: int) -> Strand:
        """A strand from the chord planes alone (the Python walker)."""
        return walk(self.exits(b), self.mask(b), row, col, d0, d1)

    def random_tap(self, rng: np.random.Generator, b: int | None = None) -> tuple[int, int, int, int, int]:
        """(board, row, col, d0, d1): a uniformly random chord of a (random) board, random orientation."""
        if b is None:
            b = int(rng.integers(self.n_boards))
        p, rows, cols = np.nonzero(self.chords(b))
        k = int(rng.integers(len(p)))
        d0, d1 = PAIRS[p[k]]
        if rng.integers(2):
            d0, d1 = d1, d0
        return b, int(rows[k]), int(cols[k]), d0, d1


def edge_planes(strand: Strand, h: int, w: int) -> np.ndarray:
    """float32 [6,h,w]: 1 on edge d of every cell the strand crosses there (both ends of each chord)."""
    out = np.zeros((6, h, w), dtype=np.float32)
    out[strand.ins, strand.rows, strand.cols] = 1
    out[strand.outs, strand.rows, strand.cols] = 1
    return out


@dataclass
class Sample:
    inputs: np.ndarray  # [22,h,w]
    edges: np.ndarray  # [6,h,w]
    closed: np.ndarray  # [1,h,w]
    on: np.ndarray  # [1,h,w]
    length: int
    grow: int
    rule: int
    level: int


def m1a_sample(f: StrandFile, b: int, row: int, col: int, d0: int, d1: int, strand: Strand | None = None) -> Sample:
    """One M1a/M1b sample: a board, a rule's chords, a tap; the strand through the tap as the target."""
    h, w = f.hw(b)
    if strand is None:
        strand = f.walk(b, row, col, d0, d1)
    x = np.zeros((N_INPUTS, h, w), dtype=np.float32)
    x[CHORD_PLANES] = f.chords(b)
    x[15 + d0, row, col] = 1
    x[15 + d1, row, col] = 1
    x[MASK_PLANE] = f.mask(b)
    edges = edge_planes(strand, h, w)
    on = (edges.max(0, keepdims=True) > 0).astype(np.float32)
    return Sample(x, edges, on * float(strand.closed), on, len(strand), strand.grow_steps(), f.rule_id, f.level)


def pattern_sample(f: StrandFile, b: int):
    """M1b without a tap ("the whole pattern"): (inputs [22,h,w] with zero tap planes, closed_edges [6,h,w]).

    closed_edges[d] = 1 where the chord using edge d belongs to a circuit. Per edge, not per cell: a cell can
    carry chords of a circuit and of a tail at once (docs/strand-data.md counts how often), so one
    "closed" plane per cell is not well defined without a tap.
    """
    h, w = f.hw(b)
    x = np.zeros((N_INPUTS, h, w), dtype=np.float32)
    x[CHORD_PLANES] = f.chords(b)
    x[MASK_PLANE] = f.mask(b)
    sid = f["chord_strand"][b, :, :h, :w]
    closed = f["strand_closed"].astype(bool)
    out = np.zeros((6, h, w), dtype=np.float32)
    for p, (a, c) in enumerate(PAIRS):
        on = sid[p] >= 0
        cl = np.zeros((h, w), dtype=bool)
        cl[on] = closed[sid[p][on]]
        out[a][cl] = 1
        out[c][cl] = 1
    return x, out


def _pad(a: np.ndarray, H: int, W: int) -> np.ndarray:
    return np.pad(a, ((0, 0), (0, H - a.shape[1]), (0, W - a.shape[2])))


@dataclass
class M1aBatcher:
    """Random M1a batches from a set of files.

    batch(n) returns a dict of float32 arrays (torch tensors with as_torch=True), padded with zeros (mask 0)
    to the batch's largest board or to pad_to = (H, W):
      inputs [n,22,H,W]  planes 0-14 the rule's chord bits (PAIRS order), 15-20 the tap's two edges on the
                         tapped cell (direction d -> plane 15 + d), 21 the board mask
      edges  [n,6,H,W]   target: edge d of the cell is crossed by the tapped strand
      closed [n,1,H,W]   target: 1 on the strand's cells iff it is a circuit (M1b)
      on     [n,1,H,W]   the strand's cells (where `closed` is defined)
      length [n]         strand length in steps (chords); grow [n] two-way growth steps from the tap
      rule [n], level [n]
    fresh=True draws a new uniformly random tap (chord and orientation) and walks it in Python; False samples
    the exported taps (walkStrand's walks). Files are loaded lazily and kept.
    """

    paths: list[str]
    seed: int = 0
    fresh: bool = True
    pad_to: tuple[int, int] | None = None
    as_torch: bool = False
    _files: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def file(self, k: int) -> StrandFile:
        if k not in self._files:
            self._files[k] = StrandFile(self.paths[k])
        return self._files[k]

    def sample(self) -> Sample:
        f = self.file(int(self.rng.integers(len(self.paths))))
        if self.fresh:
            return m1a_sample(f, *f.random_tap(self.rng))
        i = int(self.rng.integers(f.n_taps))
        return m1a_sample(f, *f.tap_args(i), strand=f.tap(i))

    def batch(self, n: int) -> dict:
        ss = [self.sample() for _ in range(n)]
        H, W = self.pad_to or (max(s.inputs.shape[1] for s in ss), max(s.inputs.shape[2] for s in ss))
        out = {
            "inputs": np.stack([_pad(s.inputs, H, W) for s in ss]),
            "edges": np.stack([_pad(s.edges, H, W) for s in ss]),
            "closed": np.stack([_pad(s.closed, H, W) for s in ss]),
            "on": np.stack([_pad(s.on, H, W) for s in ss]),
            "length": np.array([s.length for s in ss]),
            "grow": np.array([s.grow for s in ss]),
            "rule": np.array([s.rule for s in ss]),
            "level": np.array([s.level for s in ss]),
        }
        if self.as_torch:
            import torch

            out = {k: torch.from_numpy(v) for k, v in out.items()}
        return out


__all__ = ["StrandFile", "M1aBatcher", "m1a_sample", "pattern_sample", "edge_planes", "load_meta", "file_paths",
           "default_dir", "N_INPUTS", "PAIR_INDEX"]
