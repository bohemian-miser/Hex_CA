"""The whole hex kernel for the rule-at-the-tap CA (docs/spectacle-nca-options.md §2, §4 T1, §5.2).

Data: data/strand-v2 from `npx tsx scripts/strand-export.ts --rule-table` (rules-hex.json, boards.npz,
parity.npz; docs/strand-data.md "v2").

    python -m nca.strand.rules --parity        # rendering + walks against Spectacle's walkStrand (parity.npz)
    python -m nca.strand.rules --split         # the v2 split against the exporter's check values
    python -m nca.strand.rules --bench         # fresh training samples per second, per level

A RULE is (s, digits): s indexes the 7 non-empty kernel subsets (`15`, `128`, `258`, `01346`, `03456`,
`023468`, `01234568`, class 0 included), digits[t] is the position of leaf type t's matching among its
non-crossing matchings in Spectacle's order (0..4; 0 for a type with one choice or none). Its index in the
subset is the digits in mixed radix, Delta most significant; 1,953,569 rules in all. Spectacle's matching
index for type t is options[s][t][digits[t]], and its ruleKey is `hex|<subset>|<indices joined by .>`.

SPLIT (v2): per subset, the round(0.2 n) rules with the smallest (fmix32(FNV-1a('strand-split-v2|' + key)),
index) are held out -- stratified, so every subset has held-out rules (1 / 6 / 13 / 2 / 3 / 64 / 390,625).
Training also never draws the 20 "legacy" held-out rules of the v1 split (the 100-rule set without class 0),
so launch 6's legacy numbers stay comparable with the overnight runs.

THE CODE (T1, 53 bits, the rule as the game states it): 8 bits "class m carries a line" for m in MAJORS =
(0, 1, 2, 3, 4, 5, 6, 8), then per leaf type a one-hot of its digit (9 x 5).

STATIC PLANES per cell (the input options of §3), all from the cell's (type, rotation, mirror sign); rot is the
board direction local edge 0 faces, theta = 2 pi rot / 6, mirror = 1 iff the sign is -1:
  A          16: type one-hot 9, rotation one-hot 6, mirror
  C          55: per direction d (6) the one-hot class (8) of the tile's edge facing d, plane d*8 + j; then the
                 anchor (6, one-hot = A's rotation); then mirror
  D          11: type one-hot 9, rot / 6 (one number in [0, 1)), mirror
  D-cs       12: type one-hot 9, cos theta, sin theta, mirror
  D-fourier  15: type one-hot 9, cos k theta and sin k theta for k = 1, 2, 3 without sin 3 theta (0 at every
                 60-degree step), mirror: an invertible linear map of A's rotation one-hot (probe only)
  E           9: type one-hot only (the local-frame design: rotation and mirror are the cell's frame, not inputs)
A cell's local edge k faces board direction (mirror * k + rot) mod 6 (strand-data.md), so the edge facing
d is k = mirror * (d - rot) mod 6. Off-board cells get zeros. `geo` = type * 12 + rot * 2 + (mirror < 0), -1
off the board, indexes every per-cell table here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .walker import PAIR_INDEX, PAIRS, Strand, walk

MAJORS = (0, 1, 2, 3, 4, 5, 6, 8)
N_TYPES, N_DIGITS = 9, 5
CODE_BITS = len(MAJORS) + N_TYPES * N_DIGITS  # 53
N_GEO = N_TYPES * 6 * 2                       # (type, rot, mirror)
STATIC = {"a": 16, "c": 55, "d": 11, "d-cs": 12, "d-fourier": 15, "e": 9}
SPLIT_SALT = "strand-split-v2"
SPLIT_SALT_V1 = "strand-split-v1"
HELDOUT_SHARE = 0.2
GROUPS = ("L2", "L3", "L4", "L4eval", "L4full")


def default_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                        "strand-v2")


# ---------------------------------------------------------------- hashing (as scripts/strand-export.ts)

def _fmix(h):
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13))
    h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def _fnv_bytes(h, data: bytes):
    for c in data:
        h = h ^ np.uint32(c)
        h = h * np.uint32(0x01000193)
    return h


def split_hash(s: str) -> int:
    """fmix32(FNV-1a(s)) of an ASCII string (the exporter's splitHash)."""
    with np.errstate(over="ignore"):
        return int(_fmix(_fnv_bytes(np.uint32(0x811C9DC5), s.encode())))


# ---------------------------------------------------------------- the table

class RuleTable:
    """rules-hex.json, with the per-cell lookup tables and the split built once."""

    def __init__(self, data_dir: str | None = None):
        self.dir = data_dir or default_dir()
        with open(os.path.join(self.dir, "rules-hex.json")) as f:
            self.meta = meta = json.load(f)
        self.leaf_order = meta["conventions"]["leaf_order"]
        assert tuple(meta["conventions"]["majors"]) == MAJORS and len(self.leaf_order) == N_TYPES
        self.type_majors = np.array(meta["type_majors"], np.int64)  # [9, 6]
        subs = meta["subsets"]
        self.n_sub = len(subs)
        self.keys = [s["key"] for s in subs]
        self.edges = [tuple(s["edges"]) for s in subs]
        self.options = [s["options"] for s in subs]  # [s][t] -> Spectacle matching indices
        self.local_pairs = [s["local_pairs"] for s in subs]  # [s][t][digit] -> [[k0, k1], ...]
        self.n_opt = np.array([[len(o) for o in s["options"]] for s in subs], np.int64)  # [7, 9]
        self.count = np.array([s["count"] for s in subs], np.int64)
        assert (self.n_opt.prod(1) == self.count).all() and self.n_opt.max() <= N_DIGITS
        self.radix = np.array([[int(np.prod(r[t + 1:])) for t in range(N_TYPES)] for r in self.n_opt], np.int64)
        self._build_luts()
        self._held = None
        self._legacy = None

    # -- per-cell tables
    def _build_luts(self):
        """chords[s, t, digit, rot, mbit] (15-bit masks, board frame), exits [..., 6] (-1 none), local[s, t, digit]
        (15-bit, local frame); the A / C static planes per geo."""
        S = self.n_sub
        self.lut_bits = np.zeros((S, N_TYPES, N_DIGITS, 6, 2), np.int16)
        self.lut_exit = np.full((S, N_TYPES, N_DIGITS, 6, 2, 6), -1, np.int8)
        self.lut_local = np.zeros((S, N_TYPES, N_DIGITS), np.int16)
        for s in range(S):
            for t in range(N_TYPES):
                for dg, pairs in enumerate(self.local_pairs[s][t]):
                    for k0, k1 in pairs:
                        self.lut_local[s, t, dg] |= 1 << int(PAIR_INDEX[k0, k1])
                    for rot in range(6):
                        for mb, m in enumerate((1, -1)):
                            for k0, k1 in pairs:
                                a, b = (m * k0 + rot) % 6, (m * k1 + rot) % 6
                                assert self.lut_exit[s, t, dg, rot, mb, a] < 0 and self.lut_exit[s, t, dg, rot, mb, b] < 0
                                self.lut_exit[s, t, dg, rot, mb, a] = b
                                self.lut_exit[s, t, dg, rot, mb, b] = a
                                self.lut_bits[s, t, dg, rot, mb] |= 1 << int(PAIR_INDEX[a, b])
        st = {k: np.zeros((N_GEO, n), np.float32) for k, n in STATIC.items()}
        for t in range(N_TYPES):
            for rot in range(6):
                th = 2 * np.pi * rot / 6
                for mb, m in enumerate((1, -1)):
                    g = t * 12 + rot * 2 + mb
                    for k in st:
                        st[k][g, t] = k != "c"  # the type one-hot (C has none: the classes stand for it)
                    st["a"][g, 9 + rot] = 1
                    st["a"][g, 15] = mb
                    for d in range(6):
                        k = (m * (d - rot)) % 6
                        st["c"][g, d * 8 + MAJORS.index(int(self.type_majors[t, k]))] = 1
                    st["c"][g, 48 + rot] = 1
                    st["c"][g, 54] = mb
                    st["d"][g, 9:] = rot / 6, mb
                    st["d-cs"][g, 9:] = np.cos(th), np.sin(th), mb
                    st["d-fourier"][g, 9:] = (np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th), np.cos(3 * th),
                                              mb)
        self.static = st

    # -- rules
    def digits_of(self, s: int, index: int) -> np.ndarray:
        return (int(index) // self.radix[s]) % self.n_opt[s]

    def index_of(self, s: int, digits) -> int:
        return int((np.asarray(digits, np.int64) * self.radix[s]).sum())

    def key(self, s: int, digits) -> str:
        return f"hex|{self.keys[s]}|" + ".".join(str(self.options[s][t][int(d)]) for t, d in enumerate(digits))

    def code(self, s: int, digits) -> np.ndarray:
        """uint8 [53]: the T1 code."""
        out = np.zeros(CODE_BITS, np.uint8)
        for j, m in enumerate(MAJORS):
            out[j] = m in self.edges[s]
        out[len(MAJORS) + np.arange(N_TYPES) * N_DIGITS + np.asarray(digits, np.int64)] = 1
        return out

    def decode(self, code: np.ndarray):
        """(class bits [8] bool, digits [9]) from code scores [..., 53] (bits > 0.5, digits by argmax)."""
        code = np.asarray(code)
        return code[..., :8] > 0.5, code[..., 8:].reshape(code.shape[:-1] + (N_TYPES, N_DIGITS)).argmax(-1)

    def describe(self, s: int, digits) -> str:
        return f"{self.keys[s]}·" + "".join(str(int(d)) for d in digits)

    # -- the split
    def _hashes(self, s: int, salt: str) -> np.ndarray:
        """uint32 [count]: fmix32(FNV-1a(salt|key)) of every rule of subset s, in index order (vectorised)."""
        n = int(self.count[s])
        idx = np.arange(n, dtype=np.int64)
        with np.errstate(over="ignore"):
            h = np.full(n, _fnv_bytes(np.uint32(0x811C9DC5), f"{salt}|hex|{self.keys[s]}|".encode()), np.uint32)
            for t in range(N_TYPES):
                toks = [str(o).encode() + (b"." if t < N_TYPES - 1 else b"") for o in self.options[s][t]]
                dig = (idx // self.radix[s, t]) % self.n_opt[s, t]
                for pos in range(max(len(x) for x in toks)):
                    byte = np.array([x[pos] if pos < len(x) else 0 for x in toks], np.uint32)[dig]
                    live = np.array([pos < len(x) for x in toks])[dig]
                    nh = (h ^ byte) * np.uint32(0x01000193)
                    h = np.where(live, nh, h)
            return _fmix(h)

    def held_out(self, s: int) -> np.ndarray:
        """bool [count]: the v2 held-out rules of subset s."""
        if self._held is None:
            self._held = [None] * self.n_sub
        if self._held[s] is None:
            n = int(self.count[s])
            k = int(round(HELDOUT_SHARE * n))  # n * 0.2 never ends in .5 for these counts: no rounding ambiguity
            order = np.lexsort((np.arange(n), self._hashes(s, SPLIT_SALT)))
            held = np.zeros(n, bool)
            held[order[:k]] = True
            self._held[s] = held
        return self._held[s]

    def legacy_heldout(self) -> list[tuple[int, int]]:
        """The v1 split's 20 held-out rules as (s, index): per subset without class 0, the round(0.2 n) with the
        smallest fmix32(FNV-1a('strand-split-v1|' + key)) (ties: enumeration order, as the exporter's stable sort)."""
        if self._legacy is None:
            out = []
            for s in range(self.n_sub):
                if 0 in self.edges[s]:
                    continue
                n = int(self.count[s])
                order = np.lexsort((np.arange(n), self._hashes(s, SPLIT_SALT_V1)))
                out += [(s, int(i)) for i in sorted(order[:int(round(HELDOUT_SHARE * n))])]
            self._legacy = out
        return self._legacy

    def allowed(self, split: str = "train") -> list[np.ndarray]:
        """Per subset, bool [count]: the rules of `split` ('train': neither v2 held-out nor legacy held-out;
        'heldout': v2 held-out)."""
        if split == "heldout":
            return [self.held_out(s) for s in range(self.n_sub)]
        if split != "train":
            raise ValueError(split)
        if getattr(self, "_train", None) is None:
            tr = [~self.held_out(s) for s in range(self.n_sub)]
            for s, i in self.legacy_heldout():
                tr[s][i] = False
            self._train = tr
        return self._train

    def is_train(self, s: int, index: int) -> bool:
        return bool(self.allowed("train")[s][int(index)])

    def sample(self, rng: np.random.Generator, split: str = "train") -> tuple[int, np.ndarray]:
        """(s, digits): the subset uniform over the 7, then the rule uniform within it, from `split` ('train':
        neither v2 held-out nor legacy held-out; 'heldout': v2 held-out)."""
        ok = self.allowed(split)
        s = int(rng.integers(self.n_sub))  # the subset first: rejection only within it keeps the subsets uniform
        while True:
            digits = np.array([int(rng.integers(n)) for n in self.n_opt[s]], np.int64)
            if ok[s][self.index_of(s, digits)]:
                return s, digits

    # -- boards and strands
    def render_bits(self, s: int, digits, geo: np.ndarray) -> np.ndarray:
        """int16 [H,W]: the rule's chords per cell (bit p = PAIRS[p]); 0 off the board."""
        on = geo >= 0
        g = np.where(on, geo, 0)
        t, rot, mb = g // 12, (g // 2) % 6, g % 2
        dig = np.asarray(digits, np.int64)[t]
        return np.where(on, self.lut_bits[s, t, dig, rot, mb], 0).astype(np.int16)

    def exits(self, s: int, digits, geo: np.ndarray) -> np.ndarray:
        """int8 [6,H,W]: walker.exit_table of the rule's chords (exit across d for a line entering across d)."""
        on = geo >= 0
        g = np.where(on, geo, 0)
        t, rot, mb = g // 12, (g // 2) % 6, g % 2
        dig = np.asarray(digits, np.int64)[t]
        ex = self.lut_exit[s, t, dig, rot, mb]  # [H,W,6]
        return np.where(on[..., None], ex, -1).transpose(2, 0, 1).copy()


def chord_planes(bits: np.ndarray) -> np.ndarray:
    """uint8 [15,H,W] from int16 chord bits [H,W]."""
    return ((bits[None].astype(np.int32) >> np.arange(15)[:, None, None]) & 1).astype(np.uint8)


def random_chord(rng: np.random.Generator, bits: np.ndarray):
    """(row, col, d0, d1): a uniformly random chord of the board (a random orientation), or None if it has none."""
    planes = chord_planes(bits)
    p, rows, cols = np.nonzero(planes)
    if not len(p):
        return None
    k = int(rng.integers(len(p)))
    d0, d1 = PAIRS[p[k]]
    if rng.integers(2):
        d0, d1 = d1, d0
    return int(rows[k]), int(cols[k]), int(d0), int(d1)


# ---------------------------------------------------------------- boards

class Boards:
    """boards.npz: per group the geo planes [B,H,W] (int16, -1 off board), sizes and mirror signs."""

    def __init__(self, data_dir: str | None = None):
        self.dir = data_dir or default_dir()
        with np.load(os.path.join(self.dir, "boards.npz")) as z:
            a = {k: z[k] for k in z.files}
        self.geo, self.hw, self.mirror, self.tiles = {}, {}, {}, {}
        for g in GROUPS:
            t, r, m = a[f"{g}_type"].astype(np.int16), a[f"{g}_rot"].astype(np.int16), a[f"{g}_mirror"]
            mb = (m < 0).astype(np.int16)[:, None, None]
            self.geo[g] = np.where(t >= 0, t * 12 + r * 2 + mb, -1).astype(np.int16)
            self.hw[g] = np.stack([a[f"{g}_h"], a[f"{g}_w"]], 1)
            self.mirror[g] = m
            self.tiles[g] = a[f"{g}_tiles"]

    def board(self, group: str, i: int) -> np.ndarray:
        h, w = self.hw[group][i]
        return self.geo[group][i, :h, :w]


# ---------------------------------------------------------------- checks

def same_strand(a: Strand, b: Strand) -> str | None:
    if a.closed != b.closed:
        return f"closed {a.closed} != {b.closed}"
    if len(a) != len(b):
        return f"length {len(a)} != {len(b)}"
    for k in ("rows", "cols", "ins", "outs", "index"):
        if not np.array_equal(getattr(a, k), getattr(b, k)):
            return f"{k} differs at step {int(np.nonzero(getattr(a, k) != getattr(b, k))[0][0])}"
    return None


def parity(data_dir=None, decompose_all=True, out=print) -> int:
    """Rendering and walks for parity.npz's rules (every subset, class 0 included) against Spectacle's. Returns
    the number of mismatches."""
    from .walker import decompose

    t0 = time.time()
    tab, bd = RuleTable(data_dir), Boards(data_dir)
    with np.load(os.path.join(tab.dir, "parity.npz")) as z:
        p = {k: z[k] for k in z.files}
    groups = ("L2", "L3", "L4")
    n = len(p["sample_subset"])
    bad, render_bad, walk_bad, dec_bad = [], 0, 0, 0
    per_sub = np.zeros((tab.n_sub, 2), np.int64)  # (rules, taps)
    strand_of = np.split(np.arange(len(p["strand_len"])), np.searchsorted(p["strand_board"], np.arange(1, n)))
    tap_of = np.split(np.arange(len(p["tap_board"])), np.searchsorted(p["tap_board"], np.arange(1, n)))
    for j in range(n):
        s, idx = int(p["sample_subset"][j]), int(p["sample_index"][j])
        digits = tab.digits_of(s, idx)
        geo = bd.board(groups[p["sample_group"][j]], int(p["sample_board"][j]))
        h, w = geo.shape
        bits = tab.render_bits(s, digits, geo)
        if not np.array_equal(bits, p["chord_bits"][j, :h, :w]):
            render_bad += 1
            bad.append(f"sample {j} ({tab.describe(s, digits)}): chord bits differ")
            continue
        ex, mask = tab.exits(s, digits, geo), geo >= 0
        per_sub[s, 0] += 1
        for i in tap_of[j]:
            a, e = p["tap_ptr"][i], p["tap_ptr"][i + 1]
            g = lambda k: p[k][a:e].astype(np.int64)  # noqa: E731
            theirs = Strand(g("step_row"), g("step_col"), g("step_in"), g("step_out"), g("step_index"),
                            bool(p["tap_closed"][i]))
            mine = walk(ex, mask, int(p["tap_row"][i]), int(p["tap_col"][i]), int(p["tap_d0"][i]), int(p["tap_d1"][i]))
            why = same_strand(mine, theirs)
            per_sub[s, 1] += 1
            if why:
                walk_bad += 1
                bad.append(f"sample {j} ({tab.describe(s, digits)}) tap {i}: {why}")
        if decompose_all:
            _, lens, closed = decompose(ex, mask)
            mine = sorted(zip(lens.tolist(), closed.tolist()))
            theirs = sorted(zip(p["strand_len"][strand_of[j]].tolist(), p["strand_closed"][strand_of[j]].astype(bool).tolist()))
            if mine != theirs:
                dec_bad += 1
                bad.append(f"sample {j} ({tab.describe(s, digits)}): decomposition differs")
    out(f"parity: {n} rule-boards ({', '.join(f'{tab.keys[s]} {per_sub[s, 0]}' for s in range(tab.n_sub))}); "
        f"chord bits identical {n - render_bad}/{n}; walks identical {per_sub[:, 1].sum() - walk_bad}/{per_sub[:, 1].sum()}"
        + (f"; whole-board decompositions identical {n - render_bad - dec_bad}/{n - render_bad}" if decompose_all else "")
        + f"; {time.time() - t0:.1f} s")
    for line in bad[:20]:
        out("MISMATCH " + line)
    return len(bad)


def check_split(data_dir=None, out=print) -> int:
    """The v2 split and the keys against the exporter's (TypeScript) values. Returns the number of mismatches."""
    t0 = time.time()
    tab = RuleTable(data_dir)
    bad = 0
    for sm in tab.meta["samples"]:
        s, digits = sm["subset"], sm["digits"]
        ok = (tab.key(s, digits) == sm["key"] and split_hash(f"{SPLIT_SALT}|{sm['key']}") == sm["hash"]
              and tab.index_of(s, digits) == sm["index"] and list(tab.digits_of(s, sm["index"])) == digits)
        bad += not ok
    h0 = tab._hashes(6, SPLIT_SALT)[:2000]
    want = [split_hash(f"{SPLIT_SALT}|{tab.key(6, tab.digits_of(6, i))}") for i in range(2000)]
    bad += not np.array_equal(h0, np.array(want, np.uint32))
    rows = []
    for s in range(tab.n_sub):
        held = tab.held_out(s)
        idx = np.nonzero(held)[0]
        x = 0
        for v in idx.tolist():
            x ^= v
        ref = tab.meta["subsets"][s]["split"]
        mine = {"n": int(tab.count[s]), "k": int(held.sum()), "xor": x, "sum": int(idx.sum()) % 2 ** 32}
        ok = all(mine[k] == ref[k] for k in mine)
        bad += not ok
        rows.append(f"{tab.keys[s]} {mine['k']}/{mine['n']}{'' if ok else ' MISMATCH'}")
    legacy = tab.legacy_heldout()
    out(f"split v2: {len(tab.meta['samples'])} keyed samples and 2,000 vectorised hashes "
        f"{'match' if bad == 0 else 'MISMATCH'}; held out per subset {', '.join(rows)}; legacy held-out "
        f"{len(legacy)} rules; {time.time() - t0:.1f} s")
    return bad


def check_legacy(tab: RuleTable, legacy_dir: str, out=print) -> int:
    """The v1 split recomputed here = data/strand/meta.json's HELD-OUT rules, and each legacy rule's per-type
    local pairs = the table's."""
    with open(os.path.join(legacy_dir, "meta.json")) as f:
        meta = json.load(f)
    bad = 0
    mine = set(tab.legacy_heldout())
    theirs = set()
    for r in meta["rules"]:
        s = tab.keys.index(r["subset"])
        digits = [tab.options[s][t].index(m) for t, m in enumerate(r["matching"])]
        i = tab.index_of(s, digits)
        if r["split"] == "heldout":
            theirs.add((s, i))
        lp = [tab.local_pairs[s][t][d] for t, d in enumerate(digits)]
        bad += lp != r["local_pairs"]
    bad += mine != theirs
    out(f"legacy: the v1 held-out set recomputed {'=' if mine == theirs else '!='} meta.json's ({len(theirs)} rules); "
        f"local pairs of all {len(meta['rules'])} legacy rules {'=' if bad == 0 else '!='} the table's")
    return bad


def bench(data_dir=None, seconds=5.0, out=print):
    """Fresh (rule, board, tap, walk) samples per second per level: the trainer's data path, minus the planes."""
    tab, bd = RuleTable(data_dir), Boards(data_dir)
    rng = np.random.default_rng(0)
    tab.sample(rng)  # the split, once
    res = {}
    for g in ("L2", "L3", "L4"):
        n, steps, t0 = 0, 0, time.time()
        while time.time() - t0 < seconds:
            geo = bd.board(g, int(rng.integers(len(bd.hw[g]))))
            s, digits = tab.sample(rng)
            bits = tab.render_bits(s, digits, geo)
            tap = random_chord(rng, bits)
            if tap is None:
                continue
            st = walk(tab.exits(s, digits, geo), geo >= 0, *tap)
            n, steps = n + 1, steps + len(st)
        res[g] = round(n / (time.time() - t0), 1)
        out(f"{g}: {res[g]} boards/s (rule + render + tap + walk; mean strand {steps / max(1, n):.1f} chords)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data/strand-v2")
    ap.add_argument("--legacy", default=None, help="data/strand (the v1 dataset) for the legacy cross-check")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--no-decompose", action="store_true", help="--parity: taps only, not every strand")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args(argv)
    bad = 0
    if args.split:
        bad += check_split(args.data)
        legacy = args.legacy or os.path.join(os.path.dirname(default_dir()), "strand")
        if os.path.isfile(os.path.join(legacy, "meta.json")):
            bad += check_legacy(RuleTable(args.data), legacy)
    if args.parity:
        bad += parity(args.data, not args.no_decompose)
    if args.bench:
        bench(args.data)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
