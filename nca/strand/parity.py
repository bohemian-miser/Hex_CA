"""Parity: the Python walker, from the 15 chord-bit planes alone, against Spectacle's walkStrand.

    python -m nca.strand.parity [--data data/strand] [--samples 2000] [--seed 0] [--all]

1. walks: `--samples` taps drawn uniformly from every exported tap (all splits and levels; --all = every
   tap), each walked by walker.walk from the chord planes; cells, entry/exit edges, order, signed index and
   the closed flag must equal the exported walkStrand walk exactly;
2. boards: on every board of the files touched, walker.decompose gives the same partition of the chords into
   strands as walkStrand's (chord_strand), with the same lengths and closed flags;
3. rendering: the chord planes equal the rule's per-type local chords rotated by tile_rot (meta.json
   local_pairs), i.e. tile type + rotation + rule determine the planes and nothing else does.
Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .loader import StrandFile, default_dir, load_meta
from .walker import decompose, render_chords


def same_strand(a, b) -> str | None:
    if a.closed != b.closed:
        return f"closed {a.closed} != {b.closed}"
    if len(a) != len(b):
        return f"length {len(a)} != {len(b)}"
    for k in ("rows", "cols", "ins", "outs", "index"):
        if not np.array_equal(getattr(a, k), getattr(b, k)):
            i = int(np.nonzero(getattr(a, k) != getattr(b, k))[0][0])
            return f"{k} differs first at step {i}"
    return None


def same_partition(sid_a: np.ndarray, sid_b: np.ndarray) -> bool:
    on = sid_a >= 0
    if not np.array_equal(on, sid_b >= 0):
        return False
    pairs = np.unique(np.stack([sid_a[on], sid_b[on]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_dir())
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every tap and every board, not a sample")
    args = ap.parse_args(argv)
    t0 = time.time()
    meta = load_meta(args.data)
    rules = {r["id"]: r for r in meta["rules"]}
    files = meta["files"]
    taps = np.array([f["taps"] for f in files])
    rng = np.random.default_rng(args.seed)
    if args.all:
        picks = [(k, i) for k in range(len(files)) for i in range(taps[k])]
    else:
        flat = rng.choice(taps.sum(), size=args.samples, replace=False)
        starts = np.concatenate([[0], np.cumsum(taps)])
        fk = np.searchsorted(starts, flat, side="right") - 1
        picks = sorted(zip(fk.tolist(), (flat - starts[fk]).tolist()))
    cache: dict[int, StrandFile] = {}
    bad = []
    by_level: dict[int, list[int]] = {}
    for k, i in picks:
        if k not in cache:
            cache[k] = StrandFile(f"{args.data}/{files[k]['path']}")
        f = cache[k]
        b, row, col, d0, d1 = f.tap_args(i)
        mine, theirs = f.walk(b, row, col, d0, d1), f.tap(i)
        why = same_strand(mine, theirs)
        by_level.setdefault(f.level, [0, 0])[0] += 1
        if why:
            by_level[f.level][1] += 1
            bad.append(f"{files[k]['path']} tap {i}: {why}")
    n_ok = len(picks) - len(bad)
    print(f"walks: {n_ok}/{len(picks)} identical "
          + ", ".join(f"L{lv} {n - m}/{n}" for lv, (n, m) in sorted(by_level.items())))

    boards = boards_bad = render_bad = strands = 0
    for f in cache.values():
        lp = rules[f.rule_id]["local_pairs"]
        for b in range(f.n_boards):
            h, w = f.hw(b)
            boards += 1
            sid, lengths, closed = decompose(f.exits(b), f.mask(b))
            theirs = f["chord_strand"][b, :, :h, :w]
            mine_ids = np.unique(sid[sid >= 0])
            ok = same_partition(sid, theirs)
            if ok:
                # Same partition: compare lengths and closed flags strand by strand.
                on = sid >= 0
                m = dict(zip(sid[on].tolist(), theirs[on].tolist()))
                for s in mine_ids:
                    t = m[int(s)]
                    ok &= lengths[s] == f["strand_len"][t] and closed[s] == bool(f["strand_closed"][t])
            strands += len(mine_ids)
            if not ok:
                boards_bad += 1
                bad.append(f"{f.path} board {b}: decomposition differs")
            r = render_chords(f["tile_type"][b, :h, :w], f["tile_rot"][b, :h, :w], int(f["board_mirror"][b]), lp)
            if not np.array_equal(r, f.chords(b)):
                render_bad += 1
                bad.append(f"{f.path} board {b}: chord planes != type + rotation rendering")
    print(f"boards: {boards - boards_bad}/{boards} decompositions identical ({strands} strands); "
          f"{boards - render_bad}/{boards} chord planes = type + rotation rendering")
    print(f"{time.time() - t0:.1f} s")
    for line in bad[:20]:
        print("MISMATCH", line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
