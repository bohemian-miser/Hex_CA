"""Dataset statistics for the trainer (docs/strand-data.md).

    python -m nca.strand.stats [--data data/strand]

Per split and level: per strand (every strand of every board, the M1b "whole pattern") and per tap (as
sampled: uniform over chords, so length-weighted), lengths in steps (chords), closed shares, strands per
board, growth steps (two-way from the tap, one chord per step; one-way through d1 = a Spectacle head),
cells where a circuit and a tail meet (the M1b per-cell ambiguity), and disk size. Then the same per
rule subset at level 3.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .loader import StrandFile, default_dir, load_meta


def pct(a, ps=(50, 90, 99)) -> str:
    a = np.asarray(a)
    if a.size == 0:
        return "-"
    return " / ".join(f"{np.percentile(a, p):.0f}" for p in ps) + f" / {a.max()}"


def collect(data: str):
    meta = load_meta(data)
    rules = {r["id"]: r for r in meta["rules"]}
    g = defaultdict(lambda: defaultdict(list))
    for fm in meta["files"]:
        f = StrandFile(f"{data}/{fm['path']}")
        keys = [(fm["split"], f.level, "all"), (fm["split"], f.level, rules[f.rule_id]["subset"])]
        ln, cl, sb = f["strand_len"], f["strand_closed"].astype(bool), f["strand_board"]
        per_board = np.bincount(sb, minlength=f.n_boards)
        tl, tc, grow, ahead = [], [], [], []
        for i in range(f.n_taps):
            s = f.tap(i)
            n = len(s)
            tl.append(n)
            tc.append(s.closed)
            grow.append(s.grow_steps())
            ahead.append(n - 1 if s.closed else n - 1 - s.tap_pos)
        # Cells where chords of a circuit and of a tail meet; cells a strand visits twice.
        mixed = cells = twice = 0
        for b in range(f.n_boards):
            h, w = f.hw(b)
            sid = f["chord_strand"][b, :, :h, :w]
            on = sid >= 0
            c_cl = np.zeros(sid.shape, bool)
            c_cl[on] = cl[sid[on]]
            has_cl = c_cl.any(0)
            has_open = (on & ~c_cl).any(0)
            cells += int(on.any(0).sum())
            mixed += int((has_cl & has_open).sum())
            s_sorted = np.sort(np.where(on, sid, -1), axis=0)
            twice += int(((s_sorted[1:] == s_sorted[:-1]) & (s_sorted[1:] >= 0)).any(0).sum())
        for k in keys:
            d = g[k]
            d["len"].extend(ln.tolist())
            d["closed"].extend(cl.tolist())
            d["len_closed"].extend(ln[cl].tolist())
            d["len_tail"].extend(ln[~cl].tolist())
            d["per_board"].extend(per_board.tolist())
            d["tap_len"].extend(tl)
            d["tap_closed"].extend(tc)
            d["grow"].extend(grow)
            d["ahead"].extend(ahead)
            d["mixed"].append(mixed)
            d["cells"].append(cells)
            d["twice"].append(twice)
            d["bytes"].append(fm["bytes"])
            d["boards"].append(f.n_boards)
            d["tiles"].extend(f["board_tiles"].tolist())
            d["rules"].append(f.rule_id)
    return meta, g


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_dir())
    args = ap.parse_args(argv)
    meta, g = collect(args.data)
    keys = sorted(g, key=lambda k: (k[0] != "train", k[1], k[2] != "all", k[2]))

    print("| split | L | rules | boards | tiles/board | strands/board | strand len p50/p90/p99/max | closed (strands / chords) "
          "| circuit len p50/p99/max | tail len p50/p99/max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        if k[2] != "all":
            continue
        d = g[k]
        ln, cl = np.array(d["len"]), np.array(d["closed"])
        pb, tiles = np.array(d["per_board"]), np.array(d["tiles"])
        print(f"| {k[0]} | {k[1]} | {len(set(d['rules']))} | {sum(d['boards'])} | {tiles.min()}-{tiles.max()} "
              f"| {pb.mean():.0f} ({pb.min()}-{pb.max()}) | {pct(ln)} | {cl.mean():.2f} / {ln[cl].sum() / ln.sum():.2f} "
              f"| {pct(d['len_closed'], (50, 99))} | {pct(d['len_tail'], (50, 99))} |")

    print()
    print("| split | L | taps | tap strand len p50/p90/p99/max | closed | two-way grow p50/p90/p99/max "
          "| one-way (head) p99/max | circuit+tail cells | twice-visited cells | MB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        if k[2] != "all":
            continue
        d = g[k]
        ahead = np.array(d["ahead"])
        print(f"| {k[0]} | {k[1]} | {len(d['tap_len'])} | {pct(d['tap_len'])} | {np.mean(d['tap_closed']):.2f} "
              f"| {pct(d['grow'])} | {np.percentile(ahead, 99):.0f} / {ahead.max()} "
              f"| {sum(d['mixed']) / sum(d['cells']):.3f} | {sum(d['twice']) / sum(d['cells']):.3f} "
              f"| {sum(d['bytes']) / 1e6:.1f} |")

    print()
    print("| split | L | subset | rules | strand len p50/p90/p99/max | closed strands | tap len p99 | two-way grow p99/max |")
    print("|---|---|---|---|---|---|---|---|")
    for k in keys:
        if k[2] == "all" or k[1] == 2:
            continue
        d = g[k]
        grow = np.array(d["grow"])
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(set(d['rules']))} | {pct(d['len'])} | {np.mean(d['closed']):.2f} "
              f"| {np.percentile(d['tap_len'], 99):.0f} | {np.percentile(grow, 99):.0f} / {grow.max()} |")
    total = sum(f["bytes"] for f in meta["files"])
    t = meta["timing_s"]
    print()
    print(f"files {len(meta['files'])}, {total / 1e6:.1f} MB; generation {t['total']:.0f} s "
          f"(fields and lattices {t['lattices']:.1f} s); walk stops {meta['walkStops']}")


if __name__ == "__main__":
    main()
