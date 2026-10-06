"""Backfill gallery.npz for runs trained before the dashboard gallery panel existed (nca/dashboard.py):
loads each run's best.pt (else ckpt.pt), rebuilds its model from the checkpoint's OWN config (so it need
not match whatever --channels/--hidden this process would otherwise guess), rolls out the SAME fixed
held-out taps (train.GALLERY_LEVELS x GALLERY_PER_LEVEL) on the CPU, and writes runs/<name>/gallery.npz --
the same file, the same contract, as if the run had been training with the fix from the start.

    python -m nca.strand.backfill_gallery tap-c-l3 tap-big-l3 m128-l4
    python -m nca.strand.backfill_gallery tap-c-l3 --runs-dir /path/to/runs

A v1 checkpoint (nca.strand.train, e.g. m128-l4: plain chord-plane input, "channels"/"hidden" in its
config, no "inputs") uses train.py's EvalLevel + gallery_rollout. A v2 checkpoint (nca.strand.train2, e.g.
tap-c-l3 / tap-big-l3: "inputs" in its config) uses train2.py's legacy_set + gallery_rollout. Both draw
from the SAME EVAL_SEED-seeded, bucket-stratified pick of data/strand's eval boards (train.EvalLevel and
train2.legacy_set share that draw at n = GALLERY_PER_LEVEL), so the SAME physical taps line up across
architectures -- m128-l4 (pre-rendered chords) next to tap-c-l3 / tap-big-l3 (the rule at the tap) is a
fair, if-only-the-input-differs comparison.
"""

import argparse
import os

import numpy as np
import torch

from . import train as V1
from . import train2 as V2
from .loader import load_meta


def load_checkpoint(run_dir):
    for name in ("best.pt", "ckpt.pt"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return torch.load(path, map_location="cpu", weights_only=False), path
    raise SystemExit(f"no best.pt or ckpt.pt in {run_dir}")


def backfill_v1(cfg, state_dict, legacy_dir):
    """m128-l4 and friends: nca.strand.train's plain chord-plane input."""
    meta = load_meta(legacy_dir)
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    model = V1.make_model(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg.get("perception", "taps"))
    model.load_state_dict(state_dict)
    model.eval()
    stepper = lambda st, w, cs, t, sl: model.step(st, w, cs)  # noqa: E731
    groups = {}
    for L, ev in V1.gallery_sets(legacy_dir, subset_of).items():
        pred = V1.gallery_rollout(stepper, ev, cfg["channels"], torch.device("cpu"))
        mask = ev.consts[:, 0] > 0
        groups[L] = {"mask": mask, "rot": np.full(mask.shape, -1, np.int8), "tap": ev.items["tap"],
                     "tgt": ev.a < V1.INF, "pred": pred,
                     "rule": [f"{s}#{r}" for s, r in zip(ev.items["subset"], ev.items["ruleId"])],
                     "length": ev.items["length"], "ideal": ev.items["ideal"]}
    return groups


def backfill_v2(cfg, state_dict, data_dir, legacy_dir):
    """tap-c-l3, tap-big-l3 and friends: nca.strand.train2's rule-at-the-tap input."""
    tab = V2.RuleTable(data_dir)
    codes = V2.Codes(tab)
    model = V2.make_model(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["nIn"], cfg.get("depth", 1),
                          cfg["inputs"], cfg.get("dirGroups"))
    model.load_state_dict(state_dict)
    model.eval()
    planes = V2.Planes(tab, cfg["inputs"], torch.device("cpu"))
    stepper = lambda st, w, cs, t, sl: model.step(st, w, cs)  # noqa: E731
    groups = {}
    for L in V1.GALLERY_LEVELS:
        ev = V2.legacy_set("m1a", tab, legacy_dir, L, V1.GALLERY_PER_LEVEL, 8, 2000)
        pred = V2.gallery_rollout(stepper, planes, ev, cfg["channels"], torch.device("cpu"), codes)
        rot = np.where(ev.geo >= 0, (ev.geo % 12) // 2, -1).astype(np.int8)
        rule = [f"{s}/{'-'.join(str(int(d)) for d in r[1:])}" for s, r in zip(ev.items["subset"], ev.rule)]
        groups[L] = {"mask": ev.geo >= 0, "rot": rot, "tap": ev.tap, "tgt": ev.a < V2.INF, "pred": pred,
                     "rule": rule, "length": ev.items["length"], "ideal": ev.items["ideal"]}
    return groups


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run names under --runs-dir, each with a best.pt or ckpt.pt")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data", default=None, help="data/strand-v2 (v2 runs only); default nca.strand.rules' own")
    p.add_argument("--legacy", default=None, help="data/strand (the gallery's taps, both architectures); "
                   "default nca.strand.loader's own")
    args = p.parse_args(argv)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    legacy_dir = V1.ensure_data(args.legacy)
    for name in args.runs:
        run_dir = os.path.join(args.runs_dir, name)
        ck, path = load_checkpoint(run_dir)
        cfg = ck["config"]
        is_v2 = "inputs" in cfg
        print(f"{name}: {'v2' if is_v2 else 'v1'} checkpoint ({path}), "
              f"channels={cfg['channels']} hidden={cfg['hidden']}, iteration {ck.get('iteration', 0)}")
        if is_v2:
            data_dir = V2.ensure_dir(args.data, "strand-v2", "rules-hex.json")
            groups = backfill_v2(cfg, ck["model"], data_dir, legacy_dir)
        else:
            groups = backfill_v1(cfg, ck["model"], legacy_dir)
        it = int(ck.get("iteration", 0))
        V1.write_gallery(os.path.join(run_dir, "gallery.npz"), it, groups)
        print(f"{name}: wrote {os.path.join(run_dir, 'gallery.npz')} (iteration {it})")


if __name__ == "__main__":
    main()
