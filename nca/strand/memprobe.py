"""Training memory per arm, measured on the CPU (no GPU here), to size batches for the L4.

    python -m nca.strand.memprobe --inputs e --channels 96 --hidden 128 --depth 1 --batch 8 --window 4 12

For each backprop window G it runs, in a FRESH process, one gradient window of nca/strand/train2.py's iteration
on level-4 crops (the largest boards, S 42): consts built by train2.Planes, G steps of the model with the loss
over the last min(8, G) steps, backward. It reports the process's peak RSS above its RSS just before the window
(VmHWM after resetting it, so every transient counts, E's checkpointed recompute included). Two windows give the memory per
step (the slope: activations kept for backward) and the rest (the intercept: the largest transient); both scale
with batch x cells. --eval adds a no-gradient read-out at the eval batch. The GPU adds its context (~0.3-0.5 GB a
process) and the allocator's slack, and holds the pool states (slots x channels x S^2 floats per level).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch


def status_mb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    return float("nan")


def reset_peak():
    """Linux: writing 5 to clear_refs resets the peak RSS (VmHWM) to the current RSS."""
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def one(args):
    from . import train2 as T
    from .rules import Boards, RuleTable

    torch.set_num_threads(args.threads)
    tab, bd = RuleTable(), Boards()
    codes = T.Codes(tab)
    rng = np.random.default_rng(0)
    S = int(bd.hw["L4"].max())
    slots = [T.new_slot(rng, tab, bd, "L4", S, "m1a") for _ in range(args.batch)]
    stack = lambda k: np.stack([s[k] for s in slots])  # noqa: E731
    model = T.make_model(args.channels, args.hidden, [-2.0, 2.0], T.n_inputs(args.inputs), args.depth, args.inputs,
                         None)
    with torch.no_grad():
        model.w2.normal_(0, 0.01)
    planes = T.Planes(tab, args.inputs, torch.device("cpu"))
    a = torch.from_numpy(stack("a")).float()
    b = torch.from_numpy(stack("b")).float()
    oc = torch.from_numpy(stack("oc")).float()
    closed = torch.from_numpy(stack("closed"))
    cs = planes(stack("geo"), codes(stack("rule")), stack("tap"))
    state = T.fresh(cs[:, :1], args.channels)
    mk = cs[:, :1]
    ncell = 6 * mk.sum((1, 2, 3)).clamp(min=1)
    with torch.no_grad():
        for _ in range(2):
            state = model.step(state, mk, cs)
    base = status_mb("VmRSS")
    reset_peak()
    G, K = args.window, min(8, args.window)
    if args.eval:
        with torch.no_grad():
            ev = T.fresh(cs[:, :1], args.channels)
            for _ in range(G):
                ev = model.step(ev, mk, cs)
    else:
        seen, saved = set(), [0]

        def pack(x):  # autograd's saved tensors, each storage once (device-independent; E's checkpoint inputs are
            # kept alive by the checkpoint itself, not saved here)
            st = x.untyped_storage()
            if st.data_ptr() not in seen:
                seen.add(st.data_ptr())
                saved[0] += st.nbytes()
            return x
        acc = 0.0
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            for t in range(G):
                state = model.step(state, mk, cs)
                if t >= G - K:
                    age = torch.full((args.batch, 1, 1, 1), float(t + 50))
                    acc = acc + T.step_loss(state, "m1a", a, b, oc, closed, age, mk, ncell) / K
        acc.mean().backward()
    peak = status_mb("VmHWM")
    print(json.dumps({"inputs": args.inputs, "channels": args.channels, "hidden": args.hidden, "depth": args.depth,
                      "batch": args.batch, "window": G, "S": S, "eval": args.eval, "baseMB": round(base, 1),
                      "peakAboveMB": round(peak - base, 1),
                      "savedMB": None if args.eval else round(saved[0] / 2 ** 20, 1),
                      "peakMB": round(peak, 1)}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="c")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, nargs="+", default=[4, 12])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval", action="store_true", help="a no-gradient read-out instead of a gradient window")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        args.window = args.window[0]
        return one(args)
    rows = []
    for G in args.window:
        cmd = [sys.executable, "-m", "nca.strand.memprobe", "--child", "--inputs", args.inputs, "--channels",
               str(args.channels), "--hidden", str(args.hidden), "--depth", str(args.depth), "--batch", str(args.batch),
               "--window", str(G), "--threads", str(args.threads)] + (["--eval"] if args.eval else [])
        out = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
        if out.returncode:
            raise SystemExit(out.stderr)
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) >= 2 and not args.eval:
        (g0, m0), (g1, m1) = (rows[0]["window"], rows[0]["peakAboveMB"]), (rows[-1]["window"], rows[-1]["peakAboveMB"])
        slope = (m1 - m0) / (g1 - g0)
        sv = (rows[-1]["savedMB"] - rows[0]["savedMB"]) / (g1 - g0)
        print(json.dumps({"perStepMB": round(slope, 2), "restMB": round(m0 - slope * g0, 1),
                          "savedPerStepMB": round(sv, 2), "batch": args.batch, "S": rows[0]["S"]}), flush=True)


if __name__ == "__main__":
    main()
