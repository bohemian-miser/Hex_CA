"""The per-cell lookup probe of docs/spectacle-nca-options.md §3 (no CA): how fast and how exactly a
1-hidden-layer MLP learns a cell's chords from its static planes and the rule code, on HELD-OUT rules.

    python -m nca.strand.probe [--samples 1900000] [--batch 2048] [--hidden 128 1024] [--arms a c d d-cs ...]

Each training batch is fresh random draws: a cell geometry (type, rotation, mirror) uniform over the 108,
and a TRAIN rule (subset uniform over the 7, then uniform within it; rules.RuleTable.sample's split). The
test set is 20,000 fixed draws with v2 HELD-OUT rules. Arms (rules.py's static planes + the 53-bit code ->
target):
  a          A (16: type, rotation one-hot, mirror)    -> the cell's 15 chord bits in the board frame
  c          C (55: per-direction classes, anchor, mirror) -> the same
  d          D (11: type, rot / 6, mirror)             -> the same
  d-cs       D-cs (12: type, cos, sin, mirror)         -> the same
  d-fourier  D-fourier (15: type, cos/sin k theta, k = 1..3, mirror) -> the same
  e          E's local frame (9: type only)            -> the 15 chord bits in the tile's own frame (local pairs)
Every arm runs at every --hidden width (one hidden layer).
Reported per arm: held-out bit accuracy and cell-exact (all 15 bits right) at each check, cell-exact on 20,000
fixed draws of TRAIN rules too (is the lookup learned at all, apart from generalising to new rules), the samples and
seconds to reach 0.9 / 0.99 / 0.999 / 1.0 cell-exact, the final numbers and cell-exact per subset. Same width,
lr (x0.3 at 60 % of the samples, x0.1 at 85 %), batch and seed for all.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from .rules import CODE_BITS, N_DIGITS, N_GEO, N_TYPES, RuleTable


def draws(tab: RuleTable, rng: np.random.Generator, n: int, split: str):
    """n random (geo, s, digits) with rules from `split` ('train' / 'heldout'), vectorised with rejection."""
    allowed = tab.allowed(split)
    s = rng.integers(tab.n_sub, size=n)  # subsets uniform; rejection only within a subset
    d = np.zeros((n, N_TYPES), np.int64)
    todo = np.arange(n)
    while len(todo):
        st = s[todo]
        dd = np.floor(rng.random((len(todo), N_TYPES)) * tab.n_opt[st]).astype(np.int64)
        idx = (dd * tab.radix[st]).sum(1)
        ok = np.zeros(len(todo), bool)
        for k in range(tab.n_sub):
            sel = st == k
            ok[sel] = allowed[k][idx[sel]]
        d[todo[ok]] = dd[ok]
        todo = todo[~ok]
    geo = rng.integers(N_GEO, size=n)
    return geo, s, d


class Encoder:
    """Per-cell inputs and targets for one arm, as torch tensors."""

    def __init__(self, tab: RuleTable, arm: str):
        self.tab, self.arm = tab, arm
        codes = np.zeros((tab.n_sub, 8), np.float32)
        for s in range(tab.n_sub):
            codes[s] = tab.code(s, np.zeros(N_TYPES, np.int64))[:8]
        self.class_bits = torch.from_numpy(codes)
        self.static = torch.from_numpy(tab.static[arm].astype(np.float32))
        self.n_in = self.static.shape[1] + CODE_BITS

    def __call__(self, geo, s, d):
        n = len(geo)
        g = torch.from_numpy(geo)
        sv = torch.from_numpy(s)
        dig = torch.zeros(n, N_TYPES * N_DIGITS)
        dig[torch.arange(n)[:, None], torch.arange(N_TYPES)[None] * N_DIGITS + torch.from_numpy(d)] = 1
        x = torch.cat([self.static[g], self.class_bits[sv], dig], 1)
        t, rot, mb = geo // 12, (geo // 2) % 6, geo % 2
        dt = d[np.arange(n), t]
        bits = self.tab.lut_local[s, t, dt] if self.arm == "e" else self.tab.lut_bits[s, t, dt, rot, mb]
        y = torch.from_numpy(((bits[:, None].astype(np.int64) >> np.arange(15)) & 1).astype(np.float32))
        return x, y


def run_arm(tab, arm, samples, batch, hidden, lr, seed, every, test, test_train):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(tab, arm)
    xt, yt = enc(*test)
    xr, yr = enc(*test_train)
    net = nn.Sequential(nn.Linear(enc.n_in, hidden), nn.ReLU(), nn.Linear(hidden, 15))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    curve, t_train, seen, reach = [], 0.0, 0, {}
    step = 0
    while seen < samples:
        t0 = time.time()
        f = seen / samples
        for g in opt.param_groups:
            g["lr"] = lr * (1.0 if f < 0.6 else 0.3 if f < 0.85 else 0.1)
        x, y = enc(*draws(tab, rng, batch, "train"))
        loss = lossf(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        t_train += time.time() - t0
        seen += batch
        step += 1
        if step % every == 0 or seen >= samples:
            with torch.no_grad():
                p = net(xt) > 0
                tr = float(((net(xr) > 0) == (yr > 0.5)).all(1).float().mean())
            bit = float((p == (yt > 0.5)).float().mean())
            cell = float((p == (yt > 0.5)).all(1).float().mean())
            curve.append({"samples": seen, "sec": round(t_train, 1), "loss": round(float(loss), 5),
                          "bit": round(bit, 5), "cellExact": round(cell, 5), "trainExact": round(tr, 5)})
            for thr in (0.9, 0.99, 0.999, 1.0):
                if cell >= thr and str(thr) not in reach:
                    reach[str(thr)] = {"samples": seen, "sec": round(t_train, 1)}
            if cell == 1.0 and len(curve) >= 2 and curve[-2]["cellExact"] == 1.0:
                break
    with torch.no_grad():
        ok = (net(xt) > 0) == (yt > 0.5)
    cell = ok.all(1).numpy()
    by_sub = {tab.keys[k]: round(float(cell[test[1] == k].mean()), 4) for k in range(tab.n_sub)}
    return {"arm": arm, "hidden": hidden, "inputs": enc.n_in, "params": sum(q.numel() for q in net.parameters()),
            "final": curve[-1], "reach": reach, "bySubset": by_sub, "curve": curve}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--samples", type=int, default=1_900_000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=20, help="held-out check every this many batches")
    ap.add_argument("--test", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=["a", "c", "d", "d-cs", "d-fourier", "e"],
                    choices=["a", "c", "d", "d-cs", "d-fourier", "e"])
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    tab = RuleTable(args.data)
    test = draws(tab, np.random.default_rng(12345), args.test, "heldout")
    test_train = draws(tab, np.random.default_rng(54321), args.test, "train")
    res = []
    for hidden in args.hidden:
        for arm in args.arms:
            r = run_arm(tab, arm, args.samples, args.batch, hidden, args.lr, args.seed, args.every, test, test_train)
            f = r["final"]
            print(f"{arm} h{hidden}: {r['inputs']} inputs, {r['params']} params; held-out after {f['samples']:,} samples "
                  f"({f['sec']} s): bit {f['bit']:.5f}, cell-exact {f['cellExact']:.5f} (train rules "
                  f"{f['trainExact']:.5f}); reached "
                  + ", ".join(f"{k} at {v['samples']:,} ({v['sec']} s)" for k, v in r["reach"].items())
                  + f"; cell-exact by subset {r['bySubset']}", flush=True)
            res.append(r)
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump({"args": vars(args), "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
