"""Checkpoint -> web/nca-weights.json and tests/fixtures/nca-parity.json.

    python -m nca.export runs/p1/ckpt.pt [--note "..."]

Weights are rounded to 6 significant digits. The parity fixture is computed
from the ROUNDED weights, float32, synchronous steps (fireRate 1), R=5, 24
steps, on a picture with one closed loop and one open line. Then both JSON
files are reloaded, the model rebuilt from them, and the fixture reproduced
(a self-check; the TS side must match it to 1e-3).
"""

import argparse
import json
import os

import numpy as np
import torch

from .data import oracle
from .hexgrid import mask as hex_mask, side
from .model import HexNCA, fresh_state

PARITY_R = 5
PARITY_STEPS = 24


def r6(x):
    return [float(f"{v:.6g}") for v in np.asarray(x, dtype=np.float64).ravel()]


def weights_json(model, meta):
    """The web format: w1 index ((h*17)+c)*9+k, k=(drow+1)*3+(dcol+1); w2 index o*H+h."""
    w1 = (model.w1 * model.kmask).detach().numpy()  # [H,17,3,3], corners exactly zero
    return {
        "version": 1, "channels": model.channels, "hidden": model.hidden,
        "clamp": list(model.clamp) if model.clamp is not None else None,
        "fireRate": 1 if model.fire_rate >= 1 else float(model.fire_rate),
        "w1": r6(w1), "b1": r6(model.b1.detach().numpy()),
        "w2": r6(model.w2.detach().numpy()), "b2": r6(model.b2.detach().numpy()),
        "meta": meta,
    }


def model_from_json(j):
    """Rebuild a float32 HexNCA from the web JSON (fireRate 1: synchronous)."""
    C, H = j["channels"], j["hidden"]
    m = HexNCA(C, H, j["clamp"], 1.0)
    with torch.no_grad():
        m.w1.copy_(torch.tensor(j["w1"], dtype=torch.float32).view(H, C + 1, 3, 3))
        m.b1.copy_(torch.tensor(j["b1"], dtype=torch.float32))
        m.w2.copy_(torch.tensor(j["w2"], dtype=torch.float32).view(C, H, 1, 1))
        m.b2.copy_(torch.tensor(j["b2"], dtype=torch.float32))
    return m.eval()


def parity_walls(R=PARITY_R):
    """One closed loop (radius-2 ring round (q,r)=(-1,0)) and one open straight line (q=3, r=-4..1)."""
    S = side(R)
    q = (np.arange(S) - R).reshape(1, S)
    r = (np.arange(S) - R).reshape(S, 1)
    ring = np.maximum(np.maximum(np.abs(q + 1), np.abs(r)), np.abs(q + 1 + r)) == 2
    line = (q == 3) & (r >= -4) & (r <= 1)
    return ((ring | line) & (hex_mask(R) == 1)).astype(np.uint8)


@torch.no_grad()
def run_parity(model, walls_np, steps):
    """[C*S*S] float32 state after `steps` synchronous steps from the fresh state, channel-major."""
    S = walls_np.shape[0]
    walls = torch.from_numpy(walls_np).float().view(1, 1, S, S)
    mk = torch.from_numpy(hex_mask((S - 1) // 2)).float().view(1, 1, S, S)
    state = model(fresh_state(walls, model.channels), walls, mk, steps)
    return state[0].numpy().astype(np.float32).ravel()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def export(model, meta, weights_path, fixture_path):
    wj = weights_json(model, meta)
    walls = parity_walls()
    state = run_parity(model_from_json(wj), walls, PARITY_STEPS)
    fix = {"R": PARITY_R, "walls": walls.ravel().astype(int).tolist(), "steps": PARITY_STEPS,
           "state": [float(f"{v:.7g}") for v in state.tolist()]}
    write_json(weights_path, wj)
    write_json(fixture_path, fix)
    return wj, fix


def self_check(weights_path, fixture_path):
    """Reload both files, rebuild the model, reproduce the fixture. Returns the max abs difference."""
    with open(weights_path) as f:
        wj = json.load(f)
    with open(fixture_path) as f:
        fix = json.load(f)
    S = side(fix["R"])
    walls = np.array(fix["walls"], dtype=np.uint8).reshape(S, S)
    got = run_parity(model_from_json(wj), walls, fix["steps"])
    return float(np.abs(got - np.array(fix["state"], dtype=np.float32)).max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--weights", default="web/nca-weights.json")
    p.add_argument("--fixture", default="tests/fixtures/nca-parity.json")
    p.add_argument("--note", default="")
    args = p.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["config"]
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"])
    model.load_state_dict(ck["model"])
    meta = {"trainedR": cfg["R"], "steps": cfg["steps"],
            "iterations": cfg.get("prevIterations", 0) + ck["iteration"],
            "pool": bool(cfg.get("pool", False)),  # edit-trained: the page keeps the state across edits
            "note": args.note or f"{'phase 2 (pool)' if cfg['pool'] else 'phase 1'} from {args.ckpt}"}
    wj, fix = export(model, meta, args.weights, args.fixture)
    err = self_check(args.weights, args.fixture)
    S = side(fix["R"])
    pred = np.array(fix["state"]).reshape(wj["channels"], S, S)[1] > 0.5
    truth = oracle(np.array(fix["walls"]).reshape(S, S), fix["R"])[0] == 1
    print(f"wrote {args.weights} ({os.path.getsize(args.weights)} bytes) and {args.fixture}; "
          f"fixture picture: {int(pred.sum())} cells filled, oracle {int(truth.sum())}, "
          f"{int((pred != truth).sum())} wrong; self-check max |diff| = {err:.2e}")
    assert err < 1e-5, "export self-check failed"


if __name__ == "__main__":
    main()
