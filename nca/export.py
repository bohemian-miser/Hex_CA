"""Checkpoint -> web/nca-weights.json and tests/fixtures/nca-parity.json.

    python -m nca.export runs/p1/ckpt.pt [--note "..."]

Format version 3 (spec v4) for a model with the pooled perception ("taps+pool"):
version 2 plus "perception": "taps+pool" and "w1pool": [H * 2C], index
h*2C + j, j < C the hex max of state channel j, j >= C the hex min of channel
j-C (over the cell's on-board taps among the 7). Version 2 (spec v3) for a
"taps" model with the constant inputs mask, theta1, theta2: "consts" lists them
and w1 has C+3 input channels, index ((h*(C+3)) + c)*9 + k, c < C the state,
then the consts in order (spec v6: 7 of them, mask, theta1, theta2, src1, src1c, src2,
src2c -- "consts" lists whatever the checkpoint has; nothing else changes). A "taps" model with the mask alone (v1/v2
checkpoints) is written as version 1, as before; all three read back
(model_from_json).

Weights are rounded to 6 significant digits. The parity fixture is computed
from the ROUNDED weights, float32, synchronous steps (fireRate 1), R=5, 24
steps, on a picture with one closed loop, one open line and one rim-to-rim
bridge. Then both JSON files are reloaded, the model rebuilt from them, and the
fixture reproduced (a self-check; the TS side must match it to 1e-3).
"""

import argparse
import json
import os

import numpy as np
import torch

from .data import oracle
from .hexgrid import CONST_NAMES, mask as hex_mask, side
from .model import HexNCA, const_stack, fresh_state

PARITY_R = 5
PARITY_STEPS = 24


def r6(x):
    return [float(f"{v:.6g}") for v in np.asarray(x, dtype=np.float64).ravel()]


def weights_json(model, meta):
    """The web format: w1 index ((h*(C+n))+c)*9+k, k=(drow+1)*3+(dcol+1), n consts; w2 index o*H+h;
    w1pool index h*2C+j. Version 3 for a "taps+pool" model (3 consts, or 7 with spec v6's rim sources),
    version 2 with "consts" for a 3-const "taps" model, version 1 (mask only, no "consts") for a 1-const one."""
    pool = model.perception == "taps+pool"
    if model.n_consts not in (1, 3, 7) or (pool and model.n_consts == 1):
        raise ValueError(f"no export format for a {model.perception} model with {model.n_consts} consts")
    w1 = (model.w1 * model.kmask).detach().numpy()  # [H,C+n,3,3], corners exactly zero
    head = {"version": 3 if pool else 2 if model.n_consts == 3 else 1}
    head.update({"channels": model.channels, "hidden": model.hidden})
    if model.n_consts > 1:
        head["consts"] = model.const_names
    if pool:
        head["perception"] = "taps+pool"
    out = {
        **head,
        "clamp": list(model.clamp) if model.clamp is not None else None,
        "fireRate": 1 if model.fire_rate >= 1 else float(model.fire_rate),
        "w1": r6(w1), "b1": r6(model.b1.detach().numpy()),
        "w2": r6(model.w2.detach().numpy()), "b2": r6(model.b2.detach().numpy()),
    }
    if pool:
        out["w1pool"] = r6(model.w1pool.detach().numpy())  # [H,2C,1,1] -> index h*2C + j
    out["meta"] = meta
    return out


def model_from_json(j):
    """Rebuild a float32 HexNCA from the web JSON, version 1, 2 or 3 (fireRate 1: synchronous)."""
    C, H = j["channels"], j["hidden"]
    names = ["mask"] if j["version"] == 1 else list(j["consts"])
    perception = j.get("perception", "taps")
    if (j["version"] not in (1, 2, 3) or names != list(CONST_NAMES[:len(names)])
            or (perception == "taps+pool") != (j["version"] == 3)):
        raise ValueError(f"unknown weights file: version {j['version']}, consts {names}, perception {perception}")
    n = len(names)
    m = HexNCA(C, H, j["clamp"], 1.0, n, perception)
    with torch.no_grad():
        m.w1.copy_(torch.tensor(j["w1"], dtype=torch.float32).view(H, C + n, 3, 3))
        if perception == "taps+pool":
            m.w1pool.copy_(torch.tensor(j["w1pool"], dtype=torch.float32).view(H, 2 * C, 1, 1))
        m.b1.copy_(torch.tensor(j["b1"], dtype=torch.float32))
        m.w2.copy_(torch.tensor(j["w2"], dtype=torch.float32).view(C, H, 1, 1))
        m.b2.copy_(torch.tensor(j["b2"], dtype=torch.float32))
    return m.eval()


def parity_walls(R=PARITY_R):
    """One closed loop (radius-2 ring round (q,r)=(-1,0)), one open straight line (q=3, r=-4..1) and one
    rim-to-rim bridge (the row r = R-1, which cuts the rim row r = R off: 2 rim regions)."""
    S = side(R)
    q = (np.arange(S) - R).reshape(1, S)
    r = (np.arange(S) - R).reshape(S, 1)
    ring = np.maximum(np.maximum(np.abs(q + 1), np.abs(r)), np.abs(q + 1 + r)) == 2
    line = (q == 3) & (r >= -4) & (r <= 1)
    bridge = r == R - 1
    return ((ring | line | bridge) & (hex_mask(R) == 1)).astype(np.uint8)


@torch.no_grad()
def run_parity(model, walls_np, steps):
    """[C*S*S] float32 state after `steps` synchronous steps from the fresh state, channel-major."""
    S = walls_np.shape[0]
    walls = torch.from_numpy(walls_np).float().view(1, 1, S, S)
    cs = const_stack((S - 1) // 2, model.n_consts)
    state = model(fresh_state(walls, model.channels), walls, cs, steps)
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


def load_checkpoint(path, note="", label=None):
    """(model, meta) of a training checkpoint (ckpt.pt or best.pt): the model rebuilt from its config, and
    the meta the web page shows (the default note names the file as `label`, else its path). Reads the
    file, writes nothing."""
    ck = torch.load(path, map_location="cpu", weights_only=False)  # a checkpoint from a GPU loads here too
    cfg = ck["config"]
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"], cfg.get("nConsts", 1),
                   cfg.get("perception", "taps"))  # checkpoints from before v4 have no pool
    model.load_state_dict(ck["model"])
    if cfg.get("consts") and cfg["consts"] != model.const_names:
        raise ValueError(f"checkpoint consts {cfg['consts']} are not this code's {model.const_names}")
    steps = cfg["steps"]
    if isinstance(steps, dict):  # v2 trainer: a range per radius, [a*R, b*R]
        steps = [min(s[0] for s in steps.values()), max(s[1] for s in steps.values())]
    meta = {"trainedR": cfg["R"], "steps": steps, "stepsMult": cfg.get("stepsMult"),
            "iterations": cfg.get("prevIterations", 0) + ck["iteration"],
            "runIteration": ck["iteration"],  # the run's own count (iterations adds the --init chain's)
            "pool": bool(cfg.get("pool", False)),  # edit-trained: the page keeps the state across edits
            "floods": bool(cfg.get("floods", False)),  # spec v6: channels 2..7 are the hand-written floods
            "note": note or f"{'phase 2 (pool)' if cfg.get('pool') else 'phase 1'} from {label or path}"}
    return model, meta


def checkpoint_json(path, note="", label=None):
    """A checkpoint's weights in the web format (weights_json), in memory: what nca.dashboard serves to the
    play page for a run (/weights?run=NAME, --static's <run>/weights.json). Writes nothing."""
    return weights_json(*load_checkpoint(path, note, label))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--weights", default="web/nca-weights.json")
    p.add_argument("--fixture", default="tests/fixtures/nca-parity.json")
    p.add_argument("--note", default="")
    args = p.parse_args()

    model, meta = load_checkpoint(args.ckpt, args.note)
    wj, fix = export(model, meta, args.weights, args.fixture)
    err = self_check(args.weights, args.fixture)
    S = side(fix["R"])
    pred = np.array(fix["state"]).reshape(wj["channels"], S, S)[1] > 0.5
    truth = oracle(np.array(fix["walls"]).reshape(S, S), fix["R"])[0] == 1
    print(f"wrote {args.weights} (version {wj['version']}, consts {wj.get('consts', ['mask'])}, "
          f"perception {wj.get('perception', 'taps')}, "
          f"{os.path.getsize(args.weights)} bytes) and {args.fixture}; "
          f"fixture picture: {int(pred.sum())} cells filled, oracle {int(truth.sum())}, "
          f"{int((pred != truth).sum())} wrong; self-check max |diff| = {err:.2e}")
    assert err < 1e-5, "export self-check failed"


if __name__ == "__main__":
    main()
