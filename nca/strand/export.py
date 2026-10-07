"""Strand checkpoints -> the strand play page's weights JSON (web/strand.html, src/strand-nca.ts), the board and rule
data it bundles, and the TS parity fixtures.

    python -m nca.strand.export runs/tap-e2-l3/best.pt                # -> web/strand-weights.json (the page's default)
    python -m nca.strand.export runs/tap-c-l3/best.pt --out /tmp/w.json
    python -m nca.strand.export --board-data                          # -> web/strand-data.json (rules + boards)
    python -m nca.strand.export --fixtures                            # -> tests/fixtures/strand-*.json

Every strand net is one per-cell update over the 7 hex taps, so one format covers them all:
  arch "frame"  nets.FrameNCA (train2 --inputs e, e-bc; option E): each cell reads its taps and its directional
                channels in its own frame (frame f = rot * 2 + mirror bit, from the cell's geo); `dirIn` lists the
                first channel of every directional group of x = concat(state, consts).
  arch "taps"   nets.StrandNCA (train2 --inputs a, c, c-bc, d, d-cs: the plain hex conv, any depth) and the v1
                HexNCA (nca.strand.train: the rule pre-rendered as 15 chord planes, "inputs": "chords"): the same
                update with every cell in frame 0 (no permutation, `dirIn` empty).
Tensors (float32, little-endian, base64; exact, not rounded):
  w1 [H, 7, Cx]  local tap j (0 = the cell, 1 + k = the neighbour across local direction k; in frame 0 local
                 direction k is board direction k = walker.DIRS[k]), local channel c (c < C: the state, then the
                 consts in `consts` order); b1 [H]; mids: depth - 1 hidden layers {w [H, H] (out, in), b [H]};
                 w2 [C, H]; b2 [C]. One step: pre = b1 + w1 . taps; h = relu(pre); h = relu(w h + b) per mid;
                 d = w2 h + b2 (local output channels, put back to board channels by the cell's frame);
                 state = clamp(state + d); state[0] = the mask; state *= the mask.
consts (per cell, mask first): v2 [mask, static planes (`staticLut`, per geo = type * 12 + rot * 2 + mirror bit),
  (bc: the rule's 53-bit code on every board cell), the tap's code (53) and chord edges (6), on the tapped cell
  only]; v1 [mask, the rule's 15 chord planes on every cell, the tap's chord edges (6)]. `ruleEverywhere` (bc, v1):
  the consts carry one rule on every cell, so several taps can't share a board.
The tap ("tap", from train3's --tap; version 2 files only): {"mode": "held"} (train2, train: version 1 files carry no
  "tap" and are held: every tap on its cell every step), {"mode": "impulse", "steps": n} (a tap's planes are on its
  cell for the n steps after it, then zero) or {"mode": "fixed", "codeChannels": [35 channels]} (no tap planes ever;
  at the tap the cell's state is written: edge channels 1 + d0, 1 + d1 <- max(itself, 1), the code's 35-plane form
  (train3.code35: 8 class bits, then each digit as 3 bits MSB first, +-1) into codeChannels). "speed": the CA steps
  per chord it was trained to grow at. A file with an impulse or fixed tap is version 2, so a page that reads only
  version 1 (and would hold the tap) refuses it.
Output: edge planes at state channels 1-6 (direction d at 1 + d), drawn where > 0.5 (train2.evaluate).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np
import torch

from . import train as V1
from . import train2 as V2
from .nets import DROW, DCOL, FrameNCA
from .rules import CODE_BITS, MAJORS, N_GEO, N_TYPES, STATIC, RuleTable, Boards, chord_planes, default_dir, \
    random_chord
from .walker import walk

FORMAT, VERSION = "hexca-strand", 1
VERSIONS = (1, 2)  # 2: the tap is an event (impulse / fixed)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_WEIGHTS = os.path.join(ROOT, "web", "strand-weights.json")
WEB_DATA = os.path.join(ROOT, "web", "strand-data.json")
FIX_PARITY = os.path.join(ROOT, "tests", "fixtures", "strand-parity.json")
FIX_RULES = os.path.join(ROOT, "tests", "fixtures", "strand-rules.json")
BOARDS = (("l2", "L2", 2), ("l3", "L3", 3), ("l4", "L4full", 4))  # the Delta patch of each level (root 0)


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def unb64(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").reshape(shape).copy()


def is_strand_config(cfg) -> bool:
    return isinstance(cfg, dict) and cfg.get("task") in ("m1a", "m1b")


# ---------------------------------------------------------------- the model

def build(cfg):
    """(torch model, kind 'v2' | 'v1') from a checkpoint's own config (as backfill_gallery does)."""
    if "inputs" in cfg:
        return V2.make_model(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["nIn"], cfg.get("depth", 1),
                             cfg["inputs"], cfg.get("dirGroups")), "v2"
    if cfg.get("perception", "taps") != "taps":
        raise ValueError(f"a v1 strand model with perception {cfg.get('perception')!r}: only 'taps' exports")
    return V1.make_model(cfg["channels"], cfg["hidden"], cfg["clamp"], "taps"), "v1"


def tensors(model):
    """The format's tensors (numpy float32) of a FrameNCA, StrandNCA or v1 HexNCA."""
    with torch.no_grad():
        if isinstance(model, FrameNCA):
            w1 = model.w1.detach().numpy()
            mids = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in model.mids]
            w2 = model.w2.detach().numpy()
        else:
            w = (model.w1 * model.kmask).detach().numpy()  # [H, Cx, 3, 3]
            w1 = np.stack([w[:, :, 1, 1]] + [w[:, :, 1 + DROW[d], 1 + DCOL[d]] for d in range(6)], 1)
            mids = [(m.weight.detach().numpy()[:, :, 0, 0], m.bias.detach().numpy()) for m in getattr(model, "mids", [])]
            w2 = model.w2.detach().numpy()[:, :, 0, 0]
        return {"w1": w1.astype(np.float32), "b1": model.b1.detach().numpy().astype(np.float32),
                "mids": [(a.astype(np.float32), b.astype(np.float32)) for a, b in mids],
                "w2": w2.astype(np.float32), "b2": model.b2.detach().numpy().astype(np.float32)}


def layout(cfg, kind):
    """(arch, inputs, consts segments, staticLut or None, ruleEverywhere)."""
    if kind == "v1":
        return "taps", "chords", [["mask", 1], ["chords", 15], ["tapDirs", 6]], None, True
    inputs = cfg["inputs"]
    lut = static_lut(inputs)
    segs = [["mask", 1], ["static", lut.shape[1]]]
    bc = inputs in V2.BROADCAST
    if bc:
        segs.append(["code", CODE_BITS])
    segs += [["tapCode", CODE_BITS], ["tapDirs", 6]]
    return ("frame" if V2.framed(inputs) else "taps"), inputs, segs, lut, bc


_DATA = [None]


def _data_dir():
    return _DATA[0] or default_dir()


def static_tables(type_majors):
    """rules.RuleTable.static (the per-geo static input planes of every option) from the leaf types' edge classes
    alone, so a checkpoint exports without data/strand-v2 (web/strand-data.json has them); self_check compares."""
    st = {k: np.zeros((N_GEO, n), np.float32) for k, n in STATIC.items()}
    for t in range(N_TYPES):
        for rot in range(6):
            th = 2 * np.pi * rot / 6
            for mb, m in enumerate((1, -1)):
                g = t * 12 + rot * 2 + mb
                for k in st:
                    st[k][g, t] = k != "c"
                st["a"][g, 9 + rot] = 1
                st["a"][g, 15] = mb
                for d in range(6):
                    st["c"][g, d * 8 + MAJORS.index(int(type_majors[t][(m * (d - rot)) % 6]))] = 1
                st["c"][g, 48 + rot] = 1
                st["c"][g, 54] = mb
                st["d"][g, 9:] = rot / 6, mb
                st["d-cs"][g, 9:] = np.cos(th), np.sin(th), mb
                st["d-fourier"][g, 9:] = np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th), np.cos(3 * th), mb
    return st


def static_lut(inputs):
    """float32 [108, planes]: the static planes per geo of an --inputs option."""
    key = V2.static_of(inputs)
    try:
        return RuleTable(_data_dir()).static[key]
    except FileNotFoundError:
        with open(WEB_DATA) as f:
            return static_tables(json.load(f)["typeMajors"])[key]


def held_out_record(log_path, iteration):
    """The quick check at the checkpoint's iteration (the last log record with "q" there), trimmed."""
    if not log_path or not os.path.isfile(log_path):
        return None
    rec = None
    with open(log_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("iteration") == iteration and isinstance(r.get("q"), dict):
                rec = r
    if rec is None:
        return None
    q = rec["q"]
    out = {"iteration": iteration, "exact": q.get("exact"), "balanced": q.get("balanced"), "iou": q.get("iou"),
           "byLevel": q.get("byLevel"), "byLen": {k: v.get("exact") for k, v in (q.get("byLen") or {}).items()}}
    by_set = q.get("bySet") or {}
    out["bySet"] = {k: {"exact": v.get("exact"), "balanced": v.get("balanced")} for k, v in by_set.items()}
    if isinstance(q.get("exit"), dict):
        out["exit"] = q["exit"].get("rate")
    for k, sub in (("persist", "x4"), ("speed", "rate")):  # train3's
        if isinstance(q.get(k), dict):
            out[k] = q[k].get(sub)
    if isinstance(q.get("collide"), dict):
        out["wipe"] = q["collide"].get("wipe")
        out["falseWipe"] = q["collide"].get("falseWipe")
    return out


def tap_spec(cfg):
    """The weights' "tap" from a checkpoint config (train3's --tap; anything older is held)."""
    mode = cfg.get("tap") or "held"
    if mode == "impulse":
        return {"mode": "impulse", "steps": int(cfg.get("tapSteps") or 1)}
    if mode == "fixed":
        return {"mode": "fixed", "codeChannels": [int(c) for c in cfg["fixedChannels"]]}
    return {"mode": "held"}


def strand_json(model, kind, cfg, meta):
    arch, inputs, segs, lut, everywhere = layout(cfg, kind)
    t = tensors(model)
    H, _, Cx = t["w1"].shape
    C = t["w2"].shape[0]
    n_in = sum(n for _, n in segs)
    assert Cx == C + n_in, (Cx, C, n_in)
    enc = lambda a: {"shape": list(a.shape), "data": b64(a)}  # noqa: E731
    tap = tap_spec(cfg)
    out = {"format": FORMAT, "version": 1 if tap["mode"] == "held" else 2, "arch": arch, "kind": kind,
           "task": cfg.get("task", "m1a"),
           "inputs": inputs, "channels": C, "hidden": H, "depth": 1 + len(t["mids"]), "nIn": n_in,
           "clamp": list(cfg["clamp"]) if cfg.get("clamp") is not None else None,
           "dirIn": list(model.dir_in) if isinstance(model, FrameNCA) else [],
           "edges": 1, "consts": segs, "ruleEverywhere": bool(everywhere), "tap": tap,
           "speed": int(cfg.get("speed") or 1),
           "tensors": {"w1": enc(t["w1"]), "b1": enc(t["b1"]),
                       "mids": [{"w": enc(a), "b": enc(b)} for a, b in t["mids"]],
                       "w2": enc(t["w2"]), "b2": enc(t["b2"])},
           "meta": meta}
    if lut is not None:
        out["staticLut"] = {"planes": int(lut.shape[1]), "data": b64(lut)}
    return out


def model_from_json(j):
    """A torch model (FrameNCA / StrandNCA / v1 HexNCA) rebuilt from the weights JSON, float32."""
    if j.get("format") != FORMAT or j.get("version") not in VERSIONS:
        raise ValueError(f"not a {FORMAT} v{'/'.join(map(str, VERSIONS))} file")
    T = j["tensors"]
    w1 = unb64(T["w1"]["data"], T["w1"]["shape"])
    H, _, Cx = w1.shape
    C, n_in = j["channels"], j["nIn"]
    if j["kind"] == "v1":
        model, kind = V1.make_model(C, H, j["clamp"], "taps"), "v1"
    else:
        cfg = {"channels": C, "hidden": H, "clamp": j["clamp"], "nIn": n_in, "depth": j["depth"],
               "inputs": j["inputs"], "dirGroups": (len(j["dirIn"]) - 3) if j["arch"] == "frame" else None}
        model, kind = build(cfg)
    with torch.no_grad():
        if isinstance(model, FrameNCA):
            assert list(model.dir_in) == list(j["dirIn"])
            model.w1.copy_(torch.from_numpy(w1))
            for m, mj in zip(model.mids, T["mids"]):
                m.weight.copy_(torch.from_numpy(unb64(mj["w"]["data"], mj["w"]["shape"])))
                m.bias.copy_(torch.from_numpy(unb64(mj["b"]["data"], mj["b"]["shape"])))
            model.w2.copy_(torch.from_numpy(unb64(T["w2"]["data"], T["w2"]["shape"])))
        else:
            w = torch.zeros(H, Cx, 3, 3)
            w[:, :, 1, 1] = torch.from_numpy(w1[:, 0])
            for d in range(6):
                w[:, :, 1 + DROW[d], 1 + DCOL[d]] = torch.from_numpy(w1[:, 1 + d])
            model.w1.copy_(w)
            for m, mj in zip(getattr(model, "mids", []), T["mids"]):
                m.weight.copy_(torch.from_numpy(unb64(mj["w"]["data"], mj["w"]["shape"]))[:, :, None, None])
                m.bias.copy_(torch.from_numpy(unb64(mj["b"]["data"], mj["b"]["shape"])))
            model.w2.copy_(torch.from_numpy(unb64(T["w2"]["data"], T["w2"]["shape"]))[:, :, None, None])
        model.b1.copy_(torch.from_numpy(unb64(T["b1"]["data"], T["b1"]["shape"])))
        model.b2.copy_(torch.from_numpy(unb64(T["b2"]["data"], T["b2"]["shape"])))
    return model.eval(), kind


def checkpoint_json(path, label=None, log_path=None, note="", ck=None):
    """A strand checkpoint (best.pt / ckpt.pt of train2 or train) as the weights JSON. The held-out numbers come
    from the run's log.jsonl beside it (or log_path): the quick check at the checkpoint's iteration. `ck`: the
    checkpoint already loaded (nca/export.py's checkpoint_json, for the dashboard)."""
    if ck is None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if not is_strand_config(cfg):
        raise ValueError(f"{path}: not a strand checkpoint (config task {cfg.get('task')!r})")
    model, kind = build(cfg)
    model.load_state_dict(ck["model"])
    it = int(ck.get("iteration", 0))
    log_path = log_path or os.path.join(os.path.dirname(os.path.abspath(path)), "log.jsonl")
    label = label or path
    run = label.split("/")[-2] if "/" in label else None
    meta = {"run": run, "file": label, "iteration": it, "prevIterations": int(cfg.get("prevIterations", 0)),
            "levels": cfg.get("levels"), "init": os.path.basename(os.path.dirname(cfg["init"])) if cfg.get("init") else None,
            "best": ck.get("best"), "heldOut": held_out_record(log_path, it),
            "note": note or f"{kind} {cfg.get('inputs', 'chords')} from {label}"}
    return strand_json(model, kind, cfg, meta)


# ---------------------------------------------------------------- consts and rollouts (the fixture's reference)

def code_of(tab, rule):
    return tab.code(int(rule[0]), np.asarray(rule[1:], np.int64)).astype(np.float32)


def build_consts(j, tab, geo, taps, lut=None):
    """float32 [n_in (+1 frame plane for arch frame), H, W]: the consts of `taps` [(rule [s, d0..d8], (row, col,
    d0, d1)), ...] on board geo (int [H, W], -1 off board), as train2.Planes / train.consts_of build them; several
    taps add their tap planes on their own cells."""
    on = geo >= 0
    Hh, Ww = geo.shape
    parts = [on[None].astype(np.float32)]
    for name, n in j["consts"][1:]:
        if name == "static":
            lut = lut if lut is not None else unb64(j["staticLut"]["data"], (N_GEO, j["staticLut"]["planes"]))
            parts.append(np.where(on[..., None], lut[np.where(on, geo, 0)], 0).transpose(2, 0, 1).astype(np.float32))
        elif name == "code":
            assert len(taps) == 1, "a rule-everywhere model takes one tap"
            parts.append(code_of(tab, taps[0][0])[:, None, None] * on[None])
        elif name == "chords":
            assert len(taps) == 1, "a rule-everywhere model takes one tap"
            r = taps[0][0]
            parts.append(chord_planes(tab.render_bits(int(r[0]), np.asarray(r[1:], np.int64), geo)).astype(np.float32))
        elif name == "tapCode":
            p = np.zeros((CODE_BITS, Hh, Ww), np.float32)
            for rule, (r, c, _, _) in taps:
                p[:, r, c] += code_of(tab, rule)
            parts.append(p)
        elif name == "tapDirs":
            p = np.zeros((6, Hh, Ww), np.float32)
            for _, (r, c, d0, d1) in taps:
                p[d0, r, c] += 1
                p[d1, r, c] += 1
            parts.append(p)
        else:
            raise ValueError(name)
    if j["arch"] == "frame":
        parts.append(np.where(on, geo % 12, 0).astype(np.float32)[None])
    return np.concatenate(parts, 0)


def code35_of(tab, rule):
    """train3.code35 of one rule (+-1, float32 [35])."""
    code = tab.code(int(rule[0]), np.asarray(rule[1:], np.int64))
    out = [1.0 if code[k] else -1.0 for k in range(len(MAJORS))]
    for d in np.asarray(rule[1:], np.int64):
        out += [1.0 if (int(d) >> (2 - k)) & 1 else -1.0 for k in range(3)]
    return np.array(out, np.float32)


@torch.no_grad()
def rollout_events(model, j, tab, geo, taps, steps, every=None):
    """rollout() under the weights' tap mode, the taps [(rule, (row, col, d0, d1), at)] happening at age `at`
    (impulse: their planes on in steps at + 1 .. at + steps; held: from at + 1 on; fixed: the state written at age
    at, before step at + 1): the reference for the play page's tap() (tests/strand.test.ts)."""
    spec = j.get("tap") or {"mode": "held"}
    mode = spec["mode"]
    base = build_consts(j, tab, geo, [])
    state = torch.zeros(1, j["channels"], *geo.shape)
    state[:, 0:1] = torch.from_numpy(base[None, :1])
    snaps = {}
    for t in range(steps):
        on = [(r, tp) for r, tp, at in taps if (mode == "held" and at <= t) or
              (mode == "impulse" and at <= t < at + int(spec.get("steps") or 1))]
        if mode == "fixed":
            for r, (row, col, d0, d1), at in taps:
                if at == t:
                    for d in (d0, d1):
                        state[0, 1 + d, row, col] = max(float(state[0, 1 + d, row, col]), 1.0)
                    state[0, spec["codeChannels"], row, col] = torch.from_numpy(code35_of(tab, r))
        cs = torch.from_numpy(build_consts(j, tab, geo, on) if on else base)[None]
        state = model.step(state, cs[:, :1], cs)
        if every and t + 1 in every:
            snaps[t + 1] = state[0].numpy().copy()
    return state[0].numpy().copy(), snaps


@torch.no_grad()
def rollout(model, j, cs, steps, every=None):
    """States [C, H, W] after `steps` steps from the fresh state (and after each step in `every`)."""
    cs_t = torch.from_numpy(cs)[None]
    walls = cs_t[:, :1]
    state = torch.zeros(1, j["channels"], *cs.shape[1:])
    state[:, 0:1] = walls
    snaps = {}
    for t in range(1, steps + 1):
        state = model.step(state, walls, cs_t)
        if every and t in every:
            snaps[t] = state[0].numpy().copy()
    return state[0].numpy().copy(), snaps


def check_consts(j, tab):
    """build_consts = train2.Planes on a square board, one tap (the trainer's own inputs)."""
    if j["kind"] != "v2":
        return 0.0
    bd = Boards(_data_dir())
    geo = V2.pad_geo(bd.board("L2", 0), 12)
    rng = np.random.default_rng(7)
    s, digits = tab.sample(rng)
    tap = random_chord(rng, tab.render_bits(s, digits, geo))
    rule = np.concatenate([[s], digits])
    mine = build_consts(j, tab, geo, [(rule, tap)])
    theirs = V2.Planes(tab, j["inputs"], torch.device("cpu"))(geo[None], V2.Codes(tab)(rule[None]), np.array([tap]))
    return float(np.abs(mine - theirs[0].numpy()).max())


def self_check(model, kind, j, tab):
    """The JSON's model against the checkpoint's: a rollout of each on a level-2 board (must be identical: the
    tensors are exact float32), and build_consts against train2.Planes. Returns (max |diff| state, consts)."""
    bd = Boards(_data_dir())
    geo = bd.board("L2", 0)
    rng = np.random.default_rng(1)
    s, digits = tab.sample(rng)
    rule = np.concatenate([[s], digits])
    tap = random_chord(rng, tab.render_bits(s, digits, geo))
    cs = build_consts(j, tab, geo, [(rule, tap)])
    back, _ = model_from_json(j)
    a, _ = rollout(model, j, cs, 12)
    b, _ = rollout(back, j, cs, 12)
    st = static_tables(tab.type_majors)
    assert all(np.array_equal(st[k], tab.static[k]) for k in STATIC), "static_tables != rules.RuleTable.static"
    return float(np.abs(a - b).max()), check_consts(j, tab)


# ---------------------------------------------------------------- the page's data

def board_data(tab, bd):
    out = {}
    for name, group, level in BOARDS:
        i = 0
        assert int(bd.mirror[group][i]) in (1, -1)
        geo = bd.board(group, i)
        out[name] = {"level": level, "root": tab.leaf_order[0], "h": int(geo.shape[0]), "w": int(geo.shape[1]),
                     "mirror": int(bd.mirror[group][i]), "tiles": int((geo >= 0).sum()),
                     "geo": [int(x) for x in geo.ravel()]}
    return out


def page_data(data_dir=None):
    """web/strand-data.json: the rule table (rules-hex.json, cut to what the page needs) and the three boards."""
    tab, bd = RuleTable(data_dir), Boards(data_dir)
    m = tab.meta
    subsets = []
    for s, sub in enumerate(m["subsets"]):
        subsets.append({"key": sub["key"], "edges": sub["edges"], "count": sub["count"], "options": sub["options"],
                        "localPairs": sub["local_pairs"],
                        "split": {"k": sub["split"]["k"], "thrHash": sub["split"]["thrHash"],
                                  "thrIndex": sub["split"]["thrIndex"],
                                  "v1": bool(tab.is_v1_subset(s))}})
    return {"version": 1,
            "source": f"data/strand-v2 (scripts/strand-export.ts --rule-table, Spectacle {m['spectacle']['commit'][:7]}): "
                      "rules-hex.json and boards.npz (L2 / L3 / L4full, the Delta patch of each level)",
            "conventions": {"dirs": m["conventions"]["dirs_dq_dr"], "pairs": m["conventions"]["pairs"],
                            "geo": "type * 12 + rot * 2 + (mirror < 0), -1 off the board; row = r - r0, col = q - q0",
                            "tile_rot": m["conventions"]["tile_rot"], "split": m["conventions"]["split"]},
            "leafOrder": tab.leaf_order, "majors": list(m["conventions"]["majors"]),
            "typeMajors": m["type_majors"], "subsets": subsets,
            "legacyHeldout": [[s, i] for s, i in tab.legacy_heldout()],
            "boards": board_data(tab, bd)}


# ---------------------------------------------------------------- fixtures

def random_model(kind, inputs, channels, hidden, depth, seed, dir_groups=None, tap=None):
    """A small model with every parameter random (w2 included: a trained-looking update, not the identity); tap:
    a train3 --tap ("impulse:2", "fixed") for its config."""
    torch.manual_seed(seed)
    if kind == "v1":
        model = V1.make_model(channels, hidden, (-2.0, 2.0), "taps")
        cfg = {"task": "m1a", "channels": channels, "hidden": hidden, "clamp": [-2.0, 2.0], "perception": "taps"}
    else:
        n_in = V2.n_inputs(inputs)
        model = V2.make_model(channels, hidden, (-2.0, 2.0), n_in, depth, inputs, dir_groups)
        cfg = {"task": "m1a", "inputs": inputs, "channels": channels, "hidden": hidden, "clamp": [-2.0, 2.0],
               "nIn": n_in, "depth": depth, "dirGroups": dir_groups}
        if tap:
            mode, _, steps = tap.partition(":")
            cfg.update({"tap": mode, "tapSteps": int(steps or 1), "speed": 2,
                        "fixedChannels": list(range(channels - 35, channels)) if mode == "fixed" else None})
    with torch.no_grad():
        for p in model.parameters():
            fan = p[0].numel() if p.dim() > 1 else 4
            p.copy_(torch.randn_like(p) * (1.2 / fan ** 0.5))
    model.eval()
    return model, cfg


def fixture_taps(tab, geo, rng, n, rules=None):
    """n taps on distinct cells: train rules (or `rules`), a random chord each."""
    out, used = [], set()
    while len(out) < n:
        if rules:
            rule = np.asarray(rules[len(out)], np.int64)
        else:
            s, digits = tab.sample(rng)
            rule = np.concatenate([[s], digits])
        tap = random_chord(rng, tab.render_bits(int(rule[0]), rule[1:], geo))
        if tap is None or tap[:2] in used:
            continue
        used.add(tap[:2])
        out.append((rule, tap))
    return out


def parity_fixture(default_weights):
    """tests/fixtures/strand-parity.json: per case a model (the default weights file, or a small random one inline),
    a board, taps and the states after some steps, from torch."""
    tab, bd = RuleTable(_data_dir()), Boards(_data_dir())
    cases = []
    with open(default_weights) as f:
        dj = json.load(f)
    specs = [("default (" + str(dj["meta"].get("run")) + ")", None, "l2", 2, 8),
             ("frame e depth 2", ("v2", "e", 25, 16, 2, 2), "l2", 2, 6),
             ("frame e-bc depth 1", ("v2", "e-bc", 25, 16, 1, 2), "l2", 1, 6),
             ("taps c depth 2", ("v2", "c", 20, 16, 2, None), "l2", 2, 6),
             ("taps a depth 1", ("v2", "a", 16, 12, 1, None), "l2", 3, 5),
             ("v1 chords", ("v1", None, 16, 16, 1, None), "l2", 1, 6),
             ("default on l3", None, "l3", 3, 3),
             ("impulse e depth 2, taps at 0 and 2", ("v2", "e", 25, 16, 2, 2, "impulse:1"), "l2", 2, 6),
             ("impulse 2 steps, taps at 0, 1, 3", ("v2", "e", 25, 16, 2, 1, "impulse:2"), "l2", 3, 6),
             ("fixed write e, taps at 0 and 3", ("v2", "e", 50, 16, 2, 0, "fixed"), "l2", 2, 6)]
    geos = {name: bd.board(group, 0) for name, group, _ in BOARDS}
    for i, (label, spec, board, ntaps, steps) in enumerate(specs):
        rng = np.random.default_rng(100 + i)
        if spec is None:
            j = dj
            model, _ = model_from_json(j)
            inline = None
        else:
            kind, inputs, C, Hd, depth, G = spec[:6]
            model, cfg = random_model(kind, inputs, C, Hd, depth, seed=10 + i, dir_groups=G,
                                      tap=spec[6] if len(spec) > 6 else None)
            j = strand_json(model, kind, cfg, {"note": f"random {label}"})
            inline = j
        geo = geos[board]
        taps = fixture_taps(tab, geo, rng, ntaps)
        at = [0] * len(taps)
        if (j.get("tap") or {}).get("mode", "held") != "held":
            at = [int(x) for x in label.split("taps at ")[1].replace(" and", ",").split(", ")]
            final, snaps = rollout_events(model, j, tab, geo, [(r, t, a) for (r, t), a in zip(taps, at)], steps,
                                          every={1})
        else:
            cs = build_consts(j, tab, geo, taps)
            final, snaps = rollout(model, j, cs, steps, every={1})
        keep = final.shape[0] if board == "l2" else 32  # a big board: the first 32 channels (edges, hidden) only
        case = {"name": label, "weights": inline if inline is not None else "web/strand-weights.json",
                "board": board, "taps": [{"rule": [int(x) for x in r], "tap": [int(x) for x in t], "at": a}
                                         for (r, t), a in zip(taps, at)],
                "steps": steps, "shape": [keep] + list(final.shape[1:]), "state": b64(final[:keep]),
                "drawn": int(((final[1:7] > 0.5) & (geo >= 0)).sum())}
        if board == "l2":
            case["step1"] = b64(snaps[1])
        cases.append(case)
    return {"what": "states [C, H, W] (shape: the first C channels kept) from the fresh state after `steps` steps (and "
                    "after step 1 on l2), torch float32 (nca.strand.export.parity_fixture); weights: the file, or "
                    "inline; a tap's `at`: the step it happens before (event weights; held weights hold every tap "
                    "from step 0)", "cases": cases}


def rules_fixture():
    """tests/fixtures/strand-rules.json: the rule table's split, codes, renderings and walks, from rules.py (itself
    checked against Spectacle: python -m nca.strand.rules --split --parity)."""
    tab, bd = RuleTable(_data_dir()), Boards(_data_dir())
    rng = np.random.default_rng(2026)
    samples = [tab.meta["samples"][k] for k in sorted(rng.choice(len(tab.meta["samples"]), 64, replace=False))]
    held = []
    for s in range(tab.n_sub):
        n = int(tab.count[s])
        idx = np.arange(n) if n <= 64 else np.concatenate([np.nonzero(tab.held_out(s))[0][:20],
                                                           rng.choice(n, 60, replace=False)])
        for i in idx:
            held.append([s, int(i), bool(tab.held_out(s)[int(i)])])
    renders = []
    geo = bd.board("L3", 0)
    for k in range(10):
        s, digits = tab.sample(rng, "train" if k % 2 else "heldout")
        bits = tab.render_bits(s, digits, geo)
        ex = tab.exits(s, digits, geo)
        walks = []
        for _ in range(4):
            tap = random_chord(rng, bits)
            if tap is None:
                continue
            st = walk(ex, geo >= 0, *tap)
            walks.append({"tap": list(tap), "rows": st.rows.tolist(), "cols": st.cols.tolist(), "ins": st.ins.tolist(),
                          "outs": st.outs.tolist(), "index": st.index.tolist(), "closed": bool(st.closed)})
        renders.append({"rule": [int(s)] + [int(d) for d in digits], "describe": tab.describe(s, digits),
                        "key": tab.key(s, digits), "code": [int(x) for x in tab.code(s, digits)],
                        "bits": [int(x) for x in bits.ravel()], "walks": walks})
    return {"what": "rules.py's split (held [s, index, held out]), hash samples, codes, chord bits on the L3 Delta "
                    "patch and walks (nca.strand.export.rules_fixture)", "board": "l3",
            "samples": samples, "held": held, "renders": renders}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", help="a strand checkpoint (best.pt / ckpt.pt)")
    p.add_argument("--out", default=None, help=f"default {os.path.relpath(WEB_WEIGHTS, ROOT)} (or the data / fixture path)")
    p.add_argument("--label", default=None, help="the checkpoint's name in the file (default RUN/FILE from its path)")
    p.add_argument("--note", default="")
    p.add_argument("--data", default=None, help="data/strand-v2 (the rule table and boards)")
    p.add_argument("--board-data", action="store_true", help="write web/strand-data.json")
    p.add_argument("--fixtures", action="store_true", help="write tests/fixtures/strand-parity.json and strand-rules.json")
    p.add_argument("--weights", default=WEB_WEIGHTS, help="--fixtures: the default weights file")
    args = p.parse_args(argv)
    _DATA[0] = args.data
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.board_data:
        out = args.out or WEB_DATA
        write_json(out, page_data(args.data))
        print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    if args.fixtures:
        write_json(FIX_PARITY, parity_fixture(args.weights))
        write_json(FIX_RULES, rules_fixture())
        print(f"wrote {FIX_PARITY} ({os.path.getsize(FIX_PARITY)} bytes) and {FIX_RULES} ({os.path.getsize(FIX_RULES)} bytes)")
    if args.ckpt:
        parts = os.path.abspath(args.ckpt).split(os.sep)
        label = args.label or "/".join(parts[-2:])
        j = checkpoint_json(args.ckpt, label, note=args.note)
        out = args.out or WEB_WEIGHTS
        write_json(out, j)
        model, kind = model_from_json(j)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        orig, _ = build(ck["config"])
        orig.load_state_dict(ck["model"])
        d_state, d_consts = self_check(orig.eval(), kind, j, RuleTable(_data_dir()))
        ho = j["meta"].get("heldOut") or {}
        print(f"wrote {out} ({os.path.getsize(out)} bytes): {j['kind']} {j['arch']} inputs {j['inputs']}, C {j['channels']} "
              f"H {j['hidden']} depth {j['depth']}, consts {j['nIn']}; held-out exact {ho.get('exact')} at iteration "
              f"{j['meta']['iteration']}; self-check |diff| state {d_state:.1e}, consts vs train2.Planes {d_consts:.1e}")
        assert d_state == 0.0 and d_consts == 0.0, "export self-check failed"
    if not (args.ckpt or args.board_data or args.fixtures):
        p.error("nothing to do: a checkpoint, --board-data or --fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
