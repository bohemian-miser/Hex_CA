"""Local dashboard for watching an NCA training run: the pool and progress, on localhost.

Reads, never writes, two files `nca/train.py` produces under `runs/<name>/`:
  log.jsonl  one JSON object per line: an optional first-line config (no "iteration" key),
             then one record per logged iteration (iteration, loss, secPerIter, lr, time, device,
             and on quick-check iterations nested dicts of numeric metrics -- whatever is there);
             a clean end is a last line {"iteration", "stopped": "time"|"done", ...}.
  pool.npz   a snapshot of the live sample pool, written atomically every --snap-every
             iterations; see the contract at the top of this module's docstring in the repo
             notes. May not exist yet (training not started, or no --pool), may vanish, may
             change under us -- every read here tolerates that and never raises past its caller.

Run:
    python -m nca.dashboard [--port 8765] [--host 127.0.0.1] [--runs runs]
    python -m nca.dashboard --demo   # also (re)writes runs/_demo/{log.jsonl,pool.npz} first,
                                      # a small fabricated run so the page has something to show
                                      # before real training exists. The server never calls this
                                      # on its own -- only --demo does.

Endpoints:
    GET /                 the page (inline CSS + JS, no external resources)
    GET /api/runs         [{name, mtime, lastIteration, hasPool, hasGallery, hasWeights}, ...] newest log.jsonl first
    GET /api/log?run=N    {config, records} -- the parsed log.jsonl, compactly
    GET /api/pool?run=N   the pool snapshot as JSON (see pool_to_json below), or an empty one
                          (200, radii: []) if there isn't one yet / it's unreadable right now
    GET /api/gallery?run=N the gallery snapshot as JSON (see gallery_to_json below, train.write_gallery's
                          contract), or an empty one (200, levels: []) if there isn't one yet
    GET /play             the interactive board, dist/nca.html (npm run build); also /play.html. The page's
                          Play link opens it as /play?weights=<url of the run's weights>&name=<run>
    GET /weights?run=N    the run's best.pt (else ckpt.pt) as the play page's weights JSON, converted by
                          nca/export.py (torch, imported on first use only); also /N/weights.json. 404 JSON
                          without a checkpoint. Cached by the checkpoint's mtime, in memory and under the
                          system temp dir (WEIGHTS_CACHE) -- never in the run directory.

Static copy (no server): python -m nca.dashboard --static OUT_DIR [--runs runs] [--only NAME ...]
[--source pi|vm] writes OUT_DIR/index.html (the same page, with a flag baked in), runs-<source>.json
(what /api/runs returns, each entry also carrying source and exported = unix seconds) and per run
<name>/log.json, <name>/pool.json and <name>/gallery.json (what /api/log, /api/pool and /api/gallery
return). Opened from there the page
fetches those relative files instead of /api/*: it merges runs-pi.json and runs-vm.json (a fixed list,
so two publishers never write the same file and nothing is ever listed), polls every 15 s and shows
the data's own time. nca/cloud/publish.sh uploads such a copy to the bucket. Unless --no-weights, it also
writes play.html (dist/nca.html) and <name>/weights.json for each run with a checkpoint here (the list's
hasWeights); the page's Play link opens play.html?weights=<name>/weights.json&name=<name>, and for a run
whose list says no weights (the VM's) it looks for <name>/weights.json first. --weights-only writes just
play.html, those weights.json and index.html (the page alone, no data): what publish.sh --weights-only
uploads for the cloud runs from the Pi (index.html only with --keep-page).

Stdlib + numpy only (torch only for a run's weights, imported when first asked for). Single process,
single thread per request (ThreadingHTTPServer), no subprocesses, no writes anywhere but the weights
cache in the temp dir (the server; --demo and --static write files once and exit or serve), run names
are checked against the actual directory listing before touching the filesystem with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

# ── hex board geometry (duplicated from nca/hexgrid.py on purpose: this file must not import
#    nca.train / nca.data / nca.model, which pull in torch and are being rewritten elsewhere;
#    the array convention is small and stable enough to keep a second, read-only copy here). ──

NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0), (-1, 1), (1, -1))


def onboard_mask(R: int) -> np.ndarray:
    """bool [S,S], S = 2R+1: True where cell (q=col-R, r=row-R) is on the radius-R board."""
    S = 2 * R + 1
    idx = np.arange(S) - R
    q = idx.reshape(1, S)
    r = idx.reshape(S, 1)
    q, r = np.broadcast_arrays(q, r)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))
    return dist <= R


# ── run discovery & log parsing ─────────────────────────────────────────────────────────────


def valid_run_names(runs_root: Path) -> set[str]:
    """Every directory name directly under runs_root. The only thing a `run=` query is ever
    checked against, so a request can't walk outside runs_root with `..` or an absolute path."""
    try:
        return {p.name for p in runs_root.iterdir() if p.is_dir()}
    except OSError:
        return set()


def tail_last_record(log_path: Path, start_chunk: int = 8192, max_chunk: int = 1 << 20):
    """The last line of log_path that parses as a JSON object with an "iteration" key, read
    from the end of the file so a long log doesn't have to be read in full just to list runs.
    Tolerant of a half-written trailing line (the log is append-only, not atomic per line)."""
    try:
        size = log_path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    read_size = min(start_chunk, size)
    try:
        with open(log_path, "rb") as f:
            while True:
                f.seek(size - read_size)
                data = f.read(read_size)
                lines = data.split(b"\n")
                if size - read_size > 0:
                    lines = lines[1:]  # first line here may be a partial line; drop it
                for raw in reversed(lines):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and "iteration" in obj:
                        return obj
                if read_size >= size or read_size >= max_chunk:
                    return None
                read_size = min(read_size * 4, size, max_chunk)
    except OSError:
        return None


def parse_log(log_path: Path):
    """(config, records). Line 1 is the config iff it has no "iteration" key (older logs, like
    the ones in this repo today, have no config line at all -- every line is a record). Any
    line that fails to parse (most likely a write in progress at the very end) is skipped."""
    config: dict = {}
    records: list = []
    first = True
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    if first and "iteration" not in obj:
                        config = obj
                    else:
                        records.append(obj)
                first = False
    except OSError:
        pass
    return config, records


def api_runs(runs_root: Path) -> list:
    out = []
    try:
        entries = list(runs_root.iterdir())
    except OSError:
        entries = []
    for p in entries:
        if not p.is_dir():
            continue
        log_path = p / "log.jsonl"
        try:
            mtime = log_path.stat().st_mtime
        except OSError:
            continue  # no log.jsonl (or unreadable) -> not a run, per the endpoint's contract
        last = tail_last_record(log_path)
        out.append({
            "name": p.name,
            "mtime": mtime,
            "lastIteration": (last or {}).get("iteration"),
            "hasPool": (p / "pool.npz").is_file(),
            "hasGallery": (p / "gallery.npz").is_file(),
            "hasWeights": run_checkpoint(p) is not None,
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ── pool.npz -> JSON ────────────────────────────────────────────────────────────────────────


EMPTY_POOL = {"iteration": None, "radii": [], "last_R": None, "last_idx": [], "by_radius": {}}


def pool_to_json(pool_path: Path) -> dict:
    """The snapshot contract (nR = len(radii) groups, n = min(poolSize,48) samples shown per
    radius, m = min(n,6) of those carrying a full state) turned into plain JSON, flat per field
    (the client reshapes with S / C) so the payload has far fewer brackets than a nested list.

    A strand run (nca/strand/train.py, train2.py) additionally writes, for every one of the n
    samples: mask_R (the board's real shape -- the walls_R/target_R/fill_R arrays are a small
    board embedded in one corner of a much bigger square, see those modules' write_snapshot),
    tgt_edges_R uint8 [n,6,S,S] and pred_edges_R float16 [n,6,S,S] (the target / predicted edge
    planes, direction d at ch[1+d] of the per-edge state) and tap_R int16 [n,4] (row, col, d0, d1
    in the board's own, un-embedded coordinates; -1 where there is no tap). Older snapshots have
    none of these; the client falls back to the cell-level walls/fill/target."""
    with np.load(pool_path) as z:
        keys = set(z.files)
        iteration = int(z["iteration"])
        radii = [int(x) for x in np.atleast_1d(z["radii"]).tolist()]
        last_R = int(z["last_R"]) if "last_R" in keys else None
        last_idx = [int(x) for x in np.atleast_1d(z["last_idx"]).tolist()] if "last_idx" in keys else []

        by_radius: dict = {}
        for R in radii:
            def get(name):
                k = f"{name}_{R}"
                return z[k] if k in keys else None

            walls, fill, target = get("walls"), get("fill"), get("target")
            if walls is None:
                continue
            n, S, _S2 = walls.shape
            mask = get("mask")
            ntargets = get("ntargets")
            loss = get("loss")
            age = get("age")
            edits = get("edits")
            damage = get("damage")
            state = get("state")
            tgt_edges, pred_edges, tap = get("tgt_edges"), get("pred_edges"), get("tap")
            m = int(state.shape[0]) if state is not None else 0
            C = int(state.shape[1]) if state is not None else 0

            by_radius[str(R)] = {
                "S": int(S),
                "n": int(n),
                "m": m,
                "C": C,
                "walls": walls.astype(np.int16).reshape(-1).tolist(),
                "mask": mask.astype(np.int16).reshape(-1).tolist() if mask is not None else [],
                "fill": np.round(fill.astype(np.float32), 2).reshape(-1).tolist() if fill is not None else [],
                "target": target.astype(np.int16).reshape(-1).tolist() if target is not None else [],
                "ntargets": ntargets.astype(np.int16).reshape(-1).tolist() if ntargets is not None else [1] * n,
                "loss": [round(float(v), 4) for v in (loss.reshape(-1).tolist() if loss is not None else [float("nan")] * n)],
                "age": age.astype(np.int64).reshape(-1).tolist() if age is not None else [0] * n,
                "edits": edits.astype(np.int64).reshape(-1).tolist() if edits is not None else [0] * n,
                "damage": damage.astype(np.int16).reshape(-1).tolist() if damage is not None else [-1] * n,
                "state": np.round(state.astype(np.float32), 3).reshape(-1).tolist() if state is not None else [],
                "tgtEdges": tgt_edges.astype(np.int16).reshape(-1).tolist() if tgt_edges is not None else [],
                "predEdges": np.round(pred_edges.astype(np.float32), 3).reshape(-1).tolist() if pred_edges is not None else [],
                "tap": tap.astype(np.int64).reshape(-1).tolist() if tap is not None else [],
            }

        return {"iteration": iteration, "radii": radii, "last_R": last_R, "last_idx": last_idx, "by_radius": by_radius}


# ── gallery.npz -> JSON (nca/strand/train.py's write_gallery contract) ────────────────────────


EMPTY_GALLERY = {"iteration": None, "levels": [], "by_level": {}}


def gallery_to_json(gallery_path: Path) -> dict:
    """A FIXED set of held-out taps (train.GALLERY_LEVELS x GALLERY_PER_LEVEL), watched over training: per
    level, each item's own S x S board (no radius-hex embedding -- unlike pool.npz, every item here already
    is its own tightly-cropped board), flat per field, as pool_to_json."""
    with np.load(gallery_path) as z:
        keys = set(z.files)
        iteration = int(z["iteration"]) if "iteration" in keys else None
        levels = [int(x) for x in np.atleast_1d(z["levels"]).tolist()] if "levels" in keys else []

        by_level: dict = {}
        for L in levels:
            def get(name):
                k = f"{name}_{L}"
                return z[k] if k in keys else None

            mask = get("mask")
            if mask is None:
                continue
            n, S, _S2 = mask.shape
            rot, tap = get("rot"), get("tap")
            tgt, pred = get("tgt_edges"), get("pred_edges")
            rule, length, ideal = get("rule"), get("length"), get("ideal")

            by_level[str(L)] = {
                "S": int(S),
                "n": int(n),
                "mask": mask.astype(np.int16).reshape(-1).tolist(),
                "rot": rot.astype(np.int16).reshape(-1).tolist() if rot is not None else [],
                "tap": tap.astype(np.int64).reshape(-1).tolist() if tap is not None else [],
                "tgtEdges": tgt.astype(np.int16).reshape(-1).tolist() if tgt is not None else [],
                "predEdges": pred.astype(np.int16).reshape(-1).tolist() if pred is not None else [],
                "rule": [str(x) for x in rule.tolist()] if rule is not None else [""] * n,
                "length": length.astype(np.int64).reshape(-1).tolist() if length is not None else [0] * n,
                "ideal": ideal.astype(np.int64).reshape(-1).tolist() if ideal is not None else [0] * n,
            }

        return {"iteration": iteration, "levels": levels, "by_level": by_level}


def sanitize(obj):
    """NaN / inf aren't valid JSON; turn them into null so allow_nan=False can enforce that
    nothing invalid slips out (a NaN loss -- "never scored yet" -- is explicitly in the contract)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


# ── the play page and a run's weights ──────────────────────────────────────────────────────────

PLAY_HTML = Path(__file__).resolve().parent.parent / "dist" / "nca.html"  # npm run build (scripts/build-web.ts)
CHECKPOINTS = ("best.pt", "ckpt.pt")  # a run's weights: its best model, else its latest
WEIGHTS_CACHE = Path(tempfile.gettempdir()) / "hexca-play-weights"
_weights_lock = threading.Lock()
_weights_mem: dict = {}  # checkpoint path -> (path:mtime:size, JSON bytes)


def run_checkpoint(run_dir: Path):
    """The checkpoint the play page gets for a run: best.pt, else ckpt.pt, else None."""
    for f in CHECKPOINTS:
        if (run_dir / f).is_file():
            return run_dir / f
    return None


def run_weights(run_dir: Path):
    """The play page's weights JSON (bytes) of the run's checkpoint, converted by nca/export.py's
    checkpoint_json; None without a checkpoint. Cached by the file's path, mtime and size, in memory and
    under WEIGHTS_CACHE (so a publisher's next process doesn't import torch again for an unchanged file;
    older entries of the same checkpoint are dropped). Nothing is written into the run directory. Raises
    if the checkpoint can't be read or converted (caught mid-copy, or a format export.py doesn't know)."""
    ck = run_checkpoint(run_dir)
    if ck is None:
        return None
    ck = ck.resolve()
    st = ck.stat()
    key = f"{ck}:{st.st_mtime_ns}:{st.st_size}:meta3"  # meta3: runIteration, the note names <run>/<file>
    with _weights_lock:
        hit = _weights_mem.get(str(ck))
        if hit and hit[0] == key:
            return hit[1]
        stem = hashlib.sha256(str(ck).encode()).hexdigest()[:16]
        disk = WEIGHTS_CACHE / f"{stem}-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
        try:
            body = disk.read_bytes()
        except OSError:
            from .export import checkpoint_json  # torch: only here, so the dashboard runs without it
            wj = checkpoint_json(str(ck), label=f"{run_dir.name}/{ck.name}")  # no local path in a public file
            body = json.dumps(wj, separators=(",", ":"), allow_nan=False).encode("utf-8")
            try:
                WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
                for old in WEIGHTS_CACHE.glob(f"{stem}-*.json"):
                    old.unlink(missing_ok=True)
                tmp = disk.with_name(f"{disk.name}.{os.getpid()}.tmp")
                tmp.write_bytes(body)
                tmp.replace(disk)
            except OSError:
                pass  # the cache is a convenience
        _weights_mem[str(ck)] = (key, body)
        return body


def export_play(out_dir: Path, runs_root: Path, names) -> set:
    """play.html (a copy of dist/nca.html) and <name>/weights.json for each of `names` with a checkpoint,
    into out_dir. Returns the names whose weights.json it wrote. Without a built play page it writes
    nothing (weights alone can't be played). A run whose checkpoint can't be converted is skipped, said
    on stderr."""
    try:
        play = PLAY_HTML.read_bytes()
    except OSError:
        print(f"no {PLAY_HTML} (npm run build): no play page or weights written", file=sys.stderr)
        return set()
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for name in names:
        try:
            body = run_weights(runs_root / name)
        except Exception as exc:  # noqa: BLE001 -- one bad checkpoint must not stop the rest
            print(f"{name}: no weights.json ({exc!r})", file=sys.stderr)
            continue
        if body is None:
            continue
        d = out_dir / name
        d.mkdir(exist_ok=True)
        tmp = d / "weights.json.tmp"
        tmp.write_bytes(body)
        tmp.replace(d / "weights.json")
        done.add(name)
    tmp = out_dir / "play.html.tmp"
    tmp.write_bytes(play)
    tmp.replace(out_dir / "play.html")
    return done


# ── demo data: fabricates a contract-conforming runs/_demo/{log.jsonl,pool.npz}. Only called
#    from the --demo flag (see main()); nothing on the normal serving path ever calls it. ──


def _simple_targets(walls: np.ndarray, R: int) -> np.ndarray:
    """A stand-in oracle, not the real fill rule (that lives in src/fill.ts / nca/data.py,
    both out of scope here): flood from the rim through non-wall cells: unreached non-wall
    cells are "enclosed" -> target 1. Good enough to make a demo pool look plausible."""
    S = 2 * R + 1
    mask = onboard_mask(R)
    reached = np.zeros((S, S), dtype=bool)
    stack = []
    for row in range(S):
        for col in range(S):
            if not mask[row, col] or walls[row, col]:
                continue
            q, r = col - R, row - R
            if max(abs(q), abs(r), abs(q + r)) == R:
                reached[row, col] = True
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for drow, dcol in NEIGHBOURS:
            nr, nc = row + drow, col + dcol
            if 0 <= nr < S and 0 <= nc < S and mask[nr, nc] and not walls[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    return (mask & ~walls.astype(bool) & ~reached).astype(np.uint8)


def _demo_pool_arrays(rng: np.random.Generator, R: int, n: int, m: int):
    S = 2 * R + 1
    mask = onboard_mask(R)
    walls = np.zeros((n, S, S), dtype=np.uint8)
    fill = np.zeros((n, S, S), dtype=np.float16)
    target = np.zeros((n, S, S), dtype=np.uint8)
    ntargets = np.ones((n,), dtype=np.uint8)
    loss = np.zeros((n,), dtype=np.float32)
    age = rng.integers(0, 600, size=n).astype(np.int32)
    edits = rng.integers(0, 9, size=n).astype(np.int32)
    damage = rng.choice([-1, 0, 0, 1, 1, 2, 3, 4, 5], size=n).astype(np.int8)

    qidx0 = (np.arange(S) - R).reshape(1, S)
    ridx0 = (np.arange(S) - R).reshape(S, 1)
    qg, rg = np.broadcast_arrays(qidx0, ridx0)

    for i in range(n):
        density = rng.uniform(0.05, 0.2)
        w = (rng.random((S, S)) < density).astype(np.uint8) * mask
        if R >= 2 and rng.random() < 0.6:
            # plain scattered walls almost never enclose anything; OR in a ring (like the
            # project's own "stamp" damage) round a random centre so there's usually something
            # for the flood-fill target below to actually mark as filled.
            ring_r = int(rng.integers(1, max(2, R)))
            cq = int(rng.integers(-R + ring_r, R - ring_r + 1))
            cr = int(rng.integers(-R + ring_r, R - ring_r + 1))
            dq, dr = qg - cq, rg - cr
            dist = np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(dq + dr))
            w = (w | ((dist == ring_r) & mask)).astype(np.uint8)
        t = _simple_targets(w, R)
        if rng.random() < 0.15:
            ntargets[i] = 2
        noise = rng.normal(0, 0.18, size=(S, S))
        f = np.clip(t.astype(np.float32) + noise, 0.0, 1.0) * mask
        walls[i] = w
        target[i] = t
        fill[i] = f.astype(np.float16)
        if i < 2:
            loss[i] = np.nan  # "never scored yet"
        else:
            diff = (f - t.astype(np.float32)) * mask
            denom = max(1, int(mask.sum()))
            loss[i] = float((diff * diff).sum() / denom)

    C = 16
    state = np.zeros((m, C, S, S), dtype=np.float16)
    qidx = (np.arange(S) - R).reshape(1, S)
    ridx = (np.arange(S) - R).reshape(S, 1)
    qf, rf = np.broadcast_arrays(qidx.astype(np.float32), ridx.astype(np.float32))
    for i in range(m):
        state[i, 0] = walls[i].astype(np.float16)
        state[i, 1] = fill[i]
        for c in range(2, C):
            phase = c * 0.7
            field = np.sin(qf * 0.5 + phase) + np.cos(rf * 0.4 - phase) + rng.normal(0, 0.3, size=(S, S))
            state[i, c] = (field.astype(np.float32) * mask).astype(np.float16)
    return walls, fill, target, ntargets, loss, age, edits, damage, state


def build_demo(run_dir: Path, iters_total: int = 3000, last_iteration: int = 1450) -> None:
    """Writes runs/_demo/log.jsonl and runs/_demo/pool.npz, contract-conforming fakes, so the
    dashboard has something to render and can be checked before any real run exists."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    radii = [4, 5, 6]

    config = {
        "name": "_demo", "R": radii, "iters": iters_total, "batch": 16, "lr": 0.002, "seed": 0,
        "hidden": 96, "channels": 16, "perception": "taps+pool", "pool": True, "poolSize": 256,
        "damage": 0.5, "evalMults": [8, 16], "evalEvery": 200, "snapEvery": 25, "demo": True,
    }
    lines = [json.dumps(config)]
    lr0 = config["lr"]
    it = 50
    while it <= last_iteration:
        progress = it / iters_total
        loss = 0.8 * math.exp(-it / 400.0) + 0.01 + max(0.0, rng.normal(0, 0.01))
        sec_per_iter = 2.0 + rng.normal(0, 0.08)
        lr = lr0 * max(0.1, 1.0 - progress)
        rec = {
            "iteration": it,
            "loss": round(loss, 6),
            "secPerIter": round(float(sec_per_iter), 4),
            "lr": round(float(lr), 6),
            "maxRssMB": int(600 + it * 0.12 + rng.normal(0, 4)),
        }
        if it == 300:
            rec["skipped"] = "non-finite grad"
        if it == 500:
            rec["steer"] = {"reason": "pool below band", "toward": "fresh"}
        if it in (850, 1100):
            rec["lossFill"] = round(loss * 0.7, 6)
            rec["lossAux"] = round(loss * 0.3, 6)
        if it % 200 == 0:
            learn = min(1.0, progress * 1.6)
            jitter = lambda s=0.05: max(0.0, min(1.0, rng.normal(0, s)))
            rec["q8"] = {
                "none": round(max(0.0, 1.0 - learn + jitter()), 4),
                "both": round(max(0.0, 0.2 * (1 - learn) + jitter()), 4),
                "mix": round(min(1.0, learn + jitter()), 4),
                "bridge": round(min(1.0, 0.7 * learn + jitter()), 4),
                "page": round(min(1.0, learn * 1.1 + jitter()), 4),
                "gap": round(min(1.0, 0.5 * learn + jitter(0.03)), 4),
                "bridgeByR": {str(r): round(min(1.0, 0.6 * learn + jitter(0.04)), 4) for r in radii},
            }
            rec["evalSec"] = round(float(18 + rng.normal(0, 2)), 2)
            rec["best"] = round(min(1.0, 0.5 * learn + jitter(0.02)), 4)
            rec["poolStats"] = {
                "filled": round(min(1.0, 0.3 + 0.5 * learn + jitter(0.03)), 4),
                "multiRegion": round(max(0.0, 0.15 - 0.1 * learn + jitter(0.02)), 4),
                "wallDensity": round(float(0.28 + rng.normal(0, 0.02)), 4),
            }
        lines.append(json.dumps(rec))
        it += 50

    (run_dir / "log.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n, m = 48, 6
    payload = {"iteration": np.int64(last_iteration), "radii": np.array(radii, dtype=np.int64)}
    last_R = radii[len(radii) // 2]
    batch = sorted(rng.choice(48, size=8, replace=False).tolist() + rng.choice(np.arange(48, 256), size=8, replace=False).tolist())
    payload["last_R"] = np.int64(last_R)
    payload["last_idx"] = np.array(batch, dtype=np.int64)
    for R in radii:
        walls, fill, target, ntargets, loss, age, edits, damage, state = _demo_pool_arrays(rng, R, n, m)
        payload[f"walls_{R}"] = walls
        payload[f"fill_{R}"] = fill
        payload[f"target_{R}"] = target
        payload[f"ntargets_{R}"] = ntargets
        payload[f"loss_{R}"] = loss
        payload[f"age_{R}"] = age
        payload[f"edits_{R}"] = edits
        payload[f"damage_{R}"] = damage
        payload[f"state_{R}"] = state

    # np.savez_compressed appends ".npz" to a path that doesn't already end with it, so the
    # temp name has to end in ".npz" itself for the atomic rename to land on the right file.
    tmp = run_dir / "pool.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(run_dir / "pool.npz")


# ── static copy: the page plus the JSON the API would serve, for a plain file host ───────────────

STATIC_SOURCES = ("pi", "vm")  # runs-<source>.json, the fixed list the static page merges (its RUN_LISTS)
STATIC_FLAG = "var STATIC = false; /*STATIC*/"  # in PAGE_HTML; the static copy makes it true


def _write_json(path: Path, obj) -> None:
    """Atomically (a temp file, then a rename), the way the API serialises: NaN -> null, compact."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sanitize(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def export_static(out_dir: Path, runs_root: Path, only=None, source: str = "pi", weights: bool = True) -> list:
    """Write the static copy (see the module docstring) of the runs under runs_root (only those named in
    `only`, if given) into out_dir; with `weights`, play.html and their weights.json too (export_play).
    Returns the runs-<source>.json entries."""
    if source not in STATIC_SOURCES:
        raise ValueError(f"--source must be one of {STATIC_SOURCES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [r for r in api_runs(runs_root) if not only or r["name"] in only]
    played = export_play(out_dir, runs_root, [r["name"] for r in runs if r["hasWeights"]]) if weights else set()
    for r in runs:
        r["hasWeights"] = r["name"] in played  # in this copy: a weights.json next to it
    now = round(time.time(), 1)
    for r in runs:
        r["source"], r["exported"] = source, now
        d = out_dir / r["name"]
        d.mkdir(exist_ok=True)
        config, records = parse_log(runs_root / r["name"] / "log.jsonl")
        _write_json(d / "log.json", {"config": config, "records": records})
        pool_path = runs_root / r["name"] / "pool.npz"
        try:
            pool = pool_to_json(pool_path) if pool_path.is_file() else EMPTY_POOL
        except Exception:  # caught mid-replace or unreadable: an empty one, as the API does
            pool = EMPTY_POOL
        _write_json(d / "pool.json", pool)
        gallery_path = runs_root / r["name"] / "gallery.npz"
        try:
            gallery = gallery_to_json(gallery_path) if gallery_path.is_file() else EMPTY_GALLERY
        except Exception:  # caught mid-replace or unreadable: an empty one, as the API does
            gallery = EMPTY_GALLERY
        _write_json(d / "gallery.json", gallery)
    _write_json(out_dir / f"runs-{source}.json", runs)  # after the runs' files, so it never names a missing one
    write_static_page(out_dir)
    return runs


def write_static_page(out_dir: Path) -> None:
    """out_dir/index.html: the page with STATIC baked in (code only, no data)."""
    page = PAGE_HTML.replace(STATIC_FLAG, "var STATIC = true;")
    assert page != PAGE_HTML, "the static flag is missing from the page"
    tmp = out_dir / "index.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out_dir / "index.html")


# ── HTTP ────────────────────────────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "nca-dashboard/1"
    protocol_version = "HTTP/1.1"

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(sanitize(obj), separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str, status: int = 200) -> None:
        self._bytes(text.encode("utf-8"), "text/html; charset=utf-8", status)

    def _bytes(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _weights(self, name: str, runs_root: Path) -> None:
        """/weights?run=NAME and /NAME/weights.json: the run's checkpoint as the play page's weights."""
        if name not in valid_run_names(runs_root):
            self._json({"error": "unknown run"}, 404)
            return
        try:
            body = run_weights(runs_root / name)
        except Exception as exc:  # noqa: BLE001 -- caught mid-copy, or a format export.py doesn't know
            self._json({"error": f"could not convert {name}'s checkpoint: {exc!r}"}, 500)
            return
        if body is None:
            self._json({"error": f"no checkpoint ({' or '.join(CHECKPOINTS)}) in run {name}"}, 404)
            return
        self._bytes(body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            runs_root: Path = self.server.runs_root  # type: ignore[attr-defined]

            if path in ("/", "/index.html"):
                self._html(PAGE_HTML)
            elif path in ("/play", "/play.html"):
                try:
                    self._bytes(PLAY_HTML.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    self._html(f"<!doctype html><meta charset=utf-8><title>Play page not built</title>"
                               f"<p>The play page isn't built: no <code>{PLAY_HTML}</code>. Run <code>npm run "
                               f"build</code> in <code>{PLAY_HTML.parent.parent}</code>, then reload.</p>", 404)
            elif path == "/weights":
                self._weights((qs.get("run") or [""])[0], runs_root)
            elif re.fullmatch(r"/[^/]+/weights\.json", path):
                self._weights(unquote(path.split("/")[1]), runs_root)
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/api/runs":
                self._json(api_runs(runs_root))
            elif path == "/api/log":
                name = (qs.get("run") or [""])[0]
                if name not in valid_run_names(runs_root):
                    self._json({"error": "unknown run"}, 404)
                    return
                log_path = runs_root / name / "log.jsonl"
                if not log_path.is_file():
                    self._json({"error": "no log"}, 404)
                    return
                config, records = parse_log(log_path)
                self._json({"config": config, "records": records})
            elif path == "/api/pool":
                name = (qs.get("run") or [""])[0]
                if name not in valid_run_names(runs_root):
                    self._json({"error": "unknown run"}, 404)
                    return
                pool_path = runs_root / name / "pool.npz"
                if not pool_path.is_file():
                    # No snapshot yet (no --pool, or training hasn't reached --snap-every): an
                    # ordinary, expected state, not an error -- answer 200 with an empty-shaped
                    # snapshot so the page (and the browser's network log) stay quiet, per the
                    # contract's "return 404/empty" -- this picks "empty".
                    self._json(EMPTY_POOL)
                    return
                try:
                    data = pool_to_json(pool_path)
                except Exception:
                    # half-written (caught mid os.replace) or otherwise unreadable: never crash,
                    # just say so -- the writer will replace it with a good one shortly.
                    self._json(EMPTY_POOL)
                    return
                self._json(data)
            elif path == "/api/gallery":
                name = (qs.get("run") or [""])[0]
                if name not in valid_run_names(runs_root):
                    self._json({"error": "unknown run"}, 404)
                    return
                gallery_path = runs_root / name / "gallery.npz"
                if not gallery_path.is_file():
                    self._json(EMPTY_GALLERY)  # no gallery yet (older run, or before the first quick check)
                    return
                try:
                    data = gallery_to_json(gallery_path)
                except Exception:
                    self._json(EMPTY_GALLERY)  # half-written: the writer will replace it shortly
                    return
                self._json(data)
            else:
                self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # the loop (and the page) must survive any one bad request
            try:
                self._json({"error": f"internal error: {exc}"}, 500)
            except Exception:
                pass

    def log_message(self, fmt: str, *args) -> None:  # quiet: polled every ~3s by design
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Local dashboard for watching an NCA training run.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", default="runs", help="directory holding runs/<name>/{log.jsonl,pool.npz}")
    ap.add_argument("--demo", action="store_true", help="(re)write runs/_demo/ with fabricated data first")
    ap.add_argument("--static", metavar="OUT_DIR", help="write a static copy of the page and its data to OUT_DIR "
                    "and exit (no server); see the module docstring")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="--static: just these runs")
    ap.add_argument("--source", default="pi", choices=STATIC_SOURCES,
                    help="--static: which publisher this is (writes runs-<source>.json)")
    ap.add_argument("--no-weights", action="store_true",
                    help="--static: no play.html or <run>/weights.json (the VM: its CPU is for training)")
    ap.add_argument("--weights-only", action="store_true",
                    help="--static: only play.html, index.html (no data) and <run>/weights.json of the --only runs")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.demo:
        build_demo(runs_root / "_demo")
        print(f"wrote demo run to {runs_root / '_demo'}", file=sys.stderr)

    if args.static and args.weights_only:
        names = sorted(n for n in (args.only or []) if n in valid_run_names(runs_root))
        done = export_play(Path(args.static), runs_root, names)
        write_static_page(Path(args.static))
        print(f"wrote {args.static}: play.html, index.html and weights.json of {sorted(done)}"
              + (f"; none for {sorted(set(args.only or []) - done)}" if set(args.only or []) - done else ""),
              file=sys.stderr)
        return

    if args.static:
        runs = export_static(Path(args.static), runs_root, args.only, args.source, not args.no_weights)
        missing = sorted(set(args.only or []) - {r["name"] for r in runs})
        print(f"wrote {args.static}: index.html, runs-{args.source}.json, {len(runs)} run(s)"
              + (f"; no log.jsonl for {missing}" if missing else ""), file=sys.stderr)
        return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runs_root = runs_root  # type: ignore[attr-defined]
    print(f"nca dashboard: http://{args.host}:{args.port}/  (runs: {runs_root.resolve()})", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ── the page ────────────────────────────────────────────────────────────────────────────────

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>NCA training</title>
<style>
:root {
  --bg: #eef1ee; --panel: #f8faf8; --fg: #1d2421; --muted: #5d6a64; --hair: #cdd6d1;
  --accent: #0f7a66; --wall: #2c3631; --empty: #e4e9e4; --bad: #c0392b; --good: #1f8a4c;
  --amber: #b8860b; --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --body: -apple-system, system-ui, "Segoe UI", sans-serif;
  /* board/small-multiple thumbnail size: the "size" control (next to sort) swaps these via JS,
     default here is "m" so the first paint (before JS runs) already matches it -- no flash. */
  --board-w: 224px; --board-h: 172px; --detail-min: 192px; --detail-h: 172px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12302c0d; --bg: #131816; --panel: #1a201d; --fg: #e3ebe6; --muted: #93a39b;
    --hair: #2c3631; --accent: #4fd1b0; --wall: #0b0e0c; --empty: #26312b; --bad: #ef7a64;
    --good: #5cc98a; --amber: #e0b23d;
  }
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--fg); margin: 0; }
body { font: 14px/1.45 var(--body); padding: 14px 16px 40px; }
h1 { font-size: 18px; margin: 0 0 2px; }
a { color: var(--accent); }
.hdr { display: grid; gap: 8px; margin-bottom: 14px; }
.hdr-row { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }
select, button { font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--hair);
  border-radius: 4px; padding: 5px 9px; cursor: pointer; }
a.play { font: 600 13px/1.2 var(--body); text-decoration: none; color: var(--bg); background: var(--accent);
  border: 1px solid var(--accent); border-radius: 4px; padding: 6px 12px; white-space: nowrap; }
a.play:hover { filter: brightness(1.1); }
a.play:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
a.play.off { color: var(--muted); background: var(--panel); border-color: var(--hair); cursor: not-allowed; filter: none; }
.stat { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.stat .k { font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.stat .v { font: 13px/1.2 var(--mono); }
.progress { flex: 1 1 180px; min-width: 140px; height: 9px; background: var(--empty); border-radius: 5px;
  overflow: hidden; border: 1px solid var(--hair); }
.progress > i { display: block; height: 100%; background: var(--accent); }
.badge { font: 12px var(--mono); padding: 2px 7px; border-radius: 999px; border: 1px solid currentColor; }
.badge.ok { color: var(--good); } .badge.amber { color: var(--amber); } .badge.bad { color: var(--bad); }
details.cfg { font-size: 12px; color: var(--muted); }
details.cfg summary { cursor: pointer; }
details.cfg pre { font: 12px var(--mono); white-space: pre-wrap; word-break: break-word; margin: 6px 0 0;
  color: var(--fg); max-height: 220px; overflow: auto; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-bottom: 18px; }
.chart-box { background: var(--panel); border: 1px solid var(--hair); border-radius: 6px; padding: 8px 10px; min-width: 0; }
.chart-box h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
  margin: 0 0 4px; font-weight: 600; }
.chart-box h2 .showAll { float: right; text-transform: none; letter-spacing: normal; font-weight: 400;
  font-size: 11px; cursor: pointer; }
.chart-box canvas { display: block; width: 100%; height: 150px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 10px; margin-top: 6px; font-size: 11px; }
.legend .grp { width: 100%; font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
  margin-top: 4px; }
.legend .grp:first-child { margin-top: 0; }
.legend button { display: inline-flex; align-items: center; gap: 5px; border: none; background: none; padding: 1px 3px;
  cursor: pointer; color: var(--fg); font: 11px var(--mono); border-radius: 3px; }
.legend button.off { opacity: .35; text-decoration: line-through; }
.legend button .sw { width: 9px; height: 9px; border-radius: 2px; flex: none; }
.tooltip { position: fixed; pointer-events: none; background: var(--panel); border: 1px solid var(--hair);
  border-radius: 4px; padding: 4px 7px; font: 11px var(--mono); z-index: 50; white-space: pre; box-shadow: 0 2px 8px #0003; }
.radius-section { margin-bottom: 20px; }
.radius-hdr { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 16px; margin-bottom: 6px; }
.radius-hdr h2 { font-size: 14px; margin: 0; }
.summary { font: 12px var(--mono); color: var(--muted); }
.controls { display: flex; gap: 10px; align-items: center; margin: 4px 0 10px; flex-wrap: wrap; font-size: 12px; }
.controls label { display: flex; align-items: center; gap: 5px; color: var(--muted); }
.grid { display: flex; flex-wrap: wrap; gap: 6px; }
.board-wrap { background: var(--panel); border: 1px solid var(--hair); border-radius: 5px; padding: 4px;
  width: var(--board-w); max-width: 100%; }
.board-wrap.highlight { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.board-wrap.clickable canvas { cursor: pointer; }
.board-wrap canvas { display: block; width: 100%; height: var(--board-h); background: var(--empty); border-radius: 3px; }
.cap { font: 10px/1.3 var(--mono); color: var(--muted); margin-top: 3px; display: flex; justify-content: space-between; gap: 4px; }
.cap .dmg { padding: 0 4px; border-radius: 3px; background: var(--hair); color: var(--fg); }
.cap .state-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); display: inline-block; }
.empty-note { color: var(--muted); font-size: 12px; padding: 8px 0; }
.gallery-section { margin-bottom: 20px; }
.gallery-hdr { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 14px; margin-bottom: 6px; }
.gallery-hdr h2 { font-size: 14px; margin: 0; }
.gallery-note { font: 11px var(--mono); color: var(--muted); }
.gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; margin-bottom: 10px; }
.gallery-tile { background: var(--panel); border: 1px solid var(--hair); border-radius: 6px; padding: 6px; }
.gallery-tile canvas { display: block; width: 100%; height: 230px; background: var(--empty); border-radius: 4px; }
.gallery-cap { font: 11px/1.4 var(--mono); color: var(--muted); margin-top: 4px; display: flex; justify-content: space-between; gap: 6px; }
.gallery-cap .ok { color: var(--good); font-weight: 700; }
.gallery-legend { font: 11px var(--mono); color: var(--muted); margin-bottom: 8px; }
.gallery-legend .good { color: var(--good); } .gallery-legend .bad { color: var(--bad); }
.overlay { position: fixed; inset: 0; background: #0008; display: flex; align-items: center; justify-content: center;
  z-index: 100; padding: 20px; }
.overlay[hidden] { display: none; }
.detail-panel { background: var(--panel); border: 1px solid var(--hair); border-radius: 8px; padding: 14px;
  max-width: 920px; max-height: 90vh; overflow: auto; position: relative; }
.detail-panel h3 { margin: 0 0 8px; font-size: 14px; }
.detail-panel button.close { position: absolute; top: 8px; right: 10px; border: none; background: none;
  font-size: 18px; cursor: pointer; color: var(--muted); }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--detail-min), 1fr)); gap: 8px; }
.detail-cell { text-align: center; }
.detail-cell canvas { width: 100%; height: var(--detail-h); background: var(--empty); border-radius: 3px; }
.detail-cell .lbl { font: 10px var(--mono); color: var(--muted); margin-top: 2px; }
footer { color: var(--muted); font-size: 11px; margin-top: 20px; }
</style>
</head>
<body>
<div class="hdr">
  <h1>NCA training</h1>
  <div class="hdr-row">
    <select id="runSelect"><option value="latest">latest</option></select>
    <a class="play off" id="playLink" target="_blank" rel="noopener" aria-disabled="true">&#9654; Play</a>
    <div class="stat"><span class="k">iteration</span><span class="v" id="statIter">-</span></div>
    <div class="progress"><i id="progressBar" style="width:0%"></i></div>
    <div class="stat"><span class="k">sec/iter</span><span class="v" id="statSpi">-</span></div>
    <div class="stat"><span class="k">eta</span><span class="v" id="statEta">-</span></div>
    <div class="stat"><span class="k">lr</span><span class="v" id="statLr">-</span></div>
    <div class="stat"><span class="k">loss</span><span class="v" id="statLoss">-</span></div>
    <span class="badge ok" id="staleBadge">-</span>
  </div>
  <details class="cfg" id="cfgDetails"><summary id="cfgSummary">config</summary><pre id="cfgPre"></pre></details>
</div>

<div class="charts">
  <div class="chart-box">
    <h2>loss (log scale)</h2>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>quick-check metrics (0..1) <a href="#" id="metricShowAll" class="showAll">show all</a></h2>
    <canvas id="metricChart"></canvas>
    <div class="legend" id="metricLegend"></div>
  </div>
  <div class="chart-box" id="poolStatsBox" hidden>
    <h2>pool statistics</h2>
    <canvas id="poolStatsChart"></canvas>
    <div class="legend" id="poolStatsLegend"></div>
  </div>
</div>

<div id="gallerySection"></div>
<div id="poolSection"></div>

<div class="overlay" id="detailOverlay" hidden>
  <div class="detail-panel">
    <button class="close" id="detailClose">&times;</button>
    <h3 id="detailTitle"></h3>
    <div class="detail-grid" id="detailGrid"></div>
  </div>
</div>

<footer><span id="footerPoll">polls every ~3s</span> &middot; <span id="footerNote"></span></footer>

<script>
(function () {
"use strict";

// The static copy (python -m nca.dashboard --static) bakes in STATIC = true: the page then reads the
// exported files next to it instead of /api/*, never lists anything, and polls every 15 s.
var STATIC = false; /*STATIC*/
var POLL_MS = STATIC ? 15000 : 3000;
var RUN_LISTS = ["runs-pi.json", "runs-vm.json"];   // one per publisher (dashboard.STATIC_SOURCES)
var SAFE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]*$/;      // a run name used as a relative path
var listTimes = {};                                  // runs-<source>.json -> its exported time (static)
function logURL(name) { return STATIC ? encodeURIComponent(name) + "/log.json" : "/api/log?run=" + encodeURIComponent(name); }
function poolURL(name) { return STATIC ? encodeURIComponent(name) + "/pool.json" : "/api/pool?run=" + encodeURIComponent(name); }
function galleryURL(name) { return STATIC ? encodeURIComponent(name) + "/gallery.json" : "/api/gallery?run=" + encodeURIComponent(name); }
// The runs, newest log first: /api/runs, or (static) the publishers' lists merged, the newer entry winning a
// name both have.
function fetchRuns() {
  if (!STATIC) return getJSON("/api/runs");
  return Promise.all(RUN_LISTS.map(function (f) {
    return getJSON(f).then(function (l) { return Array.isArray(l) ? l : []; }, function () { return []; });
  })).then(function (lists) {
    var byName = Object.create(null);
    lists.forEach(function (l, i) {
      listTimes[RUN_LISTS[i]] = null;
      l.forEach(function (r) {
        if (!r || typeof r.name !== "string" || !SAFE_NAME.test(r.name)) return;
        if (typeof r.exported === "number") listTimes[RUN_LISTS[i]] = r.exported;
        var old = byName[r.name];
        if (!old || (r.mtime || 0) > (old.mtime || 0)) byName[r.name] = r;
      });
    });
    return Object.keys(byName).map(function (k) { return byName[k]; })
      .sort(function (a, b) { return (b.mtime || 0) - (a.mtime || 0); });
  });
}

var PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac","#86bcb6","#d37295"];
var colorMap = Object.create(null);
function colorFor(label) {
  if (!colorMap[label]) {
    var n = Object.keys(colorMap).length;
    colorMap[label] = PALETTE[n % PALETTE.length];
  }
  return colorMap[label];
}

var DAMAGE_LABELS = { "-1": "none", "0": "edit", "1": "burst", "2": "erase", "3": "stamp", "4": "state", "5": "spiral" };

// ---- small fetch helper --------------------------------------------------------------------
function getJSON(url) {
  return fetch(url, { cache: "no-store" }).then(function (r) {
    if (!r.ok) { var e = new Error("http " + r.status); e.status = r.status; throw e; }
    return r.json();
  });
}

// ---- state ----------------------------------------------------------------------------------
var state = {
  runs: [],
  selected: "latest",       // "latest" or a run name
  activeRun: null,
  config: {},
  records: [],
  pool: null,                // last successfully loaded pool JSON, or null
  gallery: null,             // last successfully loaded gallery JSON, or null
  hiddenMetrics: Object.create(null),
  hiddenPoolStats: Object.create(null),
  sortMode: "pool",
  showTarget: false,
  boardSize: "m",
};

var boardDom = Object.create(null); // "<R>:<idx>" -> {wrap, canvas}
var currentGridRun = null;
var currentGridRadii = null;

// ---- thumbnail size: pool boards and the channel-detail small multiples share one "size"
//      control (next to sort), remembered in localStorage. Changing it only touches CSS custom
//      properties (the hex geometry itself is resolution-independent, drawBoard reads the
//      canvas's own CSS box each time), then asks the pool to redraw at the new box size.
var BOARD_SIZES = {
  s: { w: 112, h: 86, detailMin: 96, detailH: 86 },     // the original size
  m: { w: 224, h: 172, detailMin: 192, detailH: 172 },  // default: ~2x the original
  l: { w: 320, h: 246, detailMin: 272, detailH: 246 },
};
var BOARD_SIZE_KEY = "ncaDash.boardSize";
function loadBoardSize() {
  try {
    var v = localStorage.getItem(BOARD_SIZE_KEY);
    if (v && BOARD_SIZES[v]) return v;
  } catch (e) { /* no storage (private mode, etc): fall back to the default */ }
  return "m";
}
function applyBoardSize(size) {
  var p = BOARD_SIZES[size] || BOARD_SIZES.m;
  var root = document.documentElement.style;
  root.setProperty("--board-w", p.w + "px");
  root.setProperty("--board-h", p.h + "px");
  root.setProperty("--detail-min", p.detailMin + "px");
  root.setProperty("--detail-h", p.detailH + "px");
}
function setBoardSize(size) {
  if (!BOARD_SIZES[size]) size = "m";
  state.boardSize = size;
  applyBoardSize(size);
  try { localStorage.setItem(BOARD_SIZE_KEY, size); } catch (e) { /* best-effort */ }
  if (state.pool) renderPool(state.pool);
}
state.boardSize = loadBoardSize();
applyBoardSize(state.boardSize);

// ---- header -----------------------------------------------------------------------------------
function fmtDuration(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return "-";
  sec = Math.round(sec);
  var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return h + "h " + m + "m";
  if (m) return m + "m " + s + "s";
  return s + "s";
}
var TARGET_KEYS = ["iters", "iterations", "targetIterations", "totalIterations", "maxIterations", "nIters", "steps"];
// The config line's shape isn't pinned down: the demo (and an older trainer) write it flat,
// the current one wraps it one level under "config" (alongside sibling keys like
// startIteration). Look in both places rather than assuming either.
function configCandidates(cfg) {
  var out = [cfg || {}];
  if (cfg && cfg.config && typeof cfg.config === "object" && !Array.isArray(cfg.config)) out.push(cfg.config);
  return out;
}
// A strand run (nca/strand/train.py, train2.py) is told apart from a flood run (nca/train.py) by its
// config's "task" (m1a / m1b) -- never by the run's name, which is free text and may say anything.
function strandTask(cfg) {
  var candidates = configCandidates(cfg);
  for (var c = 0; c < candidates.length; c++) {
    var t = candidates[c].task;
    if (t === "m1a" || t === "m1b") return t;
  }
  return null;
}
function isStrandConfig(cfg) { return !!strandTask(cfg); }
function configNumber(cfg, keys) {
  var candidates = configCandidates(cfg);
  for (var c = 0; c < candidates.length; c++) {
    for (var i = 0; i < keys.length; i++) {
      var v = candidates[c][keys[i]];
      if (typeof v === "number" && isFinite(v)) return v;
    }
  }
  return null;
}
function targetIterations(cfg) {
  var v = configNumber(cfg, TARGET_KEYS);
  return v != null && v > 0 ? v : null;
}

function updateHeader(run, cfg, records, runsMeta) {
  var last = records.length ? records[records.length - 1] : null;
  var iter = last ? last.iteration : (run ? run.lastIteration : null);
  var target = targetIterations(cfg);
  document.getElementById("statIter").textContent = (iter == null ? "-" : iter) + (target ? " / " + target : "");
  var bar = document.getElementById("progressBar");
  if (target && iter != null) { bar.style.width = Math.max(0, Math.min(100, 100 * iter / target)) + "%"; }
  else { bar.style.width = "0%"; }

  // a clean end ({"stopped": ...}) is the last line but has no loss or s/iter: read those off the last
  // record that has them
  for (var li = records.length - 1; li >= 0; li--) {
    if (typeof records[li].secPerIter === "number") { last = Object.assign({}, records[li], { iteration: last.iteration }); break; }
  }
  var spi = last && typeof last.secPerIter === "number" ? last.secPerIter : null;
  document.getElementById("statSpi").textContent = spi != null ? spi.toFixed(2) + " s" : "-";
  var lr = last && typeof last.lr === "number" ? last.lr : configNumber(cfg, ["lr"]);
  document.getElementById("statLr").textContent = lr != null ? lr.toExponential(2) : "-";
  var loss = last && typeof last.loss === "number" ? last.loss : null;
  document.getElementById("statLoss").textContent = loss != null ? loss.toPrecision(4) : "-";

  var eta = (target && iter != null && spi != null && iter < target) ? (target - iter) * spi : null;
  document.getElementById("statEta").textContent = (target && iter != null && iter >= target) ? "done" : fmtDuration(eta);

  // staleness: compare time since the log file's mtime to the expected gap between log lines,
  // estimated from the last two records' iteration delta x the last record's secPerIter.
  var badge = document.getElementById("staleBadge");
  if (run && run.mtime != null) {
    var dataTime = run.mtime;
    if (STATIC) {  // the data's own time: the last log line's "time" (the trainer writes one), else the file's
      for (var ti = records.length - 1; ti >= 0; ti--) {
        if (typeof records[ti].time === "number") { dataTime = Math.max(records[ti].time, 0); break; }
      }
    }
    var age = Date.now() / 1000 - dataTime;
    var expected = null;
    if (records.length >= 2 && spi) {
      var dIter = records[records.length - 1].iteration - records[records.length - 2].iteration;
      if (dIter > 0) expected = dIter * spi;
    }
    if (expected == null && spi) expected = 50 * spi; // fallback: the common 50-iteration cadence
    var ratio = expected ? age / expected : 0;
    badge.textContent = (STATIC ? "data from " + new Date(dataTime * 1000).toLocaleString() + ", " : "updated ")
      + fmtDuration(age) + " ago";
    badge.className = "badge " + (ratio >= 5 ? "bad" : ratio >= 2 ? "amber" : "ok");
  } else {
    badge.textContent = "-";
    badge.className = "badge";
  }

  var summaryObj = (cfg && cfg.config && typeof cfg.config === "object" && !Array.isArray(cfg.config)) ? cfg.config : cfg;
  var summaryKeys = Object.keys(summaryObj);
  var summary = summaryKeys.slice(0, 6).map(function (k) { return k + "=" + JSON.stringify(summaryObj[k]); }).join("  ");
  document.getElementById("cfgSummary").textContent = "config" + (summary ? "  (" + summary + (summaryKeys.length > 6 ? ", ..." : "") + ")" : " (none)");
  document.getElementById("cfgPre").textContent = JSON.stringify(cfg, null, 2);
}

// ---- numeric flattening for charts ----------------------------------------------------------
// time (unix seconds), evalN and a stop line's minutes / slowestIterSec are bookkeeping, not metrics to plot
var INFRA_KEYS = { iteration: 1, loss: 1, secPerIter: 1, lr: 1, maxRssMB: 1, evalSec: 1, time: 1, evalN: 1,
                   minutes: 1, slowestIterSec: 1, startIteration: 1 };
// Walk a record's own (non-infra) fields; numeric leaves become dotted-path metrics, grouped by
// their top-level key. A top-level key containing "pool" (any case) is pool-statistics, not a
// quick-check metric -- e.g. a future "poolStats": {...} record field.
function flattenMetrics(record) {
  var metrics = {}, pool = {};
  Object.keys(record).forEach(function (key) {
    if (INFRA_KEYS[key]) return;
    var isPool = /pool/i.test(key);
    var target = isPool ? pool : metrics;
    (function walk(prefix, v) {
      if (typeof v === "number" && isFinite(v)) { target[prefix] = v; return; }
      if (Array.isArray(v)) {
        // a numeric array (e.g. one value per radius) isn't a dict, but it's still numbers --
        // index it rather than drop it, so nothing logged goes unplotted.
        v.forEach(function (item, i) { walk(prefix + "." + i, item); });
        return;
      }
      if (v && typeof v === "object") {
        Object.keys(v).forEach(function (k) { walk(prefix + "." + k, v[k]); });
      }
      // strings/bools/null: not a plottable metric, skip silently
    })(key, record[key]);
  });
  return { metrics: metrics, pool: pool };
}
function collectSeries(records, which) {
  var series = Object.create(null); // label -> [[iteration, value], ...]
  records.forEach(function (rec) {
    if (typeof rec.iteration !== "number") return;
    var flat = flattenMetrics(rec)[which];
    Object.keys(flat).forEach(function (label) {
      (series[label] || (series[label] = [])).push([rec.iteration, flat[label]]);
    });
  });
  return series;
}

// ---- generic canvas line chart ---------------------------------------------------------------
function devicePixelSetup(canvas) {
  var rect = canvas.getBoundingClientRect();
  var w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height || 150));
  var dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  var ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx: ctx, w: w, h: h };
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function LineChart(canvasId, opts) {
  this.canvas = document.getElementById(canvasId);
  this.opts = opts || {};
  this.series = {};     // label -> points
  this.hidden = opts.hidden || Object.create(null);
  this._lastLayout = null;
  var self = this;
  this.canvas.addEventListener("mousemove", function (ev) { self._hover(ev); });
  this.canvas.addEventListener("mouseleave", function () { self._hideTip(); });
}
LineChart.prototype._hideTip = function () {
  if (this._tip) { this._tip.remove(); this._tip = null; }
};
LineChart.prototype._hover = function (ev) {
  var L = this._lastLayout;
  if (!L || !L.allX.length) return;
  var rect = this.canvas.getBoundingClientRect();
  var mx = ev.clientX - rect.left;
  var frac = (mx - L.padL) / (L.w - L.padL - L.padR);
  var target = L.minX + frac * (L.maxX - L.minX);
  var best = L.allX[0], bestD = Infinity;
  for (var i = 0; i < L.allX.length; i++) { var d = Math.abs(L.allX[i] - target); if (d < bestD) { bestD = d; best = L.allX[i]; } }
  var lines = ["iter " + best];
  Object.keys(this.series).forEach(function (label) {
    if (this.hidden[label]) return;
    var pts = this.series[label];
    for (var i = 0; i < pts.length; i++) {
      if (pts[i][0] === best) { lines.push(label + ": " + (+pts[i][1].toPrecision(4))); break; }
    }
  }, this);
  if (!this._tip) {
    this._tip = document.createElement("div");
    this._tip.className = "tooltip";
    document.body.appendChild(this._tip);
  }
  this._tip.textContent = lines.join("\n");
  this._tip.style.left = (ev.clientX + 12) + "px";
  this._tip.style.top = (ev.clientY + 12) + "px";
};
LineChart.prototype.setSeries = function (series) { this.series = series; };
LineChart.prototype.draw = function () {
  var d = devicePixelSetup(this.canvas), ctx = d.ctx, w = d.w, h = d.h;
  var fg = cssVar("--fg"), muted = cssVar("--muted"), hair = cssVar("--hair");
  ctx.clearRect(0, 0, w, h);
  var padL = 34, padR = 6, padT = 6, padB = 16;
  var labels = Object.keys(this.series).filter(function (l) { return !this.hidden[l]; }, this);
  var allX = [], minX = Infinity, maxX = -Infinity;
  labels.forEach(function (l) {
    this.series[l].forEach(function (p) { if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0]; allX.push(p[0]); });
  }, this);
  if (!isFinite(minX)) { minX = 0; maxX = 1; }
  if (minX === maxX) maxX = minX + 1;
  allX = Array.from(new Set(allX)).sort(function (a, b) { return a - b; });

  var log = !!this.opts.log;
  var fixed01 = !!this.opts.fixed01;
  var minY = Infinity, maxY = -Infinity;
  labels.forEach(function (l) {
    this.series[l].forEach(function (p) {
      var v = p[1]; if (log) v = Math.log10(Math.max(v, 1e-8));
      if (v < minY) minY = v; if (v > maxY) maxY = v;
    });
  }, this);
  // "0..1 axis": fixed, deliberately -- a quick-check metric is a share/accuracy and belongs
  // here. A future numeric field logged outside that range (a raw count, say) still draws
  // (nothing throws), just pinned off the top of this chart rather than stretching the shared
  // axis and crushing every well-behaved 0..1 line into a sliver near zero.
  if (fixed01) { minY = 0; maxY = 1; }
  if (!isFinite(minY) || !isFinite(maxY)) { minY = 0; maxY = 1; }
  if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
  var yPad = (maxY - minY) * 0.08;
  minY -= yPad; maxY += yPad;

  this._lastLayout = { padL: padL, padR: padR, w: w, h: h, minX: minX, maxX: maxX, allX: allX };

  function X(x) { return padL + (w - padL - padR) * (x - minX) / (maxX - minX || 1); }
  function Y(y) { if (log) y = Math.log10(Math.max(y, 1e-8)); return padT + (h - padT - padB) * (1 - (y - minY) / (maxY - minY || 1)); }

  ctx.strokeStyle = hair; ctx.fillStyle = muted; ctx.font = "10px monospace"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
  var yTicks = 4;
  for (var i = 0; i <= yTicks; i++) {
    var yv = minY + (maxY - minY) * i / yTicks;
    var py = padT + (h - padT - padB) * (1 - i / yTicks);
    ctx.strokeStyle = hair; ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
    ctx.globalAlpha = 1;
    var label = log ? Math.pow(10, yv).toPrecision(2) : yv.toFixed(2);
    ctx.fillText(label, 2, py + 3);
  }
  if (allX.length) {
    ctx.fillText(String(minX), padL, h - 4);
    var mx = String(maxX);
    ctx.fillText(mx, w - padR - ctx.measureText(mx).width, h - 4);
  }

  labels.forEach(function (label) {
    var pts = this.series[label];
    if (!pts.length) return;
    ctx.strokeStyle = colorFor(label);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    var started = false;
    pts.forEach(function (p) {
      var x = X(p[0]), y = Y(p[1]);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
    if (pts.length < 40) {
      ctx.fillStyle = colorFor(label);
      pts.forEach(function (p) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 1.6, 0, 7); ctx.fill(); });
    }
  }, this);

  if (!labels.length) {
    ctx.fillStyle = muted; ctx.fillText("nothing logged yet", padL + 4, h / 2);
  }
};

function buildLegend(container, series, hiddenSet, chart, onToggle) {
  container.innerHTML = "";
  var groups = {};
  Object.keys(series).sort().forEach(function (label) {
    var g = label.indexOf(".") >= 0 ? label.slice(0, label.indexOf(".")) : "";
    (groups[g] || (groups[g] = [])).push(label);
  });
  Object.keys(groups).sort().forEach(function (g) {
    if (g) { var h = document.createElement("div"); h.className = "grp"; h.textContent = g; container.appendChild(h); }
    groups[g].forEach(function (label) {
      var btn = document.createElement("button");
      btn.className = hiddenSet[label] ? "off" : "";
      var sw = document.createElement("span"); sw.className = "sw"; sw.style.background = colorFor(label);
      btn.appendChild(sw);
      btn.appendChild(document.createTextNode(label));
      btn.addEventListener("click", function () {
        hiddenSet[label] = !hiddenSet[label];
        btn.className = hiddenSet[label] ? "off" : "";
        chart.hidden = hiddenSet;
        chart.draw();
        if (onToggle) onToggle();
      });
      container.appendChild(btn);
    });
  });
}

// ---- metric visibility: ~60 quick-check series in one tangle by default is not useful -- show a
// sensible few (strand: exact/balanced/byLen; flood: each q<N> group's both/bridge/mix) and hide the
// rest behind the legend's existing toggles (or "show all"), remembered per browser (localStorage,
// best-effort: a private window or cleared storage just means it decides afresh every visit). Decided
// PER LABEL, once, the first time it's seen -- so a metric that starts appearing later (e.g. q.code
// once the probe kicks in) still gets a sensible default instead of silently popping in hidden or not.
var METRIC_HIDDEN_KEY = "ncaDash.hiddenMetrics";
var STRAND_METRIC_SHOW = ["q.exact", "q.balanced", "q.byLen.short.exact", "q.byLen.medium.exact",
                          "q.byLen.long.exact", "q.code.exact"];
function loadStoredHidden(key) {
  try {
    var p = JSON.parse(localStorage.getItem(key) || "null");
    return (p && typeof p === "object" && !Array.isArray(p)) ? p : {};
  } catch (e) { return {}; }
}
function saveStoredHidden(key, hiddenSet) {
  try { localStorage.setItem(key, JSON.stringify(hiddenSet)); } catch (e) { /* best-effort: private mode, etc */ }
}
var metricHiddenStored = loadStoredHidden(METRIC_HIDDEN_KEY);
var metricLabelsSeen = Object.create(null);
function defaultVisibleMetric(label, isStrand) {
  if (isStrand) return STRAND_METRIC_SHOW.indexOf(label) >= 0;
  return /^q\d+\.(both|bridge|mix)$/.test(label);  // flood: q8 / q16's nearest analogues of exact/bridge/mix
}
function ensureMetricDefaults(labels, isStrand) {
  labels.forEach(function (label) {
    if (metricLabelsSeen[label]) return;
    metricLabelsSeen[label] = true;
    state.hiddenMetrics[label] = Object.prototype.hasOwnProperty.call(metricHiddenStored, label)
      ? !!metricHiddenStored[label] : !defaultVisibleMetric(label, isStrand);
  });
}

var lossChart = new LineChart("lossChart", { log: true });
var metricChart = new LineChart("metricChart", { fixed01: true, hidden: state.hiddenMetrics });
var poolStatsChart = new LineChart("poolStatsChart", { hidden: state.hiddenPoolStats });

function updateCharts(records) {
  var lossSeries = { loss: [] };
  records.forEach(function (r) { if (typeof r.iteration === "number" && typeof r.loss === "number" && isFinite(r.loss) && r.loss > 0) lossSeries.loss.push([r.iteration, r.loss]); });
  lossChart.setSeries(lossSeries);
  lossChart.draw();

  var metricSeries = collectSeries(records, "metrics");
  ensureMetricDefaults(Object.keys(metricSeries), isStrandConfig(state.config));
  metricChart.setSeries(metricSeries);
  metricChart.hidden = state.hiddenMetrics;
  metricChart.draw();
  buildLegend(document.getElementById("metricLegend"), metricSeries, state.hiddenMetrics, metricChart,
    function () { saveStoredHidden(METRIC_HIDDEN_KEY, state.hiddenMetrics); });

  var poolSeries = collectSeries(records, "pool");
  var box = document.getElementById("poolStatsBox");
  if (Object.keys(poolSeries).length) {
    box.hidden = false;
    poolStatsChart.setSeries(poolSeries);
    poolStatsChart.hidden = state.hiddenPoolStats;
    poolStatsChart.draw();
    buildLegend(document.getElementById("poolStatsLegend"), poolSeries, state.hiddenPoolStats, poolStatsChart);
  } else {
    box.hidden = true;
  }
}

// ---- hex board geometry (pointy-top; matches nca/hexgrid.py: cell (q,r) at array[r+R][c+R],
//      x = sqrt(3)*(q + r/2), y = 1.5*r; see the repo's array convention note) ------------------
var layoutCache = Object.create(null);
function hexLayout(R, boxW, boxH) {
  var key = R + ":" + boxW + "x" + boxH;
  if (layoutCache[key]) return layoutCache[key];
  var S = 2 * R + 1;
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (var row = 0; row < S; row++) {
    for (var col = 0; col < S; col++) {
      var q = col - R, r = row - R;
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > R) continue;
      var cx = Math.sqrt(3) * (q + r / 2), cy = 1.5 * r;
      for (var i = 0; i < 6; i++) {
        var ang = Math.PI / 180 * (60 * i - 30);
        var x = cx + Math.cos(ang), y = cy + Math.sin(ang);
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  var pad = 2;
  // clamp >= 0: a box read mid-reflow (e.g. a wrap still settling into its new --board-w) can
  // come in smaller than 2*pad; a negative size would feed ctx.arc() a negative radius and throw.
  var size = Math.max(0, Math.min((boxW - 2 * pad) / (maxX - minX), (boxH - 2 * pad) / (maxY - minY)));
  var layout = { S: S, size: size, offsetX: boxW / 2 - size * (minX + maxX) / 2, offsetY: boxH / 2 - size * (minY + maxY) / 2 };
  layoutCache[key] = layout;
  return layout;
}
function hexPath(ctx, px, py, size) {
  ctx.beginPath();
  for (var i = 0; i < 6; i++) {
    var ang = Math.PI / 180 * (60 * i - 30);
    var x = px + size * Math.cos(ang), y = py + size * Math.sin(ang);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}
function mixColor(empty, accent, t) {
  // both are "#rrggbb"
  function hx(c) { return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)]; }
  var a = hx(empty), b = hx(accent);
  var r = Math.round(a[0] + (b[0] - a[0]) * t), g = Math.round(a[1] + (b[1] - a[1]) * t), bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return "rgb(" + r + "," + g + "," + bl + ")";
}

function drawBoard(canvas, R, S, cellsWalls, cellsFill, cellsTarget, showTarget) {
  var rect = canvas.getBoundingClientRect();
  var w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height || 86));
  var dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  var ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var L = hexLayout(R, w, h);
  var empty = cssVar("--empty"), wallColor = cssVar("--wall"), accent = cssVar("--accent"), bad = cssVar("--bad");
  for (var row = 0; row < S; row++) {
    for (var col = 0; col < S; col++) {
      var q = col - R, r = row - R;
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > R) continue;
      var i = row * S + col;
      var isWall = cellsWalls[i] === 1;
      var px = L.offsetX + L.size * Math.sqrt(3) * (q + r / 2), py = L.offsetY + L.size * 1.5 * r;
      hexPath(ctx, px, py, L.size * 0.96);
      if (isWall) {
        ctx.fillStyle = wallColor;
      } else if (showTarget) {
        ctx.fillStyle = cellsTarget[i] ? accent : empty;
      } else {
        ctx.fillStyle = mixColor(empty, accent, Math.max(0, Math.min(1, cellsFill[i])));
      }
      ctx.fill();
      if (!isWall && !showTarget) {
        var thresholded = cellsFill[i] > 0.5 ? 1 : 0;
        var target = cellsTarget[i];
        if (thresholded && !target) {
          ctx.fillStyle = bad;
          ctx.beginPath(); ctx.arc(px, py, L.size * 0.32, 0, 7); ctx.fill();
        } else if (!thresholded && target) {
          ctx.strokeStyle = bad; ctx.lineWidth = Math.max(1, L.size * 0.18);
          ctx.beginPath(); ctx.arc(px, py, L.size * 0.4, 0, 7); ctx.stroke();
        }
      }
    }
  }
}

// ---- strand boards: a line pattern, not a flood fill (nca/strand/train.py, train2.py) ------------
// The trainer embeds its small S0 x S0 board (S0 = R + 1) in one corner of the R-hex's S x S array
// (see write_snapshot in both trainers), at q = col - R in 0..R, r = row - R in -R..0 -- exactly a
// sixth of the hex. Laying out just that sixth (instead of the whole hex, mostly empty padding) is
// what makes a strand board legible instead of a speck in a sea of "empty".
function hexLayoutStrand(R, boxW, boxH) {
  var key = "s" + R + ":" + boxW + "x" + boxH;
  if (layoutCache[key]) return layoutCache[key];
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (var q = 0; q <= R; q++) {
    for (var r = -R; r <= 0; r++) {
      var cx = Math.sqrt(3) * (q + r / 2), cy = 1.5 * r;
      for (var i = 0; i < 6; i++) {
        var ang = Math.PI / 180 * (60 * i - 30);
        var x = cx + Math.cos(ang), y = cy + Math.sin(ang);
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (!isFinite(minX)) { minX = maxX = minY = maxY = 0; }
  var pad = 2;
  var size = Math.max(0, Math.min((boxW - 2 * pad) / Math.max(1e-6, maxX - minX),
                                   (boxH - 2 * pad) / Math.max(1e-6, maxY - minY)));
  var layout = { size: size, offsetX: boxW / 2 - size * (minX + maxX) / 2, offsetY: boxH / 2 - size * (minY + maxY) / 2 };
  layoutCache[key] = layout;
  return layout;
}
// Direction d (0..5, walker.py's PAIRS / DIRS = src/hex.ts DIRS): the pixel angle of its edge's
// midpoint from the cell centre, in the same (pointy-top) layout as hexPath's vertices (60*i - 30).
// Worked out from DIRS' (dq, dr) against cx = sqrt(3)*(q + r/2), cy = 1.5*r: direction d sits at
// -60*d degrees (d=0 -> due "east", then clockwise in screen coordinates as d increases).
function edgeAngle(d) { return Math.PI / 180 * (-60 * d); }
function drawSpoke(ctx, px, py, size, d, color, dashed, widthFrac, alpha) {
  // widthFrac (default 0.16) and alpha (default 1) let a caller draw the same chord thin/faint (a
  // gallery tile's "missed" target line) or bold (its "correct" / "extra" prediction) without a
  // separate code path -- the pool grid's calls omit them and look exactly as before.
  var ang = edgeAngle(d);
  ctx.save();
  ctx.globalAlpha = alpha == null ? 1 : alpha;
  ctx.setLineDash(dashed ? [Math.max(1, size * 0.22), Math.max(1, size * 0.16)] : []);
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, size * (widthFrac == null ? 0.16 : widthFrac));
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + size * 0.82 * Math.cos(ang), py + size * 0.82 * Math.sin(ang));
  ctx.stroke();
  ctx.restore();
}
// Per-sample slices out of a strand pool group's flat arrays (see pool_to_json): cell-level fields
// are n0 = S*S long per sample, edge-level fields (tgtEdges / predEdges) are 6 x n0.
function strandHasEdges(group) {
  return !!(group.tgtEdges && group.tgtEdges.length && group.predEdges && group.predEdges.length);
}
function strandTapOf(group, idx) {
  if (!group.tap || group.tap.length < (idx + 1) * 4) return null;
  var t = group.tap.slice(idx * 4, idx * 4 + 4);
  return t[0] >= 0 ? t : null;
}
function drawStrandBoard(canvas, R, group, idx, showTarget) {
  var rect = canvas.getBoundingClientRect();
  var w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height || 86));
  var dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  var ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var S = group.S, n0 = S * S;
  var L = hexLayoutStrand(R, w, h);
  var empty = cssVar("--empty"), hair = cssVar("--hair"), accent = cssVar("--accent"),
      good = cssVar("--good"), bad = cssVar("--bad");
  var maskOff = idx * n0, hasMask = group.mask && group.mask.length >= maskOff + n0;
  var hasEdges = strandHasEdges(group), edgeOff = idx * 6 * n0;
  var hasFill = group.fill && group.fill.length >= maskOff + n0;
  var hasTarget = group.target && group.target.length >= maskOff + n0;
  for (var q = 0; q <= R; q++) {
    for (var r = -R; r <= 0; r++) {
      var row = r + R, col = q + R, i = row * S + col;
      if (hasMask && !group.mask[maskOff + i]) continue;
      var px = L.offsetX + L.size * Math.sqrt(3) * (q + r / 2), py = L.offsetY + L.size * 1.5 * r;
      hexPath(ctx, px, py, L.size * 0.96);
      ctx.fillStyle = empty;
      ctx.fill();
      ctx.strokeStyle = hair;
      ctx.lineWidth = 1;
      ctx.stroke();
      if (hasEdges) {
        for (var d = 0; d < 6; d++) {
          var k = edgeOff + d * n0 + i;
          var t = group.tgtEdges[k] === 1, p = group.predEdges[k] > 0.5;
          if (showTarget) {
            if (t) drawSpoke(ctx, px, py, L.size, d, accent, false);
          } else if (t && p) {
            drawSpoke(ctx, px, py, L.size, d, good, false);
          } else if (t && !p) {
            drawSpoke(ctx, px, py, L.size, d, bad, false);
          } else if (p) {
            drawSpoke(ctx, px, py, L.size, d, bad, true);
          }
        }
      } else {
        // older snapshot, no per-edge planes: the cell-level fallback (still mask-cropped, no
        // opaque "wall" blotting out the signal the way the flood renderer's walls do).
        var fv = hasFill ? group.fill[maskOff + i] : 0;
        var tv = hasTarget ? group.target[maskOff + i] : 0;
        if (showTarget) {
          if (tv) { ctx.fillStyle = accent; ctx.fill(); }
        } else {
          var pred = fv > 0.5 ? 1 : 0;
          if (pred || tv) {
            hexPath(ctx, px, py, L.size * 0.5);
            ctx.fillStyle = pred === tv ? good : bad;
            ctx.fill();
          }
        }
      }
    }
  }
  var tap = strandTapOf(group, idx);
  if (tap) {
    var tq = tap[1], tr = tap[0] - R;
    var tpx = L.offsetX + L.size * Math.sqrt(3) * (tq + tr / 2), tpy = L.offsetY + L.size * 1.5 * tr;
    ctx.beginPath();
    ctx.arc(tpx, tpy, L.size * 0.42, 0, 7);
    ctx.lineWidth = Math.max(1, L.size * 0.14);
    ctx.strokeStyle = accent;
    ctx.stroke();
  }
}
function strandSummaryLine(group) {
  return "n=" + group.n + (strandHasEdges(group)
    ? "  chord-level detail for every board"
    : "  cell-level only (older snapshot; chord detail needs a trainer restart on the fix)");
}

// ---- gallery panel: a FIXED set of held-out taps (train.write_gallery's contract: gallery.npz, read by
// nca.dashboard.gallery_to_json), the target strand faint/thin and the model's prediction bold, as chords
// -- what Spectacle's rule draws against what the model draws, watched over training. Shown above the pool
// grid for ANY run that has a gallery.npz (in practice, only strand runs ever write one); independent of
// isStrandConfig, which only gates the POOL grid's rendering mode.
function hexLayoutGallery(mask, S, boxW, boxH) {
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (var row = 0; row < S; row++) {
    for (var col = 0; col < S; col++) {
      if (!mask[row * S + col]) continue;
      var cx = Math.sqrt(3) * (col + row / 2), cy = 1.5 * row;
      for (var i = 0; i < 6; i++) {
        var ang = Math.PI / 180 * (60 * i - 30);
        var x = cx + Math.cos(ang), y = cy + Math.sin(ang);
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (!isFinite(minX)) { minX = maxX = minY = maxY = 0; }
  var pad = 3;
  var size = Math.max(0, Math.min((boxW - 2 * pad) / Math.max(1e-6, maxX - minX),
                                   (boxH - 2 * pad) / Math.max(1e-6, maxY - minY)));
  return { size: size, offsetX: boxW / 2 - size * (minX + maxX) / 2, offsetY: boxH / 2 - size * (minY + maxY) / 2 };
}
function galleryTileExact(group, idx) {
  var S = group.S, n0 = S * S, maskOff = idx * n0, edgeOff = idx * 6 * n0;
  for (var i = 0; i < n0; i++) {
    if (!group.mask[maskOff + i]) continue;
    for (var d = 0; d < 6; d++) {
      var k = edgeOff + d * n0 + i;
      if ((group.tgtEdges[k] === 1) !== (group.predEdges[k] === 1)) return false;
    }
  }
  return true;
}
function drawGalleryTile(canvas, group, idx) {
  var rect = canvas.getBoundingClientRect();
  var w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height || 220));
  var dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  var ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var S = group.S, n0 = S * S, maskOff = idx * n0, edgeOff = idx * 6 * n0;
  var L = hexLayoutGallery(group.mask.slice(maskOff, maskOff + n0), S, w, h);
  var empty = cssVar("--empty"), hair = cssVar("--hair"), accent = cssVar("--accent"),
      good = cssVar("--good"), bad = cssVar("--bad");
  for (var row = 0; row < S; row++) {
    for (var col = 0; col < S; col++) {
      var i = row * S + col;
      if (!group.mask[maskOff + i]) continue;
      var px = L.offsetX + L.size * Math.sqrt(3) * (col + row / 2), py = L.offsetY + L.size * 1.5 * row;
      hexPath(ctx, px, py, L.size * 0.97);
      ctx.fillStyle = empty;
      ctx.fill();
      ctx.strokeStyle = hair;
      ctx.lineWidth = 1;
      ctx.stroke();
      for (var d = 0; d < 6; d++) {
        var k = edgeOff + d * n0 + i;
        var t = group.tgtEdges[k] === 1, p = group.predEdges[k] === 1;
        if (t && p) {
          drawSpoke(ctx, px, py, L.size, d, good, false, 0.22, 1);     // correct: bold, solid
        } else if (t) {
          drawSpoke(ctx, px, py, L.size, d, bad, false, 0.08, 0.55);   // missed: thin, faint
        } else if (p) {
          drawSpoke(ctx, px, py, L.size, d, bad, true, 0.17, 1);       // extra: bold, dashed
        }
      }
    }
  }
  var tap = group.tap.slice(idx * 4, idx * 4 + 4);
  if (tap[0] >= 0) {
    var tpx = L.offsetX + L.size * Math.sqrt(3) * (tap[1] + tap[0] / 2), tpy = L.offsetY + L.size * 1.5 * tap[0];
    ctx.beginPath();
    ctx.arc(tpx, tpy, L.size * 0.4, 0, 7);
    ctx.lineWidth = Math.max(1, L.size * 0.13);
    ctx.strokeStyle = accent;
    ctx.stroke();
  }
}

var galleryDom = Object.create(null);   // "<level>:<idx>" -> {wrap, canvas, cap}
var currentGalleryRun = null;
var currentGalleryLevels = null;

function ensureGalleryGrid(gallery) {
  var levelsKey = gallery.levels.join(",");
  if (currentGalleryRun === state.activeRun && currentGalleryLevels === levelsKey) return;
  currentGalleryRun = state.activeRun;
  currentGalleryLevels = levelsKey;
  galleryDom = Object.create(null);
  var section = document.getElementById("gallerySection");
  section.innerHTML = "";
  section.className = "gallery-section";
  var hdr = document.createElement("div");
  hdr.className = "gallery-hdr";
  hdr.innerHTML = "<h2>gallery (held-out, watched over training)</h2><span class=\"gallery-note\" id=\"galleryIter\"></span>";
  section.appendChild(hdr);
  var legend = document.createElement("div");
  legend.className = "gallery-legend";
  legend.innerHTML = "target: faint line &middot; prediction: bold &middot; " +
    "<span class=\"good\">green</span> = correct &middot; <span class=\"bad\">thin red</span> = missed &middot; " +
    "<span class=\"bad\">dashed red</span> = extra &middot; ring = tap";
  section.appendChild(legend);
  gallery.levels.forEach(function (L) {
    var group = gallery.by_level[String(L)];
    if (!group) return;
    var lvl = document.createElement("div");
    var lh = document.createElement("div");
    lh.className = "summary";
    lh.textContent = "level " + L;
    lvl.appendChild(lh);
    var grid = document.createElement("div");
    grid.className = "gallery-grid";
    for (var i = 0; i < group.n; i++) {
      var wrap = document.createElement("div");
      wrap.className = "gallery-tile";
      var canvas = document.createElement("canvas");
      wrap.appendChild(canvas);
      var cap = document.createElement("div");
      cap.className = "gallery-cap";
      wrap.appendChild(cap);
      grid.appendChild(wrap);
      galleryDom[L + ":" + i] = { wrap: wrap, canvas: canvas, cap: cap };
    }
    lvl.appendChild(grid);
    section.appendChild(lvl);
  });
}

function renderGallery(gallery) {
  var section = document.getElementById("gallerySection");
  if (!gallery || !gallery.levels || !gallery.levels.length) {
    section.className = "";
    section.innerHTML = "";
    currentGalleryRun = null;
    currentGalleryLevels = null;
    return;
  }
  ensureGalleryGrid(gallery);
  var iterEl = document.getElementById("galleryIter");
  if (iterEl) iterEl.textContent = gallery.iteration != null ? "iteration " + gallery.iteration : "";
  gallery.levels.forEach(function (L) {
    var group = gallery.by_level[String(L)];
    if (!group) return;
    for (var i = 0; i < group.n; i++) {
      var dom = galleryDom[L + ":" + i];
      if (!dom) continue;
      drawGalleryTile(dom.canvas, group, i);
      var exact = galleryTileExact(group, i);
      var rule = (group.rule[i] || "").toString();
      dom.cap.innerHTML = "<span>" + rule + "</span><span>len " + group.length[i] +
        (exact ? " <span class=\"ok\">&#10003;</span>" : "") + "</span>";
    }
  });
}

// ---- pool grid ---------------------------------------------------------------------------------
function boardStats(group, idx) {
  var S = group.S, n0 = S * S;
  var off = idx * n0;
  return {
    walls: group.walls.slice(off, off + n0),
    fill: group.fill.slice(off, off + n0),
    target: group.target.slice(off, off + n0),
    loss: group.loss[idx],
    age: group.age[idx],
    edits: group.edits[idx],
    damage: group.damage[idx],
    ntargets: group.ntargets[idx],
  };
}
function sortedIdx(group, mode) {
  var idx = []; for (var i = 0; i < group.n; i++) idx.push(i);
  if (mode === "loss") {
    idx.sort(function (a, b) { var la = group.loss[a], lb = group.loss[b];
      var na = (la == null || isNaN(la)), nb = (lb == null || isNaN(lb));
      if (na && nb) return a - b; if (na) return 1; if (nb) return -1; return lb - la; });
  } else if (mode === "age") {
    idx.sort(function (a, b) { return group.age[b] - group.age[a]; });
  }
  return idx;
}
function summaryLine(group) {
  var n = group.n, exact = 0, lossSum = 0, lossN = 0, anyTarget = 0;
  for (var i = 0; i < n; i++) {
    var S = group.S, n0 = S * S, off = i * n0;
    var ok = true, hasTarget = false;
    for (var c = 0; c < n0; c++) {
      if (group.walls[off + c]) continue;
      var t = group.target[off + c];
      if (t) hasTarget = true;
      var pred = group.fill[off + c] > 0.5 ? 1 : 0;
      if (pred !== t) ok = false;
    }
    if (ok) exact++;
    if (hasTarget) anyTarget++;
    var l = group.loss[i];
    if (l != null && !isNaN(l)) { lossSum += l; lossN++; }
  }
  return "n=" + n + "  exact " + (100 * exact / n).toFixed(0) + "%" +
    "  mean loss " + (lossN ? (lossSum / lossN).toPrecision(3) : "-") +
    "  any-target " + (100 * anyTarget / n).toFixed(0) + "%";
}

function ensureGrid(pool) {
  var isStrand = isStrandConfig(state.config);
  var runChanged = currentGridRun !== state.activeRun;
  // isStrand is folded into the cache key, not just radii: the pool and log fetches race (poll()),
  // so the first grid built for a run may not yet know its config -- once it does (isStrand flips),
  // this forces a rebuild (the legend, which boards are clickable), not just a redraw.
  var radiiKey = pool.radii.join(",") + (isStrand ? ":s" : ":f");
  if (!runChanged && currentGridRadii === radiiKey) return;
  currentGridRun = state.activeRun;
  currentGridRadii = radiiKey;
  boardDom = Object.create(null);
  var section = document.getElementById("poolSection");
  section.innerHTML = "";
  var top = document.createElement("div");
  top.className = "controls";
  top.innerHTML =
    '<label>sort <select id="sortMode"><option value="pool">pool order</option><option value="loss">loss, descending</option><option value="age">age, descending</option></select></label>' +
    '<label>size <select id="boardSizeSelect"><option value="s">S</option><option value="m">M</option><option value="l">L</option></select></label>' +
    '<label><input type="checkbox" id="showTargetToggle"> show target instead of fill</label>';
  if (isStrand) {
    top.innerHTML += '<span style="color:var(--muted)">chords: <span style="color:var(--good)">green</span> = ' +
      'correct &middot; <span style="color:var(--bad)">red solid</span> = missed &middot; ' +
      '<span style="color:var(--bad)">red dashed</span> = extra &middot; ring = tap</span>';
  }
  section.appendChild(top);
  top.querySelector("#sortMode").value = state.sortMode;
  top.querySelector("#sortMode").addEventListener("change", function (e) { state.sortMode = e.target.value; renderPool(state.pool); });
  top.querySelector("#boardSizeSelect").value = state.boardSize;
  top.querySelector("#boardSizeSelect").addEventListener("change", function (e) { setBoardSize(e.target.value); });
  top.querySelector("#showTargetToggle").checked = state.showTarget;
  top.querySelector("#showTargetToggle").addEventListener("change", function (e) { state.showTarget = e.target.checked; renderPool(state.pool); });

  pool.radii.forEach(function (R) {
    var group = pool.by_radius[String(R)];
    if (!group) return;
    var sec = document.createElement("div");
    sec.className = "radius-section";
    sec.id = "radius-" + R;
    var hdr = document.createElement("div");
    hdr.className = "radius-hdr";
    hdr.innerHTML = '<h2>R = ' + R + '</h2><span class="summary" id="summary-' + R + '"></span>';
    sec.appendChild(hdr);
    var grid = document.createElement("div");
    grid.className = "grid";
    grid.id = "grid-" + R;
    for (var i = 0; i < group.n; i++) {
      var wrap = document.createElement("div");
      wrap.className = "board-wrap";
      var canvas = document.createElement("canvas");
      wrap.appendChild(canvas);
      var cap = document.createElement("div");
      cap.className = "cap";
      wrap.appendChild(cap);
      grid.appendChild(wrap);
      boardDom[R + ":" + i] = { wrap: wrap, canvas: canvas, cap: cap };
      if (isStrand || i < group.m) {
        wrap.classList.add("clickable");
        (function (R, i) { canvas.addEventListener("click", function () { openDetail(R, i); }); })(R, i);
      }
    }
    sec.appendChild(grid);
    section.appendChild(sec);
  });
}

function renderPool(pool) {
  if (!pool) { return; }
  ensureGrid(pool);
  var isStrand = isStrandConfig(state.config);
  var highlightSet = Object.create(null);
  if (pool.last_R != null) {
    (pool.last_idx || []).forEach(function (i) { highlightSet[i] = true; });
  }
  pool.radii.forEach(function (R) {
    var group = pool.by_radius[String(R)];
    if (!group) return;
    document.getElementById("summary-" + R).textContent = isStrand ? strandSummaryLine(group) : summaryLine(group);
    var order = sortedIdx(group, state.sortMode);
    var grid = document.getElementById("grid-" + R);
    order.forEach(function (idx, pos) {
      var dom = boardDom[R + ":" + idx];
      if (!dom) return;
      if (grid.children[pos] !== dom.wrap) grid.insertBefore(dom.wrap, grid.children[pos] || null);
      var b = boardStats(group, idx);
      if (isStrand) {
        drawStrandBoard(dom.canvas, R, group, idx, state.showTarget);
      } else {
        drawBoard(dom.canvas, R, group.S, b.walls, b.fill, b.target, state.showTarget);
      }
      var isHighlighted = R === pool.last_R && highlightSet[idx];
      dom.wrap.classList.toggle("highlight", !!isHighlighted);
      var dmg = DAMAGE_LABELS[String(b.damage)] || "?";
      var lossText = (b.loss == null || isNaN(b.loss)) ? "-" : (+b.loss.toPrecision(3));
      dom.cap.innerHTML =
        '<span>' + lossText + (idx < group.m ? ' <span class="state-dot" title="full state available"></span>' : '') + '</span>' +
        '<span class="dmg">' + dmg + (b.ntargets > 1 ? " &times;" + b.ntargets : "") + '</span>' +
        '<span>age ' + b.age + '&middot;e' + b.edits + '</span>';
    });
  });
}

// ---- detail view --------------------------------------------------------------------------------
function divergingColor(v) {
  // v in [-1,1]: blue (neg) .. white (0) .. red (pos)
  v = Math.max(-1, Math.min(1, v));
  if (v >= 0) { var t = v; return "rgb(" + Math.round(255) + "," + Math.round(255 * (1 - t)) + "," + Math.round(255 * (1 - t)) + ")"; }
  var t2 = -v; return "rgb(" + Math.round(255 * (1 - t2)) + "," + Math.round(255 * (1 - t2)) + "," + 255 + ")";
}
function openDetail(R, idx) {
  var group = state.pool && state.pool.by_radius[String(R)];
  if (!group) return;
  var isStrand = isStrandConfig(state.config);
  var hasChannels = !!group.C && idx < group.m;
  if (!isStrand && !hasChannels) return;  // flood: unchanged -- the per-channel view only, first group.m boards
  document.getElementById("detailTitle").textContent =
    "R=" + R + "  board #" + idx + (group.C ? "  (" + group.C + " channels)" : "");
  var grid = document.getElementById("detailGrid");
  grid.innerHTML = "";
  if (isStrand) {
    var big = document.createElement("div");
    big.className = "detail-cell";
    var bigCanvas = document.createElement("canvas");
    bigCanvas.style.height = "360px";
    big.appendChild(bigCanvas);
    var bigLbl = document.createElement("div");
    bigLbl.className = "lbl";
    bigLbl.textContent = strandHasEdges(group) ? "mask / tap / target / predicted, as chords"
      : "mask / target / predicted (cell-level; no per-edge planes in this snapshot)";
    big.appendChild(bigLbl);
    grid.appendChild(big);
    (function (bigCanvas) { requestAnimationFrame(function () { drawStrandBoard(bigCanvas, R, group, idx, false); }); })(bigCanvas);
  }
  if (!hasChannels) { document.getElementById("detailOverlay").hidden = false; return; }
  var S = group.S, C = group.C, n0 = S * S;
  var off = idx * C * n0;
  for (var c = 0; c < C; c++) {
    var chStart = off + c * n0;
    var maxAbs = 1e-6;
    for (var k = 0; k < n0; k++) { var av = Math.abs(group.state[chStart + k]); if (av > maxAbs) maxAbs = av; }
    var cell = document.createElement("div");
    cell.className = "detail-cell";
    var canvas = document.createElement("canvas");
    cell.appendChild(canvas);
    var lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.textContent = "ch" + c + (c === 0 ? " wall" : c === 1 ? " fill" : "") + "  ±" + maxAbs.toPrecision(2);
    cell.appendChild(lbl);
    grid.appendChild(cell);
    (function (canvas, chStart, maxAbs) {
      requestAnimationFrame(function () {
        var rect = canvas.getBoundingClientRect();
        var w = Math.max(1, Math.round(rect.width)) || 90, h = Math.max(1, Math.round(rect.height)) || 86;
        var dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr; canvas.height = h * dpr;
        var ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        var L = hexLayout(R, w, h);
        for (var row = 0; row < S; row++) {
          for (var col = 0; col < S; col++) {
            var q = col - R, r = row - R;
            if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) > R) continue;
            var i = row * S + col;
            var px = L.offsetX + L.size * Math.sqrt(3) * (q + r / 2), py = L.offsetY + L.size * 1.5 * r;
            hexPath(ctx, px, py, L.size * 0.96);
            ctx.fillStyle = divergingColor(group.state[chStart + i] / maxAbs);
            ctx.fill();
          }
        }
      });
    })(canvas, chStart, maxAbs);
  }
  document.getElementById("detailOverlay").hidden = false;
}
document.getElementById("detailClose").addEventListener("click", function () { document.getElementById("detailOverlay").hidden = true; });
document.getElementById("detailOverlay").addEventListener("click", function (e) { if (e.target.id === "detailOverlay") e.currentTarget.hidden = true; });
document.addEventListener("keydown", function (e) { if (e.key === "Escape") document.getElementById("detailOverlay").hidden = true; });

document.getElementById("metricShowAll").addEventListener("click", function (e) {
  e.preventDefault();
  Object.keys(state.hiddenMetrics).forEach(function (label) { state.hiddenMetrics[label] = false; });
  saveStoredHidden(METRIC_HIDDEN_KEY, state.hiddenMetrics);
  metricChart.hidden = state.hiddenMetrics;
  metricChart.draw();
  buildLegend(document.getElementById("metricLegend"), metricChart.series, state.hiddenMetrics, metricChart,
    function () { saveStoredHidden(METRIC_HIDDEN_KEY, state.hiddenMetrics); });
});

// ---- polling loop -------------------------------------------------------------------------------
function populateRunSelect(runs) {
  var sel = document.getElementById("runSelect");
  var wanted = state.selected;
  var names = runs.map(function (r) { return r.name; });
  var cur = Array.from(sel.options).map(function (o) { return o.value; });
  var want = ["latest"].concat(names);
  if (cur.join("|") !== want.join("|")) {
    sel.innerHTML = "";
    var opt0 = document.createElement("option"); opt0.value = "latest"; opt0.textContent = "latest"; sel.appendChild(opt0);
    runs.forEach(function (r) { var o = document.createElement("option"); o.value = r.name; o.textContent = r.name; sel.appendChild(o); });
  }
  sel.value = wanted;
}
document.getElementById("runSelect").addEventListener("change", function (e) {
  state.selected = e.target.value;
  updatePlay(pickActiveRun(state.runs));
});

// ---- Play: the interactive board (dist/nca.html) with the run's best weights, in a new tab ---------
// Locally /play fetching /weights?run=NAME (the server converts best.pt, else ckpt.pt); in the static copy
// play.html fetching NAME/weights.json. A run listed without weights (the VM's: the Pi publishes them,
// publish.sh --weights-only) is looked for with a HEAD of its weights.json, at most once a minute.
var weightsProbe = Object.create(null);   // run name -> {ok, at, pending}
function weightsURL(name) { return STATIC ? encodeURIComponent(name) + "/weights.json" : "/weights?run=" + encodeURIComponent(name); }
function playURL(name) {
  var q = function (v) { return encodeURIComponent(v).replace(/%2F/g, "/"); };  // a "/" may stay as it is in a query
  return (STATIC ? "play.html" : "/play") + "?weights=" + q(weightsURL(name)) + "&name=" + q(name);
}
function setPlay(name, ok, why) {
  var a = document.getElementById("playLink");
  a.classList.toggle("off", !ok);
  if (ok) {
    a.href = playURL(name);
    a.removeAttribute("aria-disabled");
    a.title = "Open the board with " + name + "'s best weights in a new tab: paint walls, watch it fill";
  } else {
    a.removeAttribute("href");
    a.setAttribute("aria-disabled", "true");
    a.title = why;
  }
}
function updatePlay(name) {
  var run = name ? state.runs.find(function (r) { return r.name === name; }) : null;
  if (!run) return setPlay(null, false, "no run");
  if (run.hasWeights) return setPlay(name, true);
  if (!STATIC) return setPlay(name, false, "no checkpoint (best.pt or ckpt.pt) in this run yet");
  var p = weightsProbe[name] || { ok: false, at: 0 };
  setPlay(name, p.ok, p.at ? "no weights published for this run" : "looking for this run's weights...");
  if (p.pending || Date.now() - p.at < 60000) return;
  p.pending = true;
  weightsProbe[name] = p;
  fetch(weightsURL(name), { method: "HEAD", cache: "no-store" })
    .then(function (r) { return r.ok; }, function () { return false; })
    .then(function (ok) {
      weightsProbe[name] = { ok: ok, at: Date.now() };
      if (pickActiveRun(state.runs) === name) updatePlay(name);
    });
}

function pickActiveRun(runs) {
  if (state.selected === "latest") return runs.length ? runs[0].name : null;
  return runs.some(function (r) { return r.name === state.selected; }) ? state.selected : (runs.length ? runs[0].name : null);
}

function poll() {
  fetchRuns().then(function (runs) {
    state.runs = runs;
    populateRunSelect(runs);
    var active = pickActiveRun(runs);
    state.activeRun = active;
    updatePlay(active);
    var runMeta = runs.find(function (r) { return r.name === active; }) || null;
    var note = runs.length + " run" + (runs.length === 1 ? "" : "s") + " found";
    if (STATIC) {
      note += Object.keys(listTimes).map(function (f) {
        return " \u00b7 " + f + ": " + (listTimes[f] ? "exported " + fmtDuration(Date.now() / 1000 - listTimes[f]) + " ago" : "none");
      }).join("");
    }
    document.getElementById("footerNote").textContent = note;
    if (!active) {
      updateHeader(null, {}, [], runs);
      updateCharts([]);
      return;
    }
    getJSON(logURL(active)).then(function (log) {
      state.config = log.config || {};
      state.records = log.records || [];
      updateHeader(runMeta, state.config, state.records, runs);
      updateCharts(state.records);
      // the pool fetch below runs in parallel and may have drawn first, before state.config (and
      // so isStrandConfig) was known -- redraw now that it is, rather than waiting a whole poll.
      if (state.pool) renderPool(state.pool);
    }).catch(function () { /* transient: keep showing the last good data */ });
    getJSON(poolURL(active)).then(function (pool) {
      if (!pool || !pool.radii || !pool.radii.length) {
        state.pool = null;
        var section = document.getElementById("poolSection");
        if (!section.querySelector(".empty-note")) {
          section.innerHTML = '<div class="empty-note">no pool snapshot for this run yet</div>';
          currentGridRun = null; currentGridRadii = null;
        }
        return;
      }
      state.pool = pool;
      renderPool(pool);
    }).catch(function () { /* transient: keep showing the last good data */ });
    getJSON(galleryURL(active)).then(function (gallery) {
      state.gallery = (gallery && gallery.levels && gallery.levels.length) ? gallery : null;
      renderGallery(state.gallery);
    }).catch(function () { /* transient: keep showing the last good data */ });
  }).catch(function () { /* server hiccup: try again next tick */ });
}

if (STATIC) document.getElementById("footerPoll").textContent = "static copy, polls every 15 s";
poll();
setInterval(poll, POLL_MS);
window.addEventListener("resize", function () {
  lossChart.draw(); metricChart.draw(); poolStatsChart.draw();
  if (state.pool) renderPool(state.pool);
  if (state.gallery) renderGallery(state.gallery);
});
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
