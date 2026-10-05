"""Is a training run making progress? One verdict from runs/<name>/log.jsonl alone, for a person or an agent
with 30 seconds.

    python -m nca.progress runs/pure-a              # a short block, then one VERDICT line
    python -m nca.progress runs/pure-a --json       # the same as one JSON object
    python -m nca.progress runs/pure-a --window 5   # K quick checks per window (default 3)
    python -m nca.progress --selftest               # synthetic logs, one per verdict

The block: iteration / target, the recent rate (iterations per wall-clock second, quick checks and checkpoints
included; logs from before the trainer wrote "time" estimate it from secPerIter + evalSec), the ETA, how long
ago the last line was written (its "time", else the file's mtime), the loss now vs earlier, and each quick-check
metric's mean over the last K checks against the K before: exact on the fresh mix / bridge / page sets, the
"none" (lower is better) and "both" shares on bridge boards, each step of the edit sequence, and the score.

A change counts as real only beyond 2 standard errors. The quick check reuses the SAME boards every time, so
averaging K checks does not shrink their sampling noise: the error of a difference of two windows is the
binomial one of a single check, sqrt(2 p (1 - p) / n), n = evalN boards per set per radius x the radii (for
the score, the mean of four such shares, the four errors combined), or the check-to-check scatter within the
windows if that is larger (training noise).

Then one line, VERDICT: <word> - reason, the first rule that matches:
  DIVERGED  the trainer's "non-finite steps in a row" stop, a non-finite (or all-skipped) loss in the recent
            window, the skipped-step guard firing in the recent window, or the loss more than 2x the window
            before
  FINISHED  a clean end: the trainer's last line {"stopped": "done"} (--iters reached) or {"stopped": "time"}
            (a --minutes chunk or time-boxed stage; --resume carries on), or the target iteration reached
  STALLED   no new line for more than max(10 min, 5x the usual gap between lines)
  WARMUP    fewer than 2K quick checks so far
  PLATEAU   no metric improved beyond noise over the last 2K checks, and the loss fell less than 3%
  PROGRESS  anything else
Exit code: 0 PROGRESS / WARMUP / FINISHED, 3 PLATEAU, 4 STALLED, 5 DIVERGED, 1 no log.jsonl to read.

Stdlib only (no torch, no numpy): quick to start, and safe to run next to a training run.
"""

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time

EXIT = {"PROGRESS": 0, "WARMUP": 0, "FINISHED": 0, "PLATEAU": 3, "STALLED": 4, "DIVERGED": 5}
EVAL_N = 32            # nca.train.EVAL_N: boards per set per radius when a log doesn't say
STALL_MIN_SEC = 600    # STALLED: no line for longer than this, or STALL_GAPS x the usual gap if longer
STALL_GAPS = 5
PLATEAU_LOSS = 0.03    # PLATEAU needs the loss to have fallen less than this (3%)
DIVERGE_RISE = 2.0     # DIVERGED: the recent loss window this many times the one before
Z = 2.0                # a change is real beyond Z standard errors
# (name, +1 if higher is better / -1 if lower is) in display order; editN come from the "edit" list
METRICS = (("mix", 1), ("bridge", 1), ("page", 1), ("none", -1), ("both", 1), ("exact", 1))


def read_log(path):
    """[{"start": the (re)start line or None, "lines": [records and stop lines]}] in file order. A line that
    doesn't parse (a write in progress) is skipped; lines without "iteration" start a session."""
    sessions = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            if "iteration" not in obj:  # the trainer's (re)start line: {"config": {...}, ...} (or a flat config)
                sessions.append({"start": obj, "lines": []})
                continue
            if not sessions:
                sessions.append({"start": None, "lines": []})  # logs from before the config line
            sessions[-1]["lines"].append(obj)
    return sessions


def metrics_of(rec):
    """{metric: value} of a quick-check record ({} if it is not one): from its first q<m> dict (the read-out the
    score uses), its edit sequence, its score, and a very old log's top-level "exact"."""
    out = {}
    q = next((v for k, v in rec.items() if k[:1] == "q" and k[1:].isdigit() and isinstance(v, dict)), None)
    for name, _ in METRICS:
        src = q if name != "exact" else rec
        if src is not None and isinstance(src.get(name), (int, float)):
            out[name] = float(src[name])
    for i, e in enumerate(rec.get("edit") or []):
        if isinstance(e, (int, float)):
            out[f"edit{i + 1}"] = float(e)
    if isinstance(rec.get("score"), (int, float)):
        out["score"] = float(rec["score"])
    return out


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt_dur(sec):
    if sec is None:
        return "?"
    sec = int(round(sec))
    d, h, m, s = sec // 86400, sec // 3600 % 24, sec // 60 % 60, sec % 60
    return f"{d}d{h:02d}h" if d else f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def analyse(run_dir, window=3, now=None):
    """Everything the verdict rests on, as a dict (also the --json output)."""
    path = run_dir if run_dir.endswith(".jsonl") else os.path.join(run_dir, "log.jsonl")
    now = time.time() if now is None else now
    sessions = read_log(path)
    K = max(1, window)

    # Records in order, a resume dropping what came after its checkpoint (those iterations ran again).
    records, last_start, last_line = [], None, None
    for si, s in enumerate(sessions):
        if s["start"] is not None:
            last_start = s["start"]
            s0 = s["start"].get("startIteration", 0) or 0
            records = [r for r in records if s0 and r["iteration"] <= s0]
            last_line = s["start"]
        for line in s["lines"]:
            last_line = line
            if "stopped" not in line and "snapshotError" not in line:
                records.append(dict(line, _session=si))
    stop = last_line if last_line is not None and "stopped" in last_line else None
    cfg = (last_start or {}).get("config") or last_start or {}  # nested (the trainer) or flat (older, the demo)
    target = cfg.get("iters")
    it_now = max([r["iteration"] for r in records] + [(last_line or {}).get("iteration", 0) or 0], default=0)

    # When the last line was written; the usual gap between lines; the rate.
    last_time = (last_line or {}).get("time")
    age_basis = "time"
    if not finite(last_time):
        last_time, age_basis = os.path.getmtime(path), "mtime"
    age = now - last_time
    gaps_t, gaps_est = [], []
    for si, s in enumerate(sessions):
        prev_it = (s["start"] or {}).get("startIteration")
        prev_t = None
        for line in s["lines"]:
            if "stopped" in line:
                continue
            t = line.get("time")
            if finite(t) and finite(prev_t):
                gaps_t.append(t - prev_t)
            prev_t = t
            d_it = line["iteration"] - prev_it if isinstance(prev_it, int) else 50
            if finite(line.get("secPerIter")) and d_it > 0:
                gaps_est.append(line["secPerIter"] * d_it + (line.get("evalSec") or 0))
            prev_it = line["iteration"]
    usual = statistics.median(gaps_t[-20:]) if gaps_t else statistics.median(gaps_est[-20:]) if gaps_est else None
    stall_after = max(STALL_MIN_SEC, STALL_GAPS * usual) if usual else STALL_MIN_SEC

    rate, rate_basis = None, None
    last_s = next((s for s in reversed(sessions) if any("stopped" not in x for x in s["lines"])), None)
    if last_s is not None:
        pts = [(x["iteration"], x["time"]) for x in last_s["lines"] if "stopped" not in x and finite(x.get("time"))]
        st = last_s["start"] or {}
        if len(pts) < 2 and finite(st.get("time")) and isinstance(st.get("startIteration"), int):
            pts.insert(0, (st["startIteration"], st["time"]))  # the start: includes setup, but better than none
        pts = pts[-21:]
        if len(pts) >= 2 and pts[-1][1] > pts[0][1] and pts[-1][0] > pts[0][0]:
            rate, rate_basis = (pts[-1][0] - pts[0][0]) / (pts[-1][1] - pts[0][1]), "wall clock"
        else:
            recs = [x for x in last_s["lines"] if "stopped" not in x][-20:]
            prev, d_it, d_t = (last_s["start"] or {}).get("startIteration"), 0, 0.0
            for x in recs:
                d = x["iteration"] - prev if isinstance(prev, int) else 50
                if finite(x.get("secPerIter")) and d > 0:
                    d_it, d_t = d_it + d, d_t + x["secPerIter"] * d + (x.get("evalSec") or 0)
                prev = x["iteration"]
            if d_t > 0:
                rate, rate_basis = d_it / d_t, "secPerIter + evalSec"
    eta = (target - it_now) / rate if rate and target and it_now < target else None

    # Elapsed training time (the sessions' spans): what a time-boxed stage has used.
    elapsed = 0.0
    for s in sessions:
        ts = [x.get("time") for x in ([s["start"]] if s["start"] else []) + s["lines"]]
        ts = [t for t in ts if finite(t)]
        if len(ts) >= 2:
            elapsed += ts[-1] - ts[0]

    # Quick checks: the last K against the K before.
    checks = [(r["iteration"], metrics_of(r), r) for r in records]
    checks = [c for c in checks if c[1]]
    last_check = checks[-1][2] if checks else {}
    eval_n = last_check.get("evalN") or (last_start or {}).get("evalN") or cfg.get("evalN") or EVAL_N
    q = next((v for k, v in last_check.items() if k[:1] == "q" and k[1:].isdigit() and isinstance(v, dict)), {})
    n_radii = len(cfg.get("R") or q.get("bridgeByR") or last_check.get("editByR") or [0])
    boards = eval_n * n_radii
    recent, before = checks[-K:], checks[-2 * K:-K] if len(checks) > K else []
    names = [n for n, _ in METRICS] + sorted({k for c in checks for k in c[1] if k.startswith("edit")}) + ["score"]
    sign = dict(METRICS)
    table = {}
    for name in names:
        r = [c[1][name] for c in recent if name in c[1]]
        b = [c[1][name] for c in before if name in c[1]]
        if not r:
            continue
        row = {"recent": mean(r), "before": mean(b), "better": "lower" if sign.get(name) == -1 else "higher"}
        if b:
            p = min(max((row["recent"] + row["before"]) / 2, 1 / boards), 1 - 1 / boards)
            row["seBinomial"] = math.sqrt(2 * p * (1 - p) / boards)
            spread = [statistics.pvariance(x) / len(x) for x in (r, b) if len(x) > 1]
            row["seScatter"] = math.sqrt(sum(spread)) if len(spread) == 2 else 0.0
            row["change"] = (row["recent"] - row["before"]) * sign.get(name, 1)  # > 0 = better
        table[name] = row
    if "score" in table and "seBinomial" in table["score"]:  # the score is the mean of 4 shares
        parts = [table[n]["seBinomial"] for n in ("mix", "bridge", "page") if "seBinomial" in table.get(n, {})]
        edits = [table[n]["seBinomial"] for n in table if n.startswith("edit") and "seBinomial" in table[n]]
        if edits:
            parts.append(mean(edits))  # the 3 edit steps share boards: no averaging-down assumed
        if parts:
            table["score"]["seBinomial"] = math.sqrt(sum(x * x for x in parts)) / 4
    for row in table.values():
        if "change" in row:
            row["se"] = max(row["seBinomial"], row["seScatter"])
            row["improved"] = row["change"] > Z * row["se"]
            row["worse"] = row["change"] < -Z * row["se"]

    # Loss windows: as many loss lines as the last K checks span (4K if there aren't K+1 checks yet).
    loss_recs = [r for r in records if "loss" in r]
    if len(checks) > K:
        w = sum(1 for r in loss_recs if checks[-K - 1][0] < r["iteration"] <= checks[-1][0])
    else:
        w = 4 * K
    w = max(2, w)
    lr_recent, lr_before = loss_recs[-w:], loss_recs[-2 * w:-w]
    vals = lambda rs: [r["loss"] for r in rs if finite(r.get("loss"))]
    loss = {"recent": mean(vals(lr_recent)), "before": mean(vals(lr_before)), "window": w,
            "recentIters": [lr_recent[0]["iteration"], lr_recent[-1]["iteration"]] if lr_recent else None,
            "beforeIters": [lr_before[0]["iteration"], lr_before[-1]["iteration"]] if lr_before else None,
            "first": next((r["loss"] for r in loss_recs if finite(r.get("loss"))), None)}
    if loss["recent"] is not None and loss["before"]:
        loss["fall"] = (loss["before"] - loss["recent"]) / loss["before"]
    bad_loss = [r["iteration"] for r in lr_recent if not finite(r.get("loss"))]
    fired = 0  # skipped steps logged in the recent window (the count is per session, cumulative)
    if lr_recent:
        lo = lr_recent[0]["iteration"]
        for si in {r["_session"] for r in lr_recent}:
            inside = [r.get("skipped", 0) or 0 for r in records if r["_session"] == si and r["iteration"] >= lo]
            earlier = [r.get("skipped", 0) or 0 for r in records if r["_session"] == si and r["iteration"] < lo]
            fired += max(inside, default=0) - (earlier[-1] if earlier else 0)

    info = {"run": run_dir, "iteration": it_now, "target": target, "device": (last_line or {}).get("device"),
            "rate": rate, "rateBasis": rate_basis, "etaSec": eta, "ageSec": age, "ageBasis": age_basis,
            "usualGapSec": usual, "stallAfterSec": stall_after, "elapsedMin": elapsed / 60,
            "sessions": len(sessions), "checks": len(checks), "window": K, "evalN": eval_n, "boards": boards,
            "radii": cfg.get("R"), "recentChecks": [c[0] for c in recent], "beforeChecks": [c[0] for c in before],
            "metrics": table, "loss": loss, "nonFiniteLoss": bad_loss, "skippedRecent": fired,
            "stopped": stop.get("stopped") if stop else None}
    info["verdict"], info["reason"] = verdict(info, K)
    info["exitCode"] = EXIT[info["verdict"]]
    return info


def verdict(info, K):
    """(word, reason) from analyse()'s numbers; the first rule that matches (see the module docstring)."""
    it, target, loss, table = info["iteration"], info["target"], info["loss"], info["metrics"]
    of = f"{it}/{target}" if target else f"{it} (target unknown)"
    stopped = info["stopped"] or ""
    if stopped and stopped not in ("time", "done"):
        return "DIVERGED", f"the trainer stopped: {stopped} at iteration {it}"
    if info["nonFiniteLoss"]:
        return "DIVERGED", (f"non-finite or all-skipped loss at iteration(s) {info['nonFiniteLoss'][:5]} "
                            "(the skipped-step guard)")
    if info["skippedRecent"] > 0:
        return "DIVERGED", f"the skipped-step guard fired {info['skippedRecent']}x in the recent window"
    if stopped == "done":
        return "FINISHED", f"reached the target: iteration {of}"
    if stopped == "time":
        return "FINISHED", (f"stopped cleanly at its --minutes limit at iteration {of} (a chunk or a time-boxed "
                            "stage; --resume carries on)")
    if target and it >= target:
        return "FINISHED", f"reached the target: iteration {of}"
    if info["ageSec"] > info["stallAfterSec"]:
        return "STALLED", (f"no new line for {fmt_dur(info['ageSec'])} (usual gap {fmt_dur(info['usualGapSec'])}, "
                           f"limit {fmt_dur(info['stallAfterSec'])}; from the {info['ageBasis']}) at iteration {of}")
    if loss["recent"] is not None and loss["before"] and loss["recent"] > DIVERGE_RISE * loss["before"]:
        return "DIVERGED", (f"loss {loss['recent']:.4g} is {loss['recent'] / loss['before']:.1f}x the window before "
                            f"({loss['before']:.4g})")
    if info["checks"] < 2 * K:
        return "WARMUP", f"{info['checks']} quick check(s) so far; a verdict needs 2K = {2 * K}"
    up = sorted((r["change"] / max(r["se"], 1e-9), n, r) for n, r in table.items() if r.get("improved"))[::-1]
    up = [f"{n} {r['change']:+.3f} (2se {Z * r['se']:.3f})" for _, n, r in up[:3]] + \
        ([f"{len(up) - 3} more up"] if len(up) > 3 else [])
    fall = loss.get("fall")
    loss_txt = f"loss {-fall * 100 + 0.0:+.1f}%" if fall is not None else "loss n/a"
    if not up and (fall is None or fall < PLATEAU_LOSS):
        best = max((r for r in table.values() if "change" in r), key=lambda r: r["change"] / max(r["se"], 1e-9),
                   default=None)
        name = next((n for n, r in table.items() if r is best), None)
        near = f"; closest: {name} {best['change']:+.3f} vs 2se {Z * best['se']:.3f}" if best else ""
        return "PLATEAU", (f"no metric improved beyond noise over the last {2 * K} checks "
                           f"(iterations {info['beforeChecks'][0]}-{info['recentChecks'][-1]}){near}; {loss_txt} "
                           f"(needs a {PLATEAU_LOSS * 100:.0f}% fall)")
    down = [f"{n} {r['change']:+.3f}" for n, r in table.items() if r.get("worse")]
    return "PROGRESS", ("; ".join(up) if up else "no metric beyond noise yet") + f"; {loss_txt}" + \
        (f"; worse: {', '.join(down)}" if down else "")


def render(info):
    """The human block (ends with the VERDICT line)."""
    f3 = lambda x: "  -   " if x is None else f"{x:.3f}"
    pct = f" ({100 * info['iteration'] / info['target']:.1f}%)" if info["target"] else ""
    rate = f"{info['rate']:.3g} it/s ({info['rateBasis']})" if info["rate"] else "rate ?"
    out = [f"{info['run']}  device {info['device'] or '?'}  radii {info['radii'] or '?'}  evalN {info['evalN']} "
           f"({info['boards']} boards per set)  {info['sessions']} session(s), {info['elapsedMin']:.0f} min logged",
           f"iteration  {info['iteration']} / {info['target'] or '?'}{pct}   {rate}   ETA {fmt_dur(info['etaSec'])}",
           f"last line  {fmt_dur(info['ageSec'])} ago ({info['ageBasis']}; usual gap {fmt_dur(info['usualGapSec'])}, "
           f"stalled after {fmt_dur(info['stallAfterSec'])})"]
    L = info["loss"]
    if L["recent"] is not None:
        line = f"loss       {L['recent']:.4g} (last {L['window']} lines, it {L['recentIters'][0]}-{L['recentIters'][1]})"
        if L["before"] is not None:
            line += (f"  vs {L['before']:.4g} (it {L['beforeIters'][0]}-{L['beforeIters'][1]})  "
                     f"{-L['fall'] * 100 + 0.0:+.1f}%")
        out.append(line + (f"   first {L['first']:.4g}" if L["first"] is not None else ""))
    if info["metrics"]:
        rc, bc = info["recentChecks"], info["beforeChecks"]
        out.append(f"checks     {info['checks']} so far; recent = it {rc[0]}-{rc[-1]} ({len(rc)})"
                   + (f" vs it {bc[0]}-{bc[-1]} ({len(bc)})" if bc else ""))
        out.append(f"  {'metric':<12}{'recent':>8}{'before':>8}{'change':>8}{'2se':>7}")
        for name, r in info["metrics"].items():
            label = name + (" (low)" if r["better"] == "lower" else "")
            mark = " up" if r.get("improved") else " DOWN" if r.get("worse") else ""
            out.append(f"  {label:<12}{f3(r['recent']):>8}{f3(r['before']):>8}"
                       f"{(format(r['change'], '+.3f') if 'change' in r else '  -  '):>8}"
                       f"{(format(Z * r['se'], '.3f') if 'se' in r else '  - '):>7}{mark}")
    out.append(f"VERDICT: {info['verdict']} - {info['reason']}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Self-test: synthetic logs, one per verdict (python -m nca.progress --selftest; nca.selftest runs it too).
# --------------------------------------------------------------------------

def _synth(path, n_checks, every=200, t0=1.0e9, gap=60.0, score=lambda k: 0.5, loss=lambda i: 0.05,
           iters=20000, timed=True, extra=None, stop=None, start_extra=None):
    """Write a trainer-shaped log: a config line, a line per 50 iterations, a quick check every `every`."""
    lines = [{"config": {"R": [4, 5, 6], "iters": iters}, "startIteration": 0, "evalN": 32, **(start_extra or {})}]
    t = t0
    if timed:
        lines[0]["time"] = t
    it, k = 0, 0
    while k < n_checks:
        rec = {"iteration": it}
        if it:
            rec.update(loss=loss(it), secPerIter=gap / 50)
        if it % every == 0:
            s = score(k)
            rec.update(q8={"mix": s, "bridge": s, "page": s, "none": 1 - s, "both": s, "bridgeByR": {}},
                       edit=[s, s, s], score=s, evalN=32)
            k += 1
        rec.update((extra or {}).get(it, {}))
        if timed:
            t += gap
            rec["time"] = t
        lines.append(rec)
        it += 50
    if stop:
        lines.append({"iteration": lines[-1]["iteration"], "stopped": stop, **({"time": t + 1} if timed else {})})
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n")
    return t


def selftest():
    def check(cond, msg):
        if not cond:
            raise AssertionError(msg)
        print(f"OK {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        def run(name, now_after=30.0, **kw):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            t = _synth(os.path.join(d, "log.jsonl"), **kw)
            return analyse(d, 3, now=t + now_after)

        rise = lambda k: 0.2 + 0.08 * k
        a = run("progress", n_checks=10, score=rise, loss=lambda i: 0.1 / (1 + i / 500))
        check(a["verdict"] == "PROGRESS" and a["metrics"]["mix"]["improved"] and a["exitCode"] == 0,
              f"progress: rising exact shares and a falling loss -> PROGRESS ({a['reason']})")
        check(abs(a["rate"] - 50 / 60) < 1e-9 and a["rateBasis"] == "wall clock" and abs(a["ageSec"] - 30) < 1e-6,
              "progress: the rate is iterations per wall-clock second from the lines' time; the age from the last")
        a = run("warmup", n_checks=4, score=rise)
        check(a["verdict"] == "WARMUP" and a["exitCode"] == 0, f"warmup: 4 checks < 2K = 6 -> WARMUP ({a['reason']})")
        a = run("plateau", n_checks=10, score=lambda k: 0.5 + 0.01 * (k % 2), loss=lambda i: 0.05)
        check(a["verdict"] == "PLATEAU" and a["exitCode"] == 3,
              f"plateau: flat shares (+-0.01 < 2se) and a flat loss -> PLATEAU ({a['reason']})")
        a = run("plateau-loss", n_checks=10, score=lambda k: 0.5, loss=lambda i: 0.1 / (1 + i / 200))
        check(a["verdict"] == "PROGRESS", f"flat shares but the loss still falling 3%+ -> PROGRESS ({a['reason']})")
        a = run("noisy", n_checks=10, score=lambda k: 0.45 + 0.05 * k / 9, loss=lambda i: 0.05)
        check(a["verdict"] == "PLATEAU", f"a +0.03 drift with 96 boards is within 2se -> PLATEAU ({a['reason']})")
        a = run("stalled", n_checks=10, score=rise, now_after=3600)
        check(a["verdict"] == "STALLED" and a["exitCode"] == 4, f"stalled: no line for 1 h -> STALLED ({a['reason']})")
        a = run("slow", n_checks=10, score=rise, gap=200.0, now_after=900)
        check(a["verdict"] == "PROGRESS", "...but 15 min with a usual gap of 200 s (limit 5x = 16m40s) is not stalled")
        a = run("diverged-loss", n_checks=10, score=rise, loss=lambda i: 0.05 if i < 1200 else 0.2)
        check(a["verdict"] == "DIVERGED" and a["exitCode"] == 5, f"loss up 4x -> DIVERGED ({a['reason']})")
        a = run("diverged-skip", n_checks=10, score=rise, extra={1700: {"skipped": 3}, 1750: {"skipped": 5}})
        check(a["verdict"] == "DIVERGED", f"skipped steps in the recent window -> DIVERGED ({a['reason']})")
        a = run("diverged-old-skip", n_checks=10, score=rise, extra={100: {"skipped": 1}, 150: {"skipped": 1}})
        check(a["verdict"] == "PROGRESS", "...but one skip long ago (the count stays 1) is not")
        a = run("diverged-nan", n_checks=10, score=rise, extra={1800: {"loss": None, "skipped": 50}})
        check(a["verdict"] == "DIVERGED", f"a window with every step skipped (loss null) -> DIVERGED ({a['reason']})")
        a = run("diverged-stop", n_checks=10, score=rise, stop="20 non-finite steps in a row", now_after=7200)
        check(a["verdict"] == "DIVERGED", f"the trainer's non-finite stop line -> DIVERGED, even when old ({a['reason']})")
        a = run("finished-time", n_checks=10, score=rise, stop="time", now_after=86400)
        check(a["verdict"] == "FINISHED" and a["exitCode"] == 0, f"a --minutes stop line -> FINISHED ({a['reason']})")
        a = run("finished-iters", n_checks=10, score=rise, iters=1800, now_after=86400)
        check(a["verdict"] == "FINISHED", f"last iteration = target -> FINISHED, however old ({a['reason']})")

        # An old log (no "time"): the age comes from the file's mtime, the rate from secPerIter + evalSec.
        d = os.path.join(tmp, "untimed")
        os.makedirs(d)
        _synth(os.path.join(d, "log.jsonl"), n_checks=10, score=rise, timed=False)
        os.utime(os.path.join(d, "log.jsonl"), (2e9, 2e9))
        a = analyse(d, 3, now=2e9 + 120)
        check(a["ageBasis"] == "mtime" and abs(a["ageSec"] - 120) < 1e-6 and a["rateBasis"] == "secPerIter + evalSec"
              and a["verdict"] == "PROGRESS", "an untimed log: age from the mtime, rate from secPerIter")
        a = analyse(d, 3, now=2e9 + 3600)
        check(a["verdict"] == "STALLED", "...and an hour-old mtime -> STALLED")

        # A resume from an earlier checkpoint: the iterations after it ran again and replace the old lines.
        d = os.path.join(tmp, "rewind")
        os.makedirs(d)
        p = os.path.join(d, "log.jsonl")
        t = _synth(p, n_checks=6, score=lambda k: 0.9, loss=lambda i: 0.05)  # up to iteration 1000, score 0.9
        with open(p, "a") as f:  # resumed from the checkpoint at 800, now a different score
            f.write(json.dumps({"config": {"R": [4, 5, 6], "iters": 20000}, "startIteration": 800, "time": t + 10}) + "\n")
            for i, it in enumerate(range(850, 1250, 50)):
                rec = {"iteration": it, "loss": 0.05, "time": t + 70 + 60 * i}
                if it % 200 == 0:
                    rec.update(q8={"mix": 0.1}, score=0.1)
                f.write(json.dumps(rec) + "\n")
        a = analyse(d, 3, now=t + 600)
        its = a["recentChecks"] + a["beforeChecks"]
        check(a["sessions"] == 2 and a["checks"] == 7 and a["recentChecks"] == [800, 1000, 1200]
              and a["metrics"]["score"]["recent"] < 0.5 and len(its) == len(set(its)),
              "a resume from iteration 800 replaces the lines after it (no check counted twice)")
        check(abs(a["elapsedMin"] - ((t - 1e9) + 480) / 60) < 0.05,
              "elapsedMin sums the sessions' spans, not the gap between them")
    print("nca.progress self-test passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", help="runs/<name> (or a log.jsonl)")
    p.add_argument("--json", action="store_true", help="one JSON object instead of the block")
    p.add_argument("--window", type=int, default=3, help="K: quick checks per window (default 3)")
    p.add_argument("--now", type=float, default=None, help=argparse.SUPPRESS)  # tests: pretend it is this time
    p.add_argument("--selftest", action="store_true", help="check the verdicts on synthetic logs")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.run_dir:
        p.error("RUN_DIR is required")
    path = args.run_dir if args.run_dir.endswith(".jsonl") else os.path.join(args.run_dir, "log.jsonl")
    if not os.path.isfile(path):
        print(f"no log at {path}", file=sys.stderr)
        return 1
    info = analyse(args.run_dir, args.window, args.now)
    print(json.dumps(info) if args.json else render(info))
    return info["exitCode"]


if __name__ == "__main__":
    sys.exit(main())
