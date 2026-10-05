#!/bin/bash
# hex_ca cloud trainer: the boot script of the one Spot VM (launch.sh passes it as the startup-script).
#
# Compute Engine runs it as root on EVERY boot (also after the driver installer's reboot), so it is
# re-entrant: each step checks whether it is already done. Called with no argument (by the guest agent) it
# hands the work to a systemd unit, hexca-main (so nothing depends on how long a startup script may run),
# and exits. The unit runs this same file with "main":
#   1. the NVIDIA driver if missing (Google's cuda_installer; it may reboot; after 3 tries, give up)
#   2. a venv with torch + numpy; give up if torch sees no GPU (a GPU VM training on its CPU is money burnt)
#   3. the repo at the launched commit; runs/ from the bucket (what a preempted VM had done)
#   4. the PLAN (metadata "plan"), one stage per line:   run-name | stage-name | minutes | train args
#      Different run-names run IN PARALLEL (a process each, sharing the GPU); a run's stages one after
#      another, each in runs/<run>-<stage> with --minutes = what is left of its minutes: --resume if its
#      ckpt.pt exists, else --init from the previous stage's ckpt.pt (unless its args name an --init; an
#      arg bucket:PATH is fetched from the bucket first, e.g. --init bucket:init/pure-a.pt). A stage whose
#      log ended {"stopped": "done"}, or {"stopped": "time"} with under 2 of its minutes left, is skipped (so
#      a finished stage dir of the same name already in the bucket counts as done; a preempted one resumes).
#      A stage that fails ends its run.
#   5. a sidecar, every 60 s: the whitelisted run files (log.jsonl pool.npz ckpt.pt best.pt stdout.log)
#      to $BUCKET/runs/, status.json (the heartbeat), the tail of this log, and the static dashboard to
#      $BUCKET/dash/ (publish.sh --source vm). The bucket is PUBLIC: nothing else ever goes up.
#   6. at the end: a last sync, DONE (JSON) in the bucket, poweroff. The service account can only use the
#      bucket, so the VM cannot delete itself: down.sh does that from outside, and --max-run-duration with
#      --instance-termination-action=DELETE deletes it anyway when the time is up.
# A shutdown (a Spot preemption gives ~30 s) stops the unit: its TERM trap syncs once more, and so does
# shutdown.sh (the shutdown-script), which runs this file with "sync".
#
# Local test (no GCE, no root, never powers anything off; nca/cloud/selftest.sh does this):
#   HEXCA_TEST=1 W=dir BUCKET=dir PLAN_FILE=file SRC=hex_ca-checkout PY=python bash startup.sh main
set -uo pipefail
export PATH=$PATH:/snap/bin   # gcloud on Ubuntu's GCE images; systemd units don't have it on PATH
WHITELIST="log.jsonl pool.npz ckpt.pt best.pt stdout.log"   # the only run files that ever leave the VM

md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
on_gce() { grep -qs 'Google Compute Engine' /sys/class/dmi/id/product_name; }
TEST=${HEXCA_TEST:-}
if [ -n "$TEST" ]; then
  : "${W:?}" "${BUCKET:?}" "${PLAN_FILE:?}" "${SRC:?}" "${PY:?}"
  PROJECT=${PROJECT:-test} NAME=${NAME:-local-test} LAUNCH=${LAUNCH:-test} COMMIT=local SYNC_SEC=${SYNC_SEC:-60}
  PLAN=$(cat "$PLAN_FILE")
else
  W=/opt/hexca SYNC_SEC=60
  BUCKET=$(md instance/attributes/bucket) COMMIT=$(md instance/attributes/commit)
  LAUNCH=$(md instance/attributes/launch) REPO=$(md instance/attributes/repo)
  TORCH=$(md instance/attributes/torch || echo torch) PLAN=$(md instance/attributes/plan)
  PROJECT=$(md project/project-id) NAME=$(md instance/name)
  PY=$W/venv/bin/python
  if [ -z "$BUCKET" ] || ! on_gce; then echo "not on Compute Engine (no metadata); see the local test above" >&2; exit 1; fi
fi
CODE=$W/Hex_CA STATE=$W/state OUT=$W/out
mkdir -p "$W" "$STATE"

if [ "${1:-}" != main ] && [ "${1:-}" != sync ]; then  # the guest agent: start the unit (once per boot)
  md instance/attributes/startup-script >"$W/main.sh"
  systemctl is-active --quiet hexca-main || systemd-run --unit=hexca-main --collect -p TimeoutStopSec=60 \
    /bin/bash "$W/main.sh" main
  exit 0
fi

log() { echo "$(date -u +%FT%TZ) $*"; }
phase() { echo "$*" >"$W/phase"; log "phase: $*"; }

# ---- the bucket: gs:// through gcloud, or a local directory (the test) ----
put() {  # put FILE REL [content-type]
  if [[ $BUCKET == gs://* ]]; then
    gcloud storage cp --project="$PROJECT" --quiet ${3:+--content-type="$3"} "$1" "$BUCKET/$2" >/dev/null
  else mkdir -p "$(dirname "$BUCKET/$2")" && cp "$1" "$BUCKET/$2"; fi
}
get() {  # get REL FILE
  mkdir -p "$(dirname "$2")"
  if [[ $BUCKET == gs://* ]]; then gcloud storage cp --project="$PROJECT" --quiet "$BUCKET/$1" "$2" >/dev/null
  else cp "$BUCKET/$1" "$2"; fi
}
sync_up() {  # sync_up DIR REL: add and update, never delete
  if [[ $BUCKET == gs://* ]]; then gcloud storage rsync -r --project="$PROJECT" --quiet "$1" "$BUCKET/$2" >/dev/null
  else mkdir -p "$BUCKET/$2" && rsync -a "$1/" "$BUCKET/$2/"; fi
}
sync_down() {  # sync_down REL DIR
  mkdir -p "$2"
  if [[ $BUCKET == gs://* ]]; then gcloud storage rsync -r --project="$PROJECT" --quiet "$BUCKET/$1" "$2" >/dev/null 2>&1
  else [ ! -d "$BUCKET/$1" ] || rsync -a "$BUCKET/$1/" "$2/"; fi
}

# ---- the plan ----
plan_lines() {  # run|stage|minutes|args, comments and blank lines dropped, fields trimmed
  printf '%s\n' "$PLAN" | sed 's/#.*//' | awk -F'|' 'NF >= 4 {
    for (i = 1; i <= NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i)
    a = $4; for (i = 5; i <= NF; i++) a = a "|" $i; print $1 "|" $2 "|" $3 "|" a }'
}
plan_runs() { plan_lines | cut -d'|' -f1 | awk '!seen[$0]++'; }
plan_dirs() { plan_lines | awk -F'|' '{ print $1 "-" $2 }'; }
set_state() {  # set_state RUN STAGE INDEX COUNT STATE  (names are [a-z0-9-]: no JSON escaping needed)
  printf '{"stage": "%s", "stageIndex": %s, "stages": %s, "dir": "%s-%s", "state": "%s", "since": %s}\n' \
    "$2" "$3" "$4" "$1" "$2" "$5" "$(date +%s)" >"$STATE/$1.tmp" && mv "$STATE/$1.tmp" "$STATE/$1"
}
progress() {  # progress DIR -> "stopped elapsedMin verdict" of runs/DIR ("- 0 NONE" without a log)
  (cd "$CODE" && "$PY" -m nca.progress "runs/$1" --json 2>/dev/null) | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
print(d["stopped"] or "-", round(d["elapsedMin"], 2), d["verdict"])' 2>/dev/null || echo "- 0 NONE"
}

run_stages() {  # run_stages RUN THREADS: the run's stages, in order
  local run=$1 threads=$2 prev="" i=0 n r stage last="" minutes args dir stopped used verdict left a rel
  local -a argv mode
  n=$(plan_lines | awk -F'|' -v r="$run" '$1 == r' | wc -l)
  while IFS='|' read -r r stage minutes args; do
    [ "$r" = "$run" ] || continue
    i=$((i + 1)) dir=$run-$stage last=$stage
    read -r -a argv <<<"$args"
    for k in "${!argv[@]}"; do  # bucket:PATH -> a local copy of $BUCKET/PATH
      a=${argv[$k]}
      [[ $a == bucket:* ]] || continue
      rel=${a#bucket:}
      [ -f "$W/bucket/$rel" ] || get "$rel" "$W/bucket/$rel" || { log "$dir: cannot fetch $a"; set_state "$run" "$stage" "$i" "$n" failed; return 1; }
      argv[k]=$W/bucket/$rel
    done
    read -r stopped used verdict < <(progress "$dir")
    left=$(awk -v m="$minutes" -v u="$used" 'BEGIN { l = m - u; printf "%.2f", (l > 0 ? l : 0) }')
    # done: its --iters reached; time: its time box ran out (a remainder under 2 min isn't worth a restart);
    # no stop line (preempted mid-stage): resume with whatever is left
    if [ "$stopped" = "done" ] || awk -v s="$stopped" -v l="$left" 'BEGIN { exit !(l < 0.2 || (s == "time" && l < 2)) }'; then
      log "$dir: already ended ($stopped, $used of $minutes min used, $verdict): next stage"
      prev=$dir
      continue
    fi
    mode=()
    if [ -f "$CODE/runs/$dir/ckpt.pt" ]; then mode=(--resume)
    elif [[ " ${argv[*]} " != *" --init "* ]] && [ -n "$prev" ]; then mode=(--init "runs/$prev/ckpt.pt"); fi
    [[ " ${argv[*]} " == *" --threads "* ]] || argv+=(--threads "$threads")
    set_state "$run" "$stage" "$i" "$n" running
    mkdir -p "$CODE/runs/$dir"
    log "$dir: python -m nca.train --name $dir ${argv[*]} ${mode[*]} --minutes $left"
    (cd "$CODE" && "$PY" -m nca.train --name "$dir" "${argv[@]}" "${mode[@]}" --minutes "$left") \
      >>"$CODE/runs/$dir/stdout.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      log "$dir: nca.train exited $rc; this run stops here (see runs/$dir/stdout.log)"
      set_state "$run" "$stage" "$i" "$n" failed
      return 1
    fi
    prev=$dir
  done < <(plan_lines)
  set_state "$run" "$last" "$i" "$n" "done"
}

# ---- the heartbeat and the copies ----
status_json() {  # status.json on stdout: time, phase, GPU, load, each run's stage and nca.progress verdict
  local py=$PY
  [ -x "$py" ] || py=python3
  "$py" - "$W" "$CODE" "$NAME" "$LAUNCH" "$COMMIT" <<'PYEOF'
import json, os, subprocess, sys, time
W, code, name, launch, commit = sys.argv[1:6]
def read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default
gpu = None
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if q.returncode == 0 and q.stdout.strip():
        n, u, mu, mt, t = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        gpu = {"name": n, "util": float(u), "memUsedMB": float(mu), "memTotalMB": float(mt), "tempC": float(t)}
except Exception:
    pass
runs = {}
sd = os.path.join(W, "state")
for f in sorted(os.listdir(sd)) if os.path.isdir(sd) else []:
    if f.endswith(".tmp"):
        continue
    try:
        st = json.loads(read(os.path.join(sd, f)))
    except ValueError:
        continue
    d = os.path.join("runs", st.get("dir", ""))
    if os.path.isfile(os.path.join(code, d, "log.jsonl")):
        r = subprocess.run([sys.executable, "-m", "nca.progress", d, "--json"], cwd=code, capture_output=True,
                           text=True, timeout=60)
        try:
            p = json.loads(r.stdout)
            st.update({k: p[k] for k in ("iteration", "target", "rate", "verdict", "reason", "device")})
        except (ValueError, KeyError):
            pass
    runs[f] = st
up = float(read("/proc/uptime", "0 0").split()[0])
print(json.dumps({"time": round(time.time(), 1), "vm": name, "launch": launch, "commit": commit,
                  "phase": read(os.path.join(W, "phase"), "boot"), "uptimeSec": up,
                  "load": [float(x) for x in read("/proc/loadavg", "0 0 0").split()[:3]], "gpu": gpu, "runs": runs}))
PYEOF
}
sync_once() {  # $1 = quick: skip the dashboard (the shutdown path has ~30 s)
  local d f
  if [ -d "$CODE/runs" ]; then  # hard links of the whitelisted files only: the bucket is public
    rm -rf "$OUT" && mkdir -p "$OUT/runs"
    for d in $(plan_dirs); do
      [ -d "$CODE/runs/$d" ] || continue
      mkdir -p "$OUT/runs/$d"
      for f in $WHITELIST; do [ -f "$CODE/runs/$d/$f" ] && ln -f "$CODE/runs/$d/$f" "$OUT/runs/$d/$f"; done
    done
    sync_up "$OUT/runs" runs || log "sync of runs/ failed"
  fi
  if status_json >"$W/status.json.tmp" 2>/dev/null; then
    mv "$W/status.json.tmp" "$W/status.json"
    put "$W/status.json" status.json application/json || log "status.json upload failed"
  fi
  tail -n 300 "$W/startup.log" >"$W/startup.tail" 2>/dev/null && put "$W/startup.tail" startup.log text/plain
  if [ "${1:-}" != quick ] && [ -x "$PY" ] && [ -f "$CODE/nca/cloud/publish.sh" ]; then
    local dirs=()
    for d in $(plan_dirs); do [ -f "$CODE/runs/$d/log.jsonl" ] && dirs+=("$d"); done
    [ ${#dirs[@]} -eq 0 ] || BUCKET=$BUCKET PROJECT=$PROJECT PY=$PY RUNS=$CODE/runs \
      bash "$CODE/nca/cloud/publish.sh" --source vm "${dirs[@]}" >/dev/null 2>&1 || log "dashboard publish failed"
  fi
}
sidecar() { while :; do sync_once; sleep "$SYNC_SEC"; done; }

finish() {  # finish OK REASON: last sync, DONE, power off (only ever on the VM)
  trap - TERM INT
  [ -n "${SIDECAR:-}" ] && kill "$SIDECAR" 2>/dev/null
  if [ "$1" = true ]; then phase "done"; else phase "failed: $2"; fi
  sync_once
  "${PY_SYS:-python3}" - "$1" "$2" "$LAUNCH" "$STATE" >"$W/DONE" <<'PYEOF'
import json, os, sys, time
ok, reason, launch, sd = sys.argv[1:5]
runs = {}
for f in sorted(os.listdir(sd)):
    try:
        runs[f] = json.load(open(os.path.join(sd, f)))
    except (ValueError, OSError):
        pass
print(json.dumps({"launch": launch, "time": round(time.time(), 1), "ok": ok == "true", "reason": reason, "runs": runs}))
PYEOF
  put "$W/DONE" DONE application/json
  log "DONE ($2)"
  if [ -z "$TEST" ] && on_gce; then log "powering off"; poweroff; fi
  exit 0
}
bye() { log "giving up: $1"; finish false "$1"; }
# shellcheck disable=SC2317  # kill_tree and on_term run from the TERM trap
kill_tree() { local p c; for p in "$@"; do for c in $(pgrep -P "$p"); do kill_tree "$c"; done; kill "$p" 2>/dev/null; done; }
# shellcheck disable=SC2317
on_term() {  # systemd stops the unit (shutdown, preemption): stop the work, one last quick sync
  log "TERM (shutdown or preemption): one last sync"
  kill_tree "${SIDECAR:-}" "${pids[@]}"
  phase "stopping: shutdown"
  sync_once quick
  exit 0
}

# ---- the steps ----
ensure_driver() {
  grep -qs 0x10de /sys/bus/pci/devices/*/vendor || return 0   # no NVIDIA device: nothing to do
  nvidia-smi >/dev/null 2>&1 && return 0
  local n
  n=$(cat "$W/driver-tries" 2>/dev/null || echo 0)
  [ "$n" -ge 3 ] && return 1
  echo $((n + 1)) >"$W/driver-tries"
  curl -fsSL -o "$W/cuda_installer.pyz" \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz || return 1
  python3 "$W/cuda_installer.pyz" install_driver --installation-mode=repo --installation-branch=prod
  nvidia-smi >/dev/null 2>&1 && return 0
  log "the driver needs a reboot"; reboot; sleep 600   # the next boot carries on from here
}
ensure_venv() {
  "$PY" -c 'import torch, numpy' 2>/dev/null && return 0
  local apt=(apt-get -o DPkg::Lock::Timeout=600)   # a fresh VM's unattended-upgrades may hold the lock
  { "${apt[@]}" update -qq && DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq python3-venv git >/dev/null &&
    python3 -m venv "$W/venv" && "$W/venv/bin/pip" install -q --upgrade pip; } || return 1
  # shellcheck disable=SC2086  # TORCH may carry pip options, e.g. "torch==2.8.0 --index-url URL"
  "$W/venv/bin/pip" install -q $TORCH numpy
}
ensure_code() {
  if [ -n "$TEST" ]; then mkdir -p "$CODE" && rsync -a --exclude __pycache__ "$SRC/nca" "$CODE/"; return; fi
  [ -d "$CODE/.git" ] || git clone -q "$REPO" "$CODE" || return 1
  git -C "$CODE" fetch -q origin && git -C "$CODE" checkout -q "$COMMIT"
}

if [ "$1" = sync ]; then sync_once quick; exit 0; fi   # shutdown.sh

[ -n "$TEST" ] || exec >>"$W/startup.log" 2>&1
log "main: vm $NAME, launch $LAUNCH, commit $COMMIT, bucket $BUCKET"
command -v gcloud >/dev/null || [[ $BUCKET != gs://* ]] || snap install google-cloud-cli --classic
pids=()
trap on_term TERM INT
sidecar &
SIDECAR=$!
if [ -z "$TEST" ]; then
  phase driver; ensure_driver || bye "the NVIDIA driver did not install (3 tries)"
  phase venv; ensure_venv || bye "the venv (pip install $TORCH numpy) failed"
  if grep -qs 0x10de /sys/bus/pci/devices/*/vendor; then
    "$PY" -c 'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, torch.cuda.get_device_name(0))' ||
      bye "torch sees no GPU (driver vs wheel?)"
  fi
fi
phase code; ensure_code || bye "could not check out $COMMIT"
phase pull; sync_down runs "$CODE/runs"   # with checkpoints: a preempted VM's work resumes
PY_SYS=$PY
phase train
runs=$(plan_runs)
nruns=$(printf '%s\n' "$runs" | grep -c .)
[ "$nruns" -gt 0 ] || bye "the plan has no stages"
threads=$(( $(nproc) / nruns )); [ "$threads" -ge 1 ] || threads=1
for run in $runs; do
  n=$(plan_lines | awk -F'|' -v r="$run" '$1 == r' | wc -l)
  set_state "$run" waiting 0 "$n" waiting
  run_stages "$run" "$threads" &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
finish true "plan finished: $nruns run(s), $failed failed"
