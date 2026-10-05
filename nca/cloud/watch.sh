#!/bin/bash
# watch.sh [minutes]: wait (default 40 min) while the cloud runs are fine; exit as soon as one needs a look,
# so a waiting agent (or person) is woken. Read-only towards the cloud (it only appends to the local ledger).
# Every WATCH_SEC (120) seconds: pull.sh, then nca.progress on each run's current stage, then a short block.
#
# Exit code (the first that applies):
#   0  the minutes ran out and all is well (every run PROGRESS / WARMUP / between stages, heartbeat fresh)
#   2  DONE is in the bucket: the plan ended (see its reason) and the VM powers itself off -> run down.sh
#   8  the VM is gone without DONE: deleted (Spot preemption, --max-run-duration, or down.sh)
#   9  the VM is stopped, not deleted (it still costs its disk) -> run down.sh
#   7  the heartbeat (status.json) is older than 10 min, or there is none 15 min after the launch
#   5  a run is DIVERGED      4  a run is STALLED      3  a run is PLATEAU
#   6  a run ENDED (all its stages done, or failed) while others go on
#   1  usage or settings error
# WATCH_IGNORE=PLATEAU,ENDED (any of PLATEAU STALLED DIVERGED ENDED) stops those from waking you again once
# you have seen them. For tests: a local BUCKET skips the VM check (FAKE_VM_STATUS=RUNNING|TERMINATED|GONE).
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
minutes=${1:-40}
[[ $minutes =~ ^[0-9]+$ ]] || { echo "usage: watch.sh [minutes]" >&2; exit 1; }
WATCH_SEC=${WATCH_SEC:-120}
STALE_SEC=${STALE_SEC:-600}
IGNORE=",${WATCH_IGNORE:-},"
end=$(( $(date +%s) + 60 * minutes ))
mkdir -p "$MIRROR"
read -r launch zone maxh _ < <(ledger_last 2>/dev/null || echo "- - - -")
created=$(awk -F'\t' -v l="$launch" '$3 == "create" && $6 == l { t = $2 } END { print t + 0 }' "$LEDGER" 2>/dev/null || echo 0)

rank() { case $1 in 5) echo 4 ;; 4) echo 3 ;; 3) echo 2 ;; 6) echo 1 ;; *) echo 0 ;; esac; }  # worst first
vm_status() {  # RUNNING, TERMINATED (stopped), ..., or GONE
  if ! is_gs; then echo "${FAKE_VM_STATUS:-RUNNING}"; return; fi
  local s
  s=$(hexca_instances 2>/dev/null | awk '{ print $3 }' | head -1) || s=
  echo "${s:-GONE}"
}
note_ledger() {  # note_ledger EVENT [UNIX]: once per launch
  [ "$launch" != - ] && ! ledger_has "$launch" "$1" && ledger_add "$1" "$VM" "$zone" "$launch" "$maxh" "seen by watch.sh" "${2:-}"
}

while :; do
  bash "$(dirname "$0")/pull.sh" >/dev/null 2>"$MIRROR/pull.err" || true
  now=$(date +%s)
  st=$MIRROR/status.json
  s_launch=$(json "$st" 'd.get("launch", "")')
  if [ -f "$st" ] && [ "$launch" != - ] && [ "$s_launch" != "$launch" ]; then
    st=/nonexistent   # a heartbeat from an earlier launch says nothing about this one
  fi
  s_time=$(json "$st" 'int(d["time"])')
  if [ -n "$s_time" ]; then age=$(( now - s_time )); else age=; fi
  vm=$(vm_status)
  done_ok=
  if [ -f "$MIRROR/DONE" ] && { [ "$launch" = - ] || [ "$(json "$MIRROR/DONE" 'd.get("launch", "")')" = "$launch" ]; }; then
    done_ok=$(json "$MIRROR/DONE" '("ok" if d["ok"] else "FAILED") + ": " + d["reason"]')
  fi

  # The block: the VM, then a line per run (its current stage) with nca.progress's verdict.
  echo "[$(date +%T)] $VM $vm  phase $(json "$st" 'd["phase"]' || echo '?')  heartbeat ${age:-?}s ago  $(json "$st" \
    '"gpu %d%% %.1f/%.1f GB" % (d["gpu"]["util"], d["gpu"]["memUsedMB"] / 1024, d["gpu"]["memTotalMB"] / 1024) if d.get("gpu") else "gpu -"')  load $(json "$st" 'd["load"][0]')"
  worst=0 why=
  runs=$(json "$st" '"\n".join(sorted(d.get("runs", {})))')
  for run in $runs; do
    r_state=$(json "$st" "d['runs']['$run']['state']")
    r_dir=$(json "$st" "d['runs']['$run']['dir']")
    r_stage=$(json "$st" "'%s (%s/%s)' % (d['runs']['$run']['stage'], d['runs']['$run']['stageIndex'], d['runs']['$run']['stages'])")
    if [ -f "$RUNS/$r_dir/log.jsonl" ]; then
      (cd "$ROOT" && "$PY" -m nca.progress "$RUNS/$r_dir" --json >"$MIRROR/progress.json" 2>/dev/null)
      verdict=$(json "$MIRROR/progress.json" 'd["verdict"]') reason=$(json "$MIRROR/progress.json" 'd["reason"]')
      it=$(json "$MIRROR/progress.json" '"it %s/%s %s it/s" % (d["iteration"], d["target"], "%.3g" % d["rate"] if d["rate"] else "?")')
    else
      verdict=STARTING reason="no log here yet" it=
    fi
    case $r_state/$verdict in
      done/*|failed/*) word=ENDED code=6 reason="run $r_state (stage $r_stage)" ;;
      running/FINISHED|waiting/*) word=$verdict code=0 reason="between stages" ;;
      */PLATEAU) word=PLATEAU code=3 ;;
      */STALLED) word=STALLED code=4 ;;
      */DIVERGED) word=DIVERGED code=5 ;;
      *) word=$verdict code=0 ;;
    esac
    printf '  %-14s %-12s %-28s %-9s %s\n' "$run" "$r_stage" "$it" "$word" "$reason"
    if [ "$code" -ne 0 ] && [[ $IGNORE != *",$word,"* ]]; then
      if [ "$(rank "$code")" -gt "$(rank "$worst")" ]; then worst=$code why="$run: $word - $reason"; fi
    fi
  done

  if [ -n "$done_ok" ]; then
    echo "EXIT 2: DONE ($done_ok). The VM powers itself off: run nca/cloud/down.sh to delete it and pull the checkpoints."
    note_ledger stop "$(json "$MIRROR/DONE" 'int(d["time"])')"
    exit 2
  fi
  if [ "$vm" = GONE ]; then
    echo "EXIT 8: the VM $VM is gone and there is no DONE (preempted, or out of --max-run-duration). Its last synced work is in the bucket; launch.sh again resumes it."
    note_ledger gone
    exit 8
  fi
  if [ "$vm" != RUNNING ] && [ "$vm" != PROVISIONING ] && [ "$vm" != STAGING ]; then
    echo "EXIT 9: the VM $VM is $vm: stopped, not deleted (its disk still costs). Run nca/cloud/down.sh."
    note_ledger stop
    exit 9
  fi
  if [ -n "$age" ] && [ "$age" -gt "$STALE_SEC" ]; then
    echo "EXIT 7: the heartbeat is ${age}s old (limit ${STALE_SEC}s): the VM's script is stuck or dead. See $MIRROR/startup.log."
    exit 7
  fi
  if [ -z "$age" ] && [ "$created" -gt 0 ] && [ $(( now - created )) -gt 900 ]; then
    echo "EXIT 7: no heartbeat 15 min after the launch. See $MIRROR/startup.log (if any) or the serial console."
    exit 7
  fi
  if [ "$worst" -ne 0 ]; then
    echo "EXIT $worst: $why"
    exit "$worst"
  fi
  if [ "$(date +%s)" -ge "$end" ]; then
    echo "EXIT 0: $minutes min, all well${WATCH_IGNORE:+ (not waking on: $WATCH_IGNORE)}."
    exit 0
  fi
  sleep "$WATCH_SEC"
done
