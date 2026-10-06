#!/bin/bash
# shellcheck disable=SC2012  # ls of our own plain file names
# selftest.sh: the cloud scripts end to end against a local directory standing in for the bucket. No cloud
# calls, no money, nothing written outside WORK (default: a new temp dir). PY must have torch and numpy:
#   PY=/path/to/venv/python WORK=runs/_smoke-cloud nca/cloud/selftest.sh
# It trains a few tiny runs (R 3/4, hidden 16, batch 4, one thread; one of them 400 iterations) through
# startup.sh's test mode, each "VM" in its own work dir:
#   vm1  a 2-run plan (a: 2 stages chained by --init; b: one time-boxed stage with --init bucket:...) -> DONE
#   vm2  the same plan plus run c, sent TERM once c has a checkpoint in the bucket (a preemption)
#   vm3  a fresh VM on the same bucket: a and b are skipped, c resumes from its checkpoint and finishes
#   vm4  three runs naming the same bucket: file at once (each must wait for the one fetch, not fail)
# then pull.sh, watch.sh (each exit), the ledger, launch.sh --dry-run and publish.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=${PY:-python3}
WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
B=$WORK/bucket
fails=0
check() { local msg=$1; shift; if "$@"; then echo "OK $msg"; else echo "FAIL $msg"; fails=$((fails + 1)); fi; }
j() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print($2)" "$1" 2>/dev/null; }  # j FILE EXPR
lastline() { tail -n 1 "$1" | "$PY" -c "import json,sys; d=json.loads(sys.stdin.read()); print($2)"; }
TINY="--R 3 4 --batch 4 --pool-size 16 --hidden 16 --eval-mults 1 --eval-every 50 --snap-every 5"
vm() {  # vm NAME PLAN: startup.sh main in test mode, its own work dir, launch id NAME
  HEXCA_TEST=1 W=$WORK/$1 BUCKET=$B PLAN_FILE=$2 SRC=$ROOT PY=$PY SYNC_SEC=2 NAME=$1 LAUNCH=$1 \
    bash "$HERE/startup.sh" main >"$WORK/$1.log" 2>&1
}
rm -rf "$WORK"/r-* "$WORK"/wb-*
echo "-- work dir $WORK"
if [ -n "${SKIP_VMS:-}" ] && [ -f "$B/DONE" ]; then
  echo "-- SKIP_VMS: reusing the bucket the last run left"
else
rm -rf "$B" "$WORK"/vm? "$WORK"/seed

# A checkpoint in the bucket for --init bucket:init/tiny.pt.
mkdir -p "$WORK/seed" "$B/init"
# shellcheck disable=SC2086
(cd "$WORK/seed" && PYTHONPATH=$ROOT "$PY" -m nca.train --name seed $TINY --iters 2 --threads 1 >/dev/null 2>&1)
cp "$WORK/seed/runs/seed/ckpt.pt" "$B/init/tiny.pt"
check "a seed checkpoint in the bucket" test -f "$B/init/tiny.pt"

cat >"$WORK/plan1.txt" <<EOF
# two runs at once
a | s1 | 5   | $TINY --iters 3
a | s2 | 5   | $TINY --iters 6
b | s1 | 0.4 | $TINY --iters 100000 --init bucket:init/tiny.pt
EOF
{ cat "$WORK/plan1.txt"; echo "c | s1 | 5 | $TINY --iters 400"; } >"$WORK/plan2.txt"

echo "-- vm1: plan1 to the end"
vm vm1 "$WORK/plan1.txt"
check "vm1 exits 0" test $? -eq 0
check "DONE: ok, launch vm1, runs a and b done" test "$(j "$B/DONE" '(d["ok"], d["launch"], d["runs"]["a"]["state"], d["runs"]["b"]["state"])')" = "(True, 'vm1', 'done', 'done')"
check "a-s1 ends {stopped: done} at its --iters 3" test "$(lastline "$B/runs/a-s1/log.jsonl" '(d["stopped"], d["iteration"])')" = "('done', 3)"
check "a-s2 started --init from a-s1's best.pt (not its ckpt.pt)" test "$(head -n 1 "$B/runs/a-s2/log.jsonl" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["config"]["init"])')" = "runs/a-s1/best.pt"
check "b-s1 started --init from the bucket's init/tiny.pt and ended by its time box" \
  test "$(head -n 1 "$B/runs/b-s1/log.jsonl" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["config"]["init"])')|$(lastline "$B/runs/b-s1/log.jsonl" 'd["stopped"]')" = "$WORK/vm1/bucket/init/tiny.pt|time"
bad=$(find "$B/runs" -type f | grep -vE '/(log\.jsonl|pool\.npz|gallery\.npz|ckpt\.pt|best\.pt|stdout\.log)$')
check "only whitelisted files under the bucket's runs/ ${bad:+(not: $bad)}" test -z "$bad"
check "status.json: phase done, both runs, launch vm1" test "$(j "$B/status.json" '(d["phase"], sorted(d["runs"]), d["launch"])')" = "('done', ['a', 'b'], 'vm1')"
check "the dashboard: dash/index.html (static), runs-vm.json with the 3 stage dirs, runs-pi.json an empty placeholder, a run's log, pool and gallery" \
  test "$(grep -c 'var STATIC = true;' "$B/dash/index.html")|$(j "$B/dash/runs-vm.json" 'sorted(r["name"] for r in d)')|$(cat "$B/dash/runs-pi.json")|$(ls "$B/dash/a-s2")" = "1|['a-s1', 'a-s2', 'b-s1']|[]|gallery.json
log.json
pool.json"

echo "-- vm2: plan2, TERM once run c has a checkpoint in the bucket (a preemption)"
vm vm2 "$WORK/plan2.txt" &
vm2=$!
for _ in $(seq 1200); do [ -f "$B/runs/c-s1/ckpt.pt" ] && break; sleep 0.5; done  # up to 10 min: a busy Pi is slow
check "c-s1 checkpointed (iteration 200) and synced" test -f "$B/runs/c-s1/ckpt.pt"
main=$(pgrep -f "startup.sh main" -P "$vm2" | head -1)
kill -TERM "${main:-$vm2}"
wait "$vm2"
check "vm2 exits 0 on TERM" test $? -eq 0
sleep 1
check "TERM took the trainer down too (no nca.train c-s1 left)" test -z "$(pgrep -f 'nca.train --name c-s1')"
check "status.json after TERM: phase 'stopping: shutdown', launch vm2; DONE is still vm1's" \
  test "$(j "$B/status.json" '(d["phase"], d["launch"])')|$(j "$B/DONE" 'd["launch"]')" = "('stopping: shutdown', 'vm2')|vm1"
check "c-s1's log in the bucket has no stop line (killed mid-stage)" test "$(lastline "$B/runs/c-s1/log.jsonl" '"stopped" in d')" = False

echo "-- vm3: a fresh VM on the same bucket"
vm vm3 "$WORK/plan2.txt"
check "vm3 exits 0" test $? -eq 0
check "vm3 skipped a-s1, a-s2 (done) and b-s1 (time box used)" test "$(grep -c 'already ended' "$WORK/vm3.log")" -eq 3
check "c-s1 resumed from its checkpoint (a start line with startIteration > 0) and ended done at 400" \
  test "$(grep -c '"startIteration": [1-9]' "$B/runs/c-s1/log.jsonl")|$(lastline "$B/runs/c-s1/log.jsonl" '(d["stopped"], d["iteration"])')" = "1|('done', 400)"
check "DONE: vm3, ok, three runs done" test "$(j "$B/DONE" '(d["launch"], d["ok"], sorted(k for k, v in d["runs"].items() if v["state"] == "done"))')" = "('vm3', True, ['a', 'b', 'c'])"

echo "-- vm4: three runs fetch the same bucket: file at once (launch 8's startup race), on a bucket of its own"
B4=$WORK/bucket4
rm -rf "$B4" "$WORK/vm4" && mkdir -p "$B4/init" && cp "$B/init/tiny.pt" "$B4/init/tiny.pt"
for r in x y z; do echo "$r | s1 | 5 | $TINY --iters 1 --init bucket:init/tiny.pt"; done >"$WORK/plan4.txt"
HEXCA_TEST=1 W=$WORK/vm4 BUCKET=$B4 PLAN_FILE=$WORK/plan4.txt SRC=$ROOT PY=$PY SYNC_SEC=2 NAME=vm4 LAUNCH=vm4 \
  FETCH_DELAY=2 bash "$HERE/startup.sh" main >"$WORK/vm4.log" 2>&1
check "vm4 exits 0" test $? -eq 0
check "vm4: all three runs done, the file fetched once (the others waited on its lock), the copy whole" \
  test "$(j "$B4/DONE" '(d["ok"], sorted(k for k, v in d["runs"].items() if v["state"] == "done"))')|$(grep -c 'fetched init/tiny.pt' "$WORK/vm4.log")|$(cmp -s "$B4/init/tiny.pt" "$WORK/vm4/bucket/init/tiny.pt" && echo same)" = "(True, ['x', 'y', 'z'])|1|same"
check "vm4: no temp file left beside the copy (only tiny.pt and its lock)" \
  test "$(LC_ALL=C ls -A "$WORK/vm4/bucket/init" | tr '\n' ' ')" = "tiny.pt tiny.pt.lock "

fi  # SKIP_VMS

echo "-- pull.sh"
export BUCKET=$B LEDGER=$WORK/ledger.tsv PY
mkdir -p "$WORK/r-pi/b-s1" && echo '{"iteration": 0}' >"$WORK/r-pi/b-s1/log.jsonl"   # a local run named b-s1
out=$(RUNS=$WORK/r-pi bash "$HERE/pull.sh" 2>&1)
check "pull.sh: cloud dirs marked .from-cloud, log/pool/best but no ckpt.pt; the mirror has status.json and DONE" \
  test "$(LC_ALL=C ls -A "$WORK/r-pi/a-s1" | tr '\n' ' ')|$(LC_ALL=C ls "$WORK/r-pi/_cloud" | tr '\n' ' ')" = ".from-cloud best.pt log.jsonl pool.npz stdout.log |DONE runs status.json "
check "pull.sh skipped b-s1: a local run of that name is never overwritten" \
  test "$(cat "$WORK/r-pi/b-s1/log.jsonl")|$(echo "$out" | grep -c 'skipped b-s1')" = '{"iteration": 0}|1'
RUNS=$WORK/r-pi bash "$HERE/pull.sh" --ckpt >/dev/null 2>&1
check "pull.sh --ckpt brings ckpt.pt too" test -f "$WORK/r-pi/c-s1/ckpt.pt"
(cd "$ROOT" && "$PY" -m nca.progress "$WORK/r-pi/c-s1" >/dev/null); rc=$?
check "nca.progress reads a pulled run: c-s1 FINISHED (exit 0)" test $rc -eq 0

echo "-- watch.sh (WATCH_SEC=1)"
watch() {  # watch NAME [env...]: one pass of watch.sh (0 minutes) on bucket $WORK/wb-NAME, runs $WORK/r-NAME
  local n=$1; shift
  env BUCKET="$WORK/wb-$n" RUNS="$WORK/r-$n" WATCH_SEC=1 "$@" bash "$HERE/watch.sh" 0 >"$WORK/watch-$n.log" 2>&1
  echo $?
}
fake() {  # fake NAME RUN-STATE LOG-KIND STATUS-AGE: a bucket with one running run and a synthetic log
  mkdir -p "$WORK/wb-$1/runs/p-s1"
  (cd "$ROOT" && "$PY" - "$WORK/wb-$1" "$2" "$3" "$4" <<'PYEOF'
import json, sys, time
from nca.progress import _synth
b, state, kind, age = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
now = time.time()
rise = lambda k: 0.2 + 0.08 * k
kw = {"plateau": dict(score=lambda k: 0.5, loss=lambda i: 0.05),
      "diverged": dict(score=rise, loss=lambda i: 0.05 if i < 1200 else 0.3),
      "progress": dict(score=rise, loss=lambda i: 0.1 / (1 + i / 500))}[kind]
_synth(f"{b}/runs/p-s1/log.jsonl", n_checks=10, t0=now - 37 * 60 - 30, **kw)  # 37 lines a minute apart
json.dump({"time": now - age, "launch": "x", "phase": "train", "load": [1, 1, 1], "gpu": None,
           "runs": {"p": {"stage": "s1", "stageIndex": 1, "stages": 1, "dir": "p-s1", "state": state}}},
          open(f"{b}/status.json", "w"))
PYEOF
  )
}
mkdir -p "$WORK/wb-done" && cp -r "$B/runs" "$B/status.json" "$B/DONE" "$WORK/wb-done/"
check "watch: DONE in the bucket -> exit 2" test "$(watch "done")" -eq 2
check "...and the block names each run with its verdict" grep -q 'c-s1\|^  c ' "$WORK/watch-done.log"
fake progress running progress 30
check "watch: a run in PROGRESS, fresh heartbeat, time up -> exit 0" test "$(watch progress)" -eq 0
fake gone running progress 30
check "watch: the VM gone, no DONE -> exit 8" test "$(watch gone FAKE_VM_STATUS=GONE)" -eq 8
fake stopped running progress 30
check "watch: the VM stopped, not deleted -> exit 9" test "$(watch stopped FAKE_VM_STATUS=TERMINATED)" -eq 9
grep -q 'stopped, not deleted' "$WORK/watch-stopped.log"; check "...and says 'stopped, not deleted'" test $? -eq 0
fake stale running progress 1200
check "watch: a 20-minute-old heartbeat -> exit 7" test "$(watch stale)" -eq 7
# A launch 20 min old whose status.json can't be fetched (a transient failure: the bucket has none, the mirror
# kept the last one seen, 30 s old): not a dead VM until FETCH_FAILS passes in a row fail.
fake flaky running progress 30
printf 'utc\tunix\tevent\tvm\tzone\tlaunch\tmaxHours\tnote\n-\t%s\tcreate\thexca-train\tz\tx\t6\ttest\n' \
  "$(( $(date +%s) - 1200 ))" >"$WORK/ledger-x.tsv"
mkdir -p "$WORK/r-flaky/_cloud" && mv "$WORK/wb-flaky/status.json" "$WORK/r-flaky/_cloud/status.last.json"
check "watch: one failed status fetch (the last heartbeat seen 30 s old) 20 min after the launch -> no exit 7 (exit 0)" \
  test "$(watch flaky LEDGER="$WORK/ledger-x.tsv" FETCH_PAUSE=0)" -eq 0
grep -q 'no current status.json this pass: 1 in a row' "$WORK/watch-flaky.log"; check "...and says it used the last one seen" test $? -eq 0
rm -f "$WORK/r-flaky/_cloud/status.last.json"
check "watch: no heartbeat fetched in FETCH_FAILS (3) passes in a row, 20 min after the launch -> exit 7" \
  test "$(env BUCKET="$WORK/wb-flaky" RUNS="$WORK/r-flaky" WATCH_SEC=1 FETCH_PAUSE=0 LEDGER="$WORK/ledger-x.tsv" \
          bash "$HERE/watch.sh" 1 >"$WORK/watch-flaky3.log" 2>&1; echo $?)" -eq 7
grep -q 'no heartbeat of this launch fetched in 3 passes' "$WORK/watch-flaky3.log"; check "...and says so, with the count" test $? -eq 0
fake plateau running plateau 30
check "watch: a run PLATEAU -> exit 3" test "$(watch plateau)" -eq 3
check "watch: ...not with WATCH_IGNORE=PLATEAU -> exit 0" test "$(watch plateau WATCH_IGNORE=PLATEAU)" -eq 0
fake diverged running diverged 30
check "watch: a run DIVERGED -> exit 5" test "$(watch diverged)" -eq 5
fake ended "done" progress 30
check "watch: a run ENDED (all its stages done) -> exit 6" test "$(watch ended)" -eq 6

echo "-- the ledger and launch.sh --dry-run"
export MAX_HOURS=6 BUDGET_HOURS=6.5  # what these checks assume (env.sh's defaults have moved since)
now=$(date +%s)
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$HERE/common.sh"
rm -f "$LEDGER"
ledger_add create hexca-train us-central1-a L1 6 test $((now - 7200))
ledger_add stop hexca-train us-central1-a L1 6 test $((now - 3600))
ledger_add create hexca-train us-central1-a L2 6 test $((now - 1800))
check "ledger: 1 h (create..stop) + 0.5 h (still open) = 1.50 h" test "$(ledger_hours)" = 1.50
check "ledger: the last launch is L2, open" test "$(ledger_last)" = "L2 us-central1-a 6 open"
out=$(bash "$HERE/launch.sh" --dry-run "$HERE/plan.example.txt" 2>&1); rc=$?
check "launch.sh refuses: 1.5 h used + MAX_HOURS 6 > BUDGET_HOURS 6.5" test "$rc|$(echo "$out" | grep -c 'REFUSED: the ledger has 1.50')" = "1|1"
rm -f "$LEDGER"
out=$(bash "$HERE/launch.sh" --dry-run "$HERE/plan.example.txt" 2>&1); rc=$?
check "launch.sh --dry-run with an empty ledger prints the create command (SPOT, DELETE, 21600s, storage-rw, label)" \
  test "$rc|$(echo "$out" | grep -c 'gcloud compute instances create hexca-train --project=recipe-lanes-staging .*--provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration=21600s .*--scopes=storage-rw --labels=purpose=hexca')" = "0|1"
printf 'x | s1 | 400 | --R 4\nY | s1 | 5 | --R 4\nz | s1 | 5 | --R 4 --resume\n' >"$WORK/badplan.txt"
out=$(bash "$HERE/launch.sh" --dry-run "$WORK/badplan.txt" 2>&1); rc=$?
check "launch.sh refuses a bad plan (400 min > 335, upper-case name, --resume)" \
  test "$rc|$(echo "$out" | grep -cE '> 335|names are|set by the VM')" = "1|3"

echo "-- publish.sh from 'the Pi' next to the VM's"
rm -f "$B/dash/play.html"
RUNS=$WORK/seed/runs bash "$HERE/publish.sh" seed >/dev/null 2>&1
check "publish.sh --source pi: runs-pi.json lists the Pi's run (hasWeights); runs-vm.json still the VM's 4 stage dirs; play.html and seed/weights.json (needs npm run build)" \
  test "$(j "$B/dash/runs-pi.json" '[(r["name"], r["hasWeights"]) for r in d]')|$(j "$B/dash/runs-vm.json" 'len(d)')|$(ls "$B/dash/seed" | tr '\n' ' ')|$(j "$B/dash/seed/weights.json" 'd["meta"]["iterations"] >= 0')|$(grep -c '<canvas id="board"' "$B/dash/play.html")" = "[('seed', True)]|4|gallery.json log.json pool.json weights.json |True|1"
WB=$WORK/wb-weights && rm -rf "$WB"
BUCKET=$WB RUNS=$WORK/seed/runs bash "$HERE/publish.sh" --weights-only seed nosuchrun >/dev/null 2>&1
check "publish.sh --weights-only: play.html, strand.html and seed/weights.json, nothing else" \
  test "$(cd "$WB/dash" && find . -type f | LC_ALL=C sort | tr '\n' ' ')" = "./play.html ./seed/weights.json ./strand.html "

echo
if [ "$fails" -eq 0 ]; then echo "cloud selftest: ALL OK ($WORK)"; else echo "cloud selftest: $fails FAILED ($WORK)"; fi
[ "$fails" -eq 0 ]
