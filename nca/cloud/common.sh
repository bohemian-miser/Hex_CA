# Shared by launch.sh, pull.sh, watch.sh, down.sh (sourced). Loads env.sh; bucket access that works with
# gs:// (gcloud storage) or a local directory (plain cp/rsync, for tests); the local VM-hours ledger.
# shellcheck shell=bash

CLOUD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$CLOUD_DIR/../.." && pwd)                 # the hex_ca checkout
# shellcheck source-path=SCRIPTDIR source=env.sh
. "${CLOUD_ENV:-$CLOUD_DIR/env.sh}"
RUNS=${RUNS:-$ROOT/runs}                             # local runs/ (the dashboard's)
MIRROR=${MIRROR:-$RUNS/_cloud}                       # plain mirror of the bucket: runs/, status.json, DONE, startup.log
LEDGER=${LEDGER:-$CLOUD_DIR/ledger.tsv}
PY=${PY:-$(command -v python3)}

is_gs() { [[ $BUCKET == gs://* ]]; }
gc() { gcloud --project="$PROJECT" --quiet "$@"; }   # every gcloud call names the project

# get REL DEST: copy one object (a file under a local BUCKET); fails quietly if it isn't there
get() {
  if is_gs; then gc storage cp "$BUCKET/$1" "$2" >/dev/null 2>&1
  else [ -f "$BUCKET/$1" ] && cp "$BUCKET/$1" "$2"; fi
}
# sync_down REL DEST [--ckpt]: copy a prefix down, never deleting anything local; ckpt.pt only with --ckpt
sync_down() {
  mkdir -p "$2"
  if is_gs; then
    if [ "${3:-}" = --ckpt ]; then gc storage rsync -r "$BUCKET/$1" "$2" >/dev/null
    else gc storage rsync -r --exclude='.*ckpt\.pt$' "$BUCKET/$1" "$2" >/dev/null; fi
  else
    [ -d "$BUCKET/$1" ] || return 0
    if [ "${3:-}" = --ckpt ]; then rsync -a "$BUCKET/$1/" "$2/"
    else rsync -a --exclude=ckpt.pt "$BUCKET/$1/" "$2/"; fi
  fi
}

# json FILE EXPR: a value out of a JSON file with python (EXPR sees the parsed object as d); empty if missing
json() { [ -f "$1" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print($2)" "$1" 2>/dev/null; }

# ---- the ledger: one line per event, tab-separated --------------------------------------------------------
#   utc  unix  event(create|stop|gone|delete)  vm  zone  launch  maxHours  note
# A launch's VM hours run from its create to its first stop/gone/delete; with none recorded yet, to now,
# capped at its maxHours (--max-run-duration deletes it by then whatever happens).
ledger_add() {  # ledger_add EVENT VM ZONE LAUNCH MAXHOURS NOTE [UNIX]
  [ -f "$LEDGER" ] || printf 'utc\tunix\tevent\tvm\tzone\tlaunch\tmaxHours\tnote\n' >"$LEDGER"
  local t=${7:-$(date +%s)}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u -d "@$t" +%FT%TZ)" "$t" "$1" "$2" "$3" "$4" "$5" "$6" >>"$LEDGER"
}
ledger_hours() {  # total VM hours of every launch in the ledger
  [ -f "$LEDGER" ] || { echo 0; return; }
  awk -F'\t' -v now="$(date +%s)" 'NR > 1 {
      if ($3 == "create") { t0[$6] = $2; cap[$6] = $7 * 3600; order[++n] = $6 }
      else if (($3 == "stop" || $3 == "gone" || $3 == "delete") && ($6 in t0) && !($6 in t1)) t1[$6] = $2 }
    END { for (i = 1; i <= n; i++) { l = order[i]; e = (l in t1) ? t1[l] : now
            d = e - t0[l]; if (d > cap[l]) d = cap[l]; if (d < 0) d = 0; h += d }
          printf "%.2f\n", h / 3600 }' "$LEDGER"
}
ledger_last() {  # "launch zone maxHours" of the last create, and whether it is closed: open|stopped|closed
  [ -f "$LEDGER" ] || return 1
  awk -F'\t' 'NR > 1 { if ($3 == "create") { l = $6; z = $5; m = $7; s = "open" }
                       else if ($6 == l && $3 == "stop" && s == "open") s = "stopped"
                       else if ($6 == l && ($3 == "gone" || $3 == "delete")) s = "closed" }
              END { if (l != "") print l, z, m, s; else exit 1 }' "$LEDGER"
}
ledger_has() { [ -f "$LEDGER" ] && awk -F'\t' -v l="$1" -v e="$2" '$6 == l && $3 == e { f = 1 } END { exit !f }' "$LEDGER"; }

# hexca_instances: "name zone status" of every instance labelled purpose=hexca (or named $VM) in PROJECT
hexca_instances() {
  gc compute instances list --filter="labels.${LABEL%%=*}=${LABEL#*=} OR name=$VM" \
    --format='value(name,zone.basename(),status)'
}
