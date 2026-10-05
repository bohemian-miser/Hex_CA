#!/bin/bash
# launch.sh [--dry-run] PLAN_FILE: create the one Spot VM ($VM) that trains PLAN_FILE (see plan.example.txt).
# THE LEAD / OWNER ONLY: the one script here that starts spending money.
#
# It refuses unless
#   - the plan parses, and each run's minutes + SETUP_MIN fit in MAX_HOURS;
#   - the ledger's VM hours so far + MAX_HOURS <= BUDGET_HOURS (ledger.tsv; watch.sh and down.sh add to it);
#   - HEAD is the pushed tip of its branch on GitHub and nca/ has no uncommitted changes (the VM clones it);
#   - no instance labelled purpose=hexca (or named $VM) exists in $PROJECT (the GPU quota of 1 also says so);
#   - every bucket:PATH the plan names exists in the bucket.
# Then it creates the VM in $ZONE, or in $ZONE_FALLBACKS when a zone has no Spot capacity: SPOT,
# --instance-termination-action=DELETE, --max-run-duration=MAX_HOURS, label purpose=hexca, the bucket-only
# service account with --scopes=storage-rw, metadata bucket / commit / launch / repo / torch / plan,
# startup.sh as the startup-script and shutdown.sh as the shutdown-script; and adds "create" to the ledger.
# --dry-run: the local checks only (git problems are warnings), then prints the command; nothing in the cloud.
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
dry=
[ "${1:-}" = --dry-run ] && { dry=1; shift; }
plan=${1:-}
[ -f "$plan" ] || { echo "usage: launch.sh [--dry-run] PLAN_FILE" >&2; exit 1; }
refuse() { echo "launch.sh: REFUSED: $*" >&2; exit 1; }
warn() { echo "launch.sh: warning: $*" >&2; }
gitcheck() { if [ -n "$dry" ]; then warn "$* (dry run: carrying on)"; else refuse "$*"; fi; }
[[ $plan != *,* ]] || refuse "the plan's path has a comma (it travels in --metadata-from-file)"
[[ $TORCH != *,* ]] || refuse "TORCH='$TORCH' has a comma (it travels in --metadata)"

# 1. The plan: run | stage | minutes | args; names [a-z0-9-]; each run's minutes + setup within MAX_HOURS.
max_min=$(awk -v h="$MAX_HOURS" -v s="$SETUP_MIN" 'BEGIN { printf "%d", h * 60 - s }')
if ! summary=$(sed 's/#.*//' "$plan" | awk -F'|' -v max="$max_min" '
    NF == 0 || $0 ~ /^[ \t]*$/ { next }
    { for (i = 1; i <= NF; i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    NF < 4 { print "line " NR ": want run | stage | minutes | args"; bad = 1; next }
    $1 !~ /^[a-z0-9][a-z0-9-]*$/ || $2 !~ /^[a-z0-9][a-z0-9-]*$/ { print "line " NR ": names are [a-z0-9-]"; bad = 1 }
    $3 !~ /^[0-9]+(\.[0-9]+)?$/ || $3 <= 0 { print "line " NR ": minutes must be a positive number"; bad = 1 }
    $4 ~ /(^| )--(name|resume|minutes|device)( |$)/ { print "line " NR ": --name/--resume/--minutes/--device are set by the VM"; bad = 1 }
    seen[$1 "-" $2]++ { print "line " NR ": stage " $1 "-" $2 " twice"; bad = 1 }
    { if (!($1 in tot)) order[++n] = $1; tot[$1] += $3; stages++ }
    END { if (!stages) { print "no stages"; bad = 1 }
          for (i = 1; i <= n; i++) { r = order[i]
            printf "  run %-16s %6.1f min%s\n", r, tot[r], (tot[r] > max ? "  > " max " (MAX_HOURS - SETUP_MIN)" : "")
            if (tot[r] > max) bad = 1 }
          exit bad }'); then
  refuse "the plan:"$'\n'"$summary"
fi
echo "plan $plan:"
echo "$summary"
while read -r d; do  # pull.sh never overwrites a local run: a clash would hide the cloud one here
  [ -e "$RUNS/$d" ] && [ ! -e "$RUNS/$d/.from-cloud" ] && warn "runs/$d is a local run: pull.sh will skip the cloud $d"
done < <(sed 's/#.*//' "$plan" | awk -F'|' 'NF >= 4 { gsub(/[ \t]/, "", $1); gsub(/[ \t]/, "", $2); print $1 "-" $2 }')

# 2. The budget.
used=$(ledger_hours)
if awk -v u="$used" -v m="$MAX_HOURS" -v b="$BUDGET_HOURS" 'BEGIN { exit !(u + m > b) }'; then
  refuse "the ledger has ${used} VM hours; + MAX_HOURS $MAX_HOURS > BUDGET_HOURS $BUDGET_HOURS ($LEDGER)"
fi
echo "budget: ${used} h used + ${MAX_HOURS} h max this launch <= ${BUDGET_HOURS} h"

# 3. The code the VM will run: the pushed HEAD.
branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
commit=$(git -C "$ROOT" rev-parse HEAD)
git -C "$ROOT" diff --quiet HEAD -- nca || gitcheck "nca/ has uncommitted changes; the VM runs $commit without them"
remote=$(GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" timeout 30 \
  git -C "$ROOT" ls-remote origin "refs/heads/$branch" 2>/dev/null | cut -f1)
[ "$remote" = "$commit" ] || gitcheck "HEAD $commit is not the pushed tip of origin/$branch (${remote:-unreachable or absent}): push first"
slug() { echo "$1" | sed -E 's#^.*github\.com[:/]##; s#\.git$##' | tr '[:upper:]' '[:lower:]'; }
[ "$(slug "$(git -C "$ROOT" remote get-url origin)")" = "$(slug "$REPO")" ] || gitcheck "origin is not $REPO"
echo "code: $commit ($branch), cloned from $REPO"

# 4. The cloud: nothing of ours running; the plan's bucket: files exist (read-only calls).
launch=$(date -u +%Y%m%d-%H%M%S)
if [ -z "$dry" ]; then
  existing=$(hexca_instances) || refuse "could not list instances in $PROJECT (gcloud auth?)"
  [ -z "$existing" ] || refuse "an instance exists already: $existing  (nca/cloud/down.sh first)"
  while read -r rel; do
    [ -z "$rel" ] || gc storage ls "$BUCKET/$rel" >/dev/null 2>&1 ||
      refuse "the plan names bucket:$rel, but $BUCKET/$rel is not there"
  done < <(sed 's/#.*//' "$plan" | grep -o 'bucket:[^ |]*' | sed 's/^bucket://' | sort -u)
fi

secs=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')
cmd_for() {  # cmd_for ZONE: the create command, in the array cmd
  cmd=(gcloud compute instances create "$VM" --project="$PROJECT" --zone="$1" --machine-type="$MACHINE"
    --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration="${secs}s"
    --maintenance-policy=TERMINATE
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT"
    --boot-disk-size=50GB --boot-disk-type=pd-balanced
    --service-account="$SA" --scopes=storage-rw
    --labels="$LABEL,launch=$launch,commit=${commit:0:12}"
    --metadata="bucket=$BUCKET,commit=$commit,launch=$launch,repo=$REPO,torch=$TORCH"
    --metadata-from-file="startup-script=$CLOUD_DIR/startup.sh,shutdown-script=$CLOUD_DIR/shutdown.sh,plan=$plan")
}
if [ -n "$dry" ]; then
  echo "dry run: would check that no instance labelled $LABEL (or named $VM) exists, then run:"
  cmd_for "$ZONE"
  printf '%q ' "${cmd[@]}"
  echo
  echo "(then, if $ZONE has no Spot capacity: the same with --zone= each of: $ZONE_FALLBACKS)"
  exit 0
fi
for z in $ZONE $ZONE_FALLBACKS; do
  echo "creating $VM in $z ..."
  cmd_for "$z"
  if err=$("${cmd[@]}" 2>&1); then
    ledger_add create "$VM" "$z" "$launch" "$MAX_HOURS" "commit ${commit:0:12} plan $(basename "$plan")"
    echo "$err"
    echo "created $VM in $z (launch $launch). Watch: nca/cloud/watch.sh   End: nca/cloud/down.sh"
    echo "Public dashboard: https://storage.googleapis.com/${BUCKET#gs://}/dash/index.html"
    exit 0
  fi
  echo "$err" >&2
  if [ -n "$(hexca_instances)" ]; then  # failed, yet something exists: count it, and stop here
    ledger_add create "$VM" "$z" "$launch" "$MAX_HOURS" "create reported an error but the VM exists"
    refuse "create in $z reported an error, but an instance exists now: see above (down.sh to remove it)"
  fi
  echo "$err" | grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|stockout|currently unavailable' ||
    refuse "create failed in $z (not a capacity problem; see above)"
  echo "no Spot capacity in $z; trying the next zone"
done
refuse "no Spot capacity in any of: $ZONE $ZONE_FALLBACKS (try again later)"
