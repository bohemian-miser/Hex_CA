#!/bin/bash
# down.sh: end it. THE LEAD / OWNER ONLY. Safe to run again at any time.
#   1. a final pull.sh --ckpt (checkpoints too) into runs/
#   2. delete every instance labelled purpose=hexca (or named $VM) in $PROJECT, and note it in the ledger
#   3. check nothing of ours is left in $PROJECT (a shared project: only things labelled purpose=hexca or
#      named hexca-*): instances, disks, snapshots, addresses, custom images; print one line
# The bucket and what is in it stay (the public dashboard reads it).
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
"$CLOUD_DIR/pull.sh" --ckpt || echo "down.sh: the final pull failed (the bucket keeps everything anyway)" >&2
read -r launch zone maxh state < <(ledger_last 2>/dev/null || echo "- - - -")

found=0
while read -r name z status; do
  [ -n "$name" ] || continue
  found=1
  echo "deleting $name ($z, $status) ..."
  if gc compute instances delete "$name" --zone="$z"; then
    [ "$launch" = - ] || ledger_add delete "$name" "$z" "$launch" "$maxh" "down.sh (it was $status)"
  else
    echo "down.sh: deleting $name failed" >&2
  fi
done < <(hexca_instances)
if [ "$found" = 0 ] && [ "$launch" != - ] && [ "$state" != closed ]; then
  ledger_add gone "$VM" "$zone" "$launch" "$maxh" "down.sh found no instance"
fi

# Everything of ours, by label or name; a list that fails counts as "could not check", never as clean.
filter="labels.${LABEL%%=*}=${LABEL#*=} OR name~^hexca-"
left='' unchecked=''
for kind in instances disks snapshots addresses images; do
  extra=()
  [ "$kind" = images ] && extra=(--no-standard-images)
  if out=$(gc compute "$kind" list "${extra[@]}" --filter="$filter" --format='value(name)' 2>/dev/null); then
    [ -z "$out" ] || left="$left $kind: $(echo "$out" | tr '\n' ' ')"
  else
    unchecked="$unchecked $kind"
  fi
done
if [ -n "$left" ]; then
  echo "STILL EXISTS in $PROJECT:$left -- delete it (e.g. gcloud compute disks delete NAME --zone=ZONE --project=$PROJECT)"
  exit 1
elif [ -n "$unchecked" ]; then
  echo "UNCHECKED: could not list$unchecked in $PROJECT (permissions?) -- nothing else of ours found"
  exit 1
fi
echo "clean: no hexca instances, disks, snapshots, addresses or images in $PROJECT ($(ledger_hours) VM hours in the ledger)"
