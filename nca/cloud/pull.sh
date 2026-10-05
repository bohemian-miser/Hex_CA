#!/bin/bash
# pull.sh [--ckpt]: copy the cloud runs into this machine's runs/, so the dashboard (localhost:8765) and
# nca.progress see them next to the Pi's own. Read-only towards the cloud.
#   runs/_cloud/          a plain mirror of the bucket: runs/, status.json, DONE, startup.log (the VM's log tail)
#   runs/<run>-<stage>/   each cloud run dir, marked .from-cloud; a local run of the same name is never touched
# ckpt.pt (the big file: weights + optimiser + pools) only with --ckpt; best.pt always.
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
ckpt=
[ "${1:-}" = --ckpt ] && ckpt=--ckpt
mkdir -p "$MIRROR/runs"
sync_down runs "$MIRROR/runs" $ckpt || echo "pull: nothing under $BUCKET/runs yet" >&2
for f in status.json DONE startup.log; do  # the mirror says what the bucket has: a file gone there goes here
  if get "$f" "$MIRROR/$f.tmp"; then mv "$MIRROR/$f.tmp" "$MIRROR/$f"; else rm -f "$MIRROR/$f.tmp" "$MIRROR/$f"; fi
done
n=0
for d in "$MIRROR"/runs/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d") dest=$RUNS/$(basename "$d")
  if [ -e "$dest" ] && [ ! -e "$dest/.from-cloud" ]; then
    echo "pull: skipped $name: runs/$name is a local run" >&2
    continue
  fi
  mkdir -p "$dest" && touch "$dest/.from-cloud" && rsync -a "$d" "$dest/" && n=$((n + 1))
done
echo "pull: $n cloud run dir(s) in $RUNS from $BUCKET$( [ -f "$MIRROR/DONE" ] && echo ' (DONE is there)')"
