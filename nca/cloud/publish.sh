#!/bin/bash
# publish.sh [--source pi|vm] [--every SEC] RUN...: the static dashboard of these runs (runs/RUN) to
# $BUCKET/dash/, the public page https://storage.googleapis.com/<bucket>/dash/index.html.
#
#   nca/cloud/publish.sh pure-a                  # once, from this Pi (source pi)
#   nca/cloud/publish.sh --every 60 pure-a       # in a loop (Ctrl-C to stop)
# The VM's sidecar runs it with --source vm for the cloud runs. Each publisher writes only its own list,
# runs-<source>.json, plus <run>/log.json and <run>/pool.json (so run names must differ between the
# two); the page merges runs-pi.json and runs-vm.json. A missing list is created empty, never overwritten
# (no-clobber), so the page doesn't 404. Objects go up with Cache-Control "no-cache, max-age=0" (public
# objects are otherwise cached for an hour, which would freeze the page), gzip-encoded, with their content
# types. THE BUCKET IS PUBLIC: this uploads only what nca.dashboard --static writes (the page and the
# runs' log/pool JSON), nothing else. Uploading needs gcloud credentials that may write the bucket.
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
SOURCES="pi vm"   # nca.dashboard STATIC_SOURCES
CC="no-cache, max-age=0"

source=pi every=0
while [ $# -gt 0 ]; do
  case $1 in
    --source) source=$2; shift 2 ;;
    --every) every=$2; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || { echo "usage: publish.sh [--source pi|vm] [--every SEC] RUN..." >&2; exit 1; }
[[ " $SOURCES " == *" $source "* ]] || { echo "--source must be one of: $SOURCES" >&2; exit 1; }
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

publish_once() {
  rm -rf "${out:?}"/*
  (cd "$ROOT" && "$PY" -m nca.dashboard --static "$out" --runs "$RUNS" --source "$source" --only "$@") || return 1
  local items=() d s
  for d in "$out"/*/; do [ -d "$d" ] && items+=("${d%/}"); done
  items+=("$out/runs-$source.json")
  if is_gs; then
    gc storage cp -r -Z --cache-control="$CC" --content-type=application/json "${items[@]}" "$BUCKET/dash/" >/dev/null || return 1
    for s in $SOURCES; do
      [ "$s" = "$source" ] && continue
      echo '[]' >"$out/runs-$s.json"
      gc storage cp -n --cache-control="$CC" --content-type=application/json "$out/runs-$s.json" \
        "$BUCKET/dash/runs-$s.json" >/dev/null 2>&1
    done
    gc storage cp -Z --cache-control="$CC" --content-type="text/html; charset=utf-8" "$out/index.html" \
      "$BUCKET/dash/index.html" >/dev/null || return 1
  else  # a local directory (tests)
    mkdir -p "$BUCKET/dash" && cp -r "${items[@]}" "$BUCKET/dash/" || return 1
    for s in $SOURCES; do [ -e "$BUCKET/dash/runs-$s.json" ] || echo '[]' >"$BUCKET/dash/runs-$s.json"; done
    cp "$out/index.html" "$BUCKET/dash/index.html"
  fi
  echo "$(date +%T) published $* to $BUCKET/dash/ (runs-$source.json)"
}

while :; do
  publish_once "$@" || echo "$(date +%T) publish failed" >&2
  [ "$every" -gt 0 ] 2>/dev/null || break
  sleep "$every"
done
