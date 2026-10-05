#!/bin/bash
# publish.sh [--source pi|vm] [--every SEC] RUN...: the static dashboard of these runs (runs/RUN) to
# $BUCKET/dash/, the public page https://storage.googleapis.com/<bucket>/dash/index.html.
# publish.sh --weights-only [--keep-page] [--every SEC] RUN...: only the play page and these runs' weights.
#
#   nca/cloud/publish.sh pure-a                  # once, from this Pi (source pi)
#   nca/cloud/publish.sh --every 60 pure-a       # in a loop (Ctrl-C to stop)
#   nca/cloud/publish.sh --weights-only --every 120 ha-big hb-big   # the cloud runs' weights, from the Pi
# The VM's sidecar runs it with --source vm for the cloud runs. Each publisher writes only its own list,
# runs-<source>.json, plus <run>/log.json and <run>/pool.json (so run names must differ between the
# two); the page merges runs-pi.json and runs-vm.json. A missing list is created empty, never overwritten
# (no-clobber), so the page doesn't 404. Objects go up with Cache-Control "no-cache, max-age=0" (public
# objects are otherwise cached for an hour, which would freeze the page), gzip-encoded, with their content
# types. From the Pi (not --source vm: the VM's CPU is for training, and it has no build of the play page)
# it also uploads play.html (dist/nca.html: npm run build) and <run>/weights.json for each run with a
# checkpoint, which the page's Play link opens. --weights-only uploads just those two kinds of file, for
# runs whose logs the VM publishes: never runs-*.json or a run's log/pool JSON, and index.html only with
# --keep-page; a loop sends a file again only once it has changed. --keep-page: a VM on its own code
# uploads its own dash/index.html every minute, which may differ from this checkout's (an older page, or
# one missing a later change); this looks at the served page every KEEP_SEC (5) s and, when it isn't
# byte-identical to this checkout's, puts this checkout's page back. THE BUCKET IS
# PUBLIC: this uploads only what nca.dashboard --static writes (the page, the play page and the runs'
# log/pool/weights JSON), nothing else. Uploading needs gcloud credentials that may write the bucket.
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=common.sh
. "$(dirname "$0")/common.sh"
SOURCES="pi vm"   # nca.dashboard STATIC_SOURCES
CC="no-cache, max-age=0"

source=pi every=0 weights_only=0 keep=0
KEEP_SEC=${KEEP_SEC:-5}
runs=()  # options may come before or after the run names
while [ $# -gt 0 ]; do
  case $1 in
    --source) source=$2; shift 2 ;;
    --every) every=$2; shift 2 ;;
    --weights-only) weights_only=1; shift ;;
    --keep-page) keep=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    -*) echo "publish.sh: unknown option $1" >&2; exit 1 ;;
    *) runs+=("$1"); shift ;;
  esac
done
set -- "${runs[@]}"
[ $# -ge 1 ] || { echo "usage: publish.sh [--source pi|vm | --weights-only] [--keep-page] [--every SEC] RUN..." >&2; exit 1; }
[[ " $SOURCES " == *" $source "* ]] || { echo "--source must be one of: $SOURCES" >&2; exit 1; }
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

# put FILE REL TYPE: one file to $BUCKET/dash/REL, gzip-encoded, no-cache
put() {
  if is_gs; then gc storage cp -Z --cache-control="$CC" --content-type="$3" "$1" "$BUCKET/dash/$2" >/dev/null
  else mkdir -p "$(dirname "$BUCKET/dash/$2")" && cp "$1" "$BUCKET/dash/$2"; fi
}

publish_once() {
  rm -rf "${out:?}"/*
  local nw=() items=() d s
  [ "$source" = vm ] && nw=(--no-weights)
  (cd "$ROOT" && "$PY" -m nca.dashboard --static "$out" --runs "$RUNS" --source "$source" "${nw[@]}" --only "$@") || return 1
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
  else  # a local directory (tests)
    mkdir -p "$BUCKET/dash" && cp -r "${items[@]}" "$BUCKET/dash/" || return 1
    for s in $SOURCES; do [ -e "$BUCKET/dash/runs-$s.json" ] || echo '[]' >"$BUCKET/dash/runs-$s.json"; done
  fi
  # the play page before the index whose Play links open it
  if [ -f "$out/play.html" ]; then put "$out/play.html" play.html "text/html; charset=utf-8" || return 1; fi
  put "$out/index.html" index.html "text/html; charset=utf-8" || return 1
  echo "$(date +%T) published $* to $BUCKET/dash/ (runs-$source.json)"
}

declare -A sent=()  # --weights-only: REL -> sha256 of what this process last uploaded there
weights_once() {
  rm -rf "${out:?}"/*
  (cd "$ROOT" && "$PY" -m nca.dashboard --static "$out" --runs "$RUNS" --weights-only --only "$@") || return 1
  [ -f "$out/play.html" ] || { echo "$(date +%T) no play page (npm run build in $ROOT)" >&2; return 1; }
  local f rel h n=0 names=()
  for f in "$out"/*/weights.json "$out/play.html"; do  # the weights before the page that fetches them
    [ -f "$f" ] || continue
    rel=${f#"$out"/}
    h=$(sha256sum <"$f")
    [ "$rel" = play.html ] || names+=("${rel%/weights.json}")
    [ "${sent[$rel]:-}" = "$h" ] && continue
    if [ "$rel" = play.html ]; then put "$f" "$rel" "text/html; charset=utf-8" || return 1
    else put "$f" "$rel" application/json || return 1; fi
    sent[$rel]=$h n=$((n + 1))
  done
  echo "$(date +%T) weights of ${names[*]:-no run} and play.html in $BUCKET/dash/: $n file(s) sent, the rest unchanged"
}

# --keep-page: when the served index.html is a new upload (its generation) that isn't byte-identical to
# this checkout's page, put this checkout's back. Reads the public URL with curl, so no gcloud call unless
# it uploads. Exact match, not just "has a Play link": a VM on code newer than the Play link but older
# than a later page change (the board-size control, say) would pass a Play-link-only check and never get
# replaced, leaving that change missing from the public page indefinitely.
PAGE_URL=https://storage.googleapis.com/${BUCKET#gs://}/dash/index.html
seen=
keep_page() {
  is_gs && [ -f "$out/index.html" ] || return 0
  local g page
  g=$(curl -fsSI --max-time 10 "$PAGE_URL" | tr -d '\r' | awk -F': ' 'tolower($1) == "x-goog-generation" { print $2 }')
  [ -n "$g" ] && [ "$g" != "$seen" ] || return 0
  page=$(curl -fs --max-time 20 --compressed "$PAGE_URL") || return 0  # (not piped to grep -q: pipefail)
  seen=$g
  [ "$(printf '%s' "$page" | sha256sum)" = "$(sha256sum <"$out/index.html")" ] && return 0
  put "$out/index.html" index.html "text/html; charset=utf-8" &&
    echo "$(date +%T) dash/index.html wasn't this checkout's page (a VM upload?): put ours back"
}

while :; do
  if [ "$weights_only" = 1 ]; then weights_once "$@" || echo "$(date +%T) weights publish failed" >&2
  else publish_once "$@" || echo "$(date +%T) publish failed" >&2; fi
  [ "$keep" = 1 ] && keep_page
  [ "$every" -gt 0 ] 2>/dev/null || break
  if [ "$keep" = 1 ]; then
    end=$((SECONDS + every))
    while [ "$SECONDS" -lt "$end" ]; do sleep "$KEEP_SEC"; keep_page; done
  else sleep "$every"; fi
done
