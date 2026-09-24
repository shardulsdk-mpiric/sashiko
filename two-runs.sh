#!/usr/bin/env bash
# Runs the same local review twice with one shared response cache, against
# the scripted endpoint in fake-gemini.py, and counts what reached the
# "model" on each run.
#
#   two-runs.sh SASHIKO_BINARY REPO RANGE [OUT_DIR]
#
# Needs python3. No key, no network: the fake key would be refused by
# Google if the base URL override ever failed.
set -u
BIN=$(realpath "$1"); REPO=$2; RANGE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${4:-$(mktemp -d)}
PORT=${PORT:-18766}
mkdir -p "$OUT/data" "$OUT/worktrees"
sed "s#WORKTREE_DIR#$OUT/worktrees#" "$HERE/settings.toml" > "$OUT/settings.toml"
python3 "$HERE/fake-gemini.py" "$PORT" > "$OUT/fake.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null' EXIT
sleep 1
cd "$REPO"
for run in 1 2; do
  env GEMINI_API_KEY=fake GEMINI_BASE_URL="http://127.0.0.1:$PORT" \
      GOOGLE_GEMINI_BASE_URL="http://127.0.0.1:$PORT" XDG_DATA_HOME="$OUT/data" \
      "$BIN" --debug review --settings "$OUT/settings.toml" "$RANGE" \
      > "$OUT/report$run.txt" 2> "$OUT/run$run.log"
  rc=$?
  echo "run $run: exit $rc, requests sent to the model: $(grep -c 'Sending Gemini request' "$OUT/run$run.log"), cache hits: $(grep -c 'Cache hit' "$OUT/run$run.log")"
done
echo "logs in $OUT"
