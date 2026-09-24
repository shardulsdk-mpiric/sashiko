#!/usr/bin/env bash
# Runs one local review with the response cache on, against fake-gemini.py
# set to reject its first three pre-screen answers, so the first attempt
# fails and the review is retried. Prints whether the retry got through.
#
#   retry-check.sh SASHIKO_BINARY REPO RANGE [OUT_DIR]
set -u
BIN=$(realpath "$1"); REPO=$2; RANGE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${4:-$(mktemp -d)}
PORT=${PORT:-18767}
mkdir -p "$OUT/data" "$OUT/worktrees"
sed "s#WORKTREE_DIR#$OUT/worktrees#" "$HERE/settings.toml" > "$OUT/settings.toml"
FAKE_REJECT_FIRST=3 python3 "$HERE/fake-gemini.py" "$PORT" > "$OUT/fake.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null' EXIT
sleep 1
cd "$REPO"
env GEMINI_API_KEY=fake GEMINI_BASE_URL="http://127.0.0.1:$PORT" \
    GOOGLE_GEMINI_BASE_URL="http://127.0.0.1:$PORT" XDG_DATA_HOME="$OUT/data" \
    "$BIN" --debug review --settings "$OUT/settings.toml" "$RANGE" \
    > "$OUT/report.txt" 2> "$OUT/run.log"
rc=$?
echo "exit $rc, attempts: $(( $(grep -c 'Restarting AI review' "$OUT/run.log") + 1 )), requests sent to the model: $(grep -c 'Sending Gemini request' "$OUT/run.log"), cache hits: $(grep -c 'Cache hit' "$OUT/run.log")"
echo "logs in $OUT"
