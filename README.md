# Response cache and tool order: how to reproduce

Evidence for the issue "local review: response cache misses for every stage
with tools". Three parts: unit tests on the fix series, a two-run check that
shows the cache hits, and a retry check that shows a failed review being
rescued by its retry. None of it needs a model or a key.

To get this kit, next to where you will clone Sashiko:

```
git clone --depth 1 --branch cache-tool-order-2026-09-24-v2 \
    https://github.com/shardulsdk-mpiric/sashiko kit
```

- Upstream main: `d0e8bfc`
- Fix series: branch `mpiric/cache-tool-order-series` on
  https://github.com/shardulsdk-mpiric/sashiko
  1. `fa6e1a3` ai: forget a rejected answer so a retry asks the model again
  2. `8fea21e` ai: forget a truncated answer so a retry asks the model again
  3. `3781577` reviewer: pass a worker's forget to the daemon's cache
  4. `851c77d` local_review: retry without the cache when the answers were accepted
  5. `54e9779` toolbox: list tool declarations in a stable order

## Unit tests

```
git clone https://github.com/sashiko-dev/sashiko && cd sashiko
git fetch https://github.com/shardulsdk-mpiric/sashiko mpiric/cache-tool-order-series
git checkout FETCH_HEAD

cargo test --lib ai::cache                                   # patches 1 and 2
cargo test --lib local_review::tests::test_decorated_provider_passes_forget_to_the_cache
cargo test --lib reviewer::tests::test_run_review_tool_passes_a_worker_forget_to_the_provider
cargo test --lib local_review::tests::test_attempt_ai_settings_bypasses_only_the_cache
cargo test --lib toolbox::tools_test::tests::test_tool_declarations_come_in_a_stable_order
make check-pr
```

`make check-pr` also passes at each commit of the series on its own.
Now and then a `goose_cli` or `kiro_cli` test that writes a fake executable
and runs it fails with "Text file busy (os error 26)", and passes alone. The
likely cause is another test starting a process while that file is still
open for writing. Patch 3 adds one more test that starts a process, so it
may show slightly more often; none of the series touches those tests.

To see the tests fail without the fixes, put back main's version of the file
a fix changed, then rerun:

```
git checkout d0e8bfc -- src/toolbox/framework.rs      # undo patch 5
cargo test --lib toolbox::tools_test::tests::test_tool_declarations_come_in_a_stable_order
#   declarations are not in name order (prints the scrambled list)
git checkout FETCH_HEAD -- src/toolbox/framework.rs

git checkout d0e8bfc -- src/ai/session.rs             # undo patches 1 and 2 in the runner
cargo test --lib ai::cache
#   the retry never reached the model: calls per attempt [3, 0]
#   left: [4, 0]  right: [4, 3]
#   left: [1, 0]  right: [1, 1]
git checkout FETCH_HEAD -- src/ai/session.rs
```

## Building the two binaries

```
git checkout d0e8bfc && cargo build --release && cp target/release/sashiko /tmp/sashiko-main
git checkout FETCH_HEAD && cargo build --release && cp target/release/sashiko /tmp/sashiko-series
```

Any repository and range work for the checks below; these use the Sashiko
repository and a fixed commit, so every run reviews the same thing.

## Two-run check: the cache

`two-runs.sh` runs the same local review twice with one shared response
cache and counts, for each run, the requests that reached the model and the
cache hits.

```
../kit/two-runs.sh /tmp/sashiko-main   . 'd0e8bfc~1..d0e8bfc'
../kit/two-runs.sh /tmp/sashiko-series . 'd0e8bfc~1..d0e8bfc'
```

What we got:

```
main    run 1: exit 0, requests sent to the model: 14, cache hits: 0
        run 2: exit 0, requests sent to the model: 12, cache hits: 2
series  run 1: exit 0, requests sent to the model: 14, cache hits: 0
        run 2: exit 0, requests sent to the model: 0, cache hits: 14
```

On main the two hits are pre-screen and planning, the stages without tools.

## Retry check: a failed attempt

`retry-check.sh` runs one local review with the cache on, with the stand-in
set to reject its first three pre-screen answers, so the first attempt fails
and the review is retried.

```
../kit/retry-check.sh /tmp/sashiko-main   . 'd0e8bfc~1..d0e8bfc'
../kit/retry-check.sh /tmp/sashiko-series . 'd0e8bfc~1..d0e8bfc'
```

What we got:

```
main    exit 1, attempts: 3, requests sent to the model: 3, cache hits: 6
series  exit 0, attempts: 2, requests sent to the model: 17, cache hits: 0
```

On main, attempts 2 and 3 were served the rejected answers from the cache and
never reached the model, so the review failed. With the series, attempt 2
asked again and the review completed. (The failed review exits 1 on main
because the error leaves `main` through `result?` before the check that
would exit 3.)

## How these checks differ from a normal review

No code is changed for the "main" runs. The differences are all in how they
are run:

- The model is `fake-gemini.py`, reached through `GEMINI_BASE_URL`, with
  `GEMINI_API_KEY=fake`. Every conversation with tools gets the same three
  turns of tool calls, then an answer that fits the requested schema.
- `XDG_DATA_HOME` points into the output directory, so the cache starts
  empty and belongs to that check alone.
- `settings.toml` turns on `response_cache` (off by default) and
  `log_turns`, and the review runs with `--debug`, so requests and cache
  hits are logged. The counts are `grep -c` of "Sending Gemini request",
  "Cache hit" and "Restarting AI review" in the logs, which the scripts
  keep in the output directory they print.
