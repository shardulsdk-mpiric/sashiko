# Response cache and tool order: how to reproduce

Evidence for the issue "local review: response cache misses for every stage
with tools". Two parts: unit tests on the fix branch, and a two-run check that
shows the cache hits before and after, without a model or a key.

- Upstream main: `d0e8bfc`
- Fix branch: `mpiric/cache-stable-tool-order` on
  https://github.com/shardulsdk-mpiric/sashiko
  - `c09c255` toolbox: list tool declarations in a stable order
  - `067f7e1` ai: forget a rejected answer so a retry asks the model again

## Unit tests

On the fix branch:

```
git clone https://github.com/sashiko-dev/sashiko && cd sashiko
git fetch https://github.com/shardulsdk-mpiric/sashiko mpiric/cache-stable-tool-order
git checkout FETCH_HEAD

cargo test --lib toolbox::tools_test::tests::test_tool_declarations_come_in_a_stable_order
cargo test --lib ai::cache
cargo test --lib local_review::tests::test_decorated_provider_passes_forget_to_the_cache
make check-pr
```

To see each test fail without its fix, put back the old version of the file
the fix changed, then rerun the test:

```
git checkout d0e8bfc -- src/toolbox/framework.rs     # undo the sort
cargo test --lib toolbox::tools_test::tests::test_tool_declarations_come_in_a_stable_order
#   declarations are not in name order (prints the scrambled list)
git checkout 067f7e1 -- src/toolbox/framework.rs

git checkout 067f7e1~1 -- src/ai/session.rs           # runner no longer calls forget
cargo test --lib ai::cache
#   the retry never reached the model: calls per attempt [3, 0]
#   left: [4, 0]  right: [4, 3]
git checkout 067f7e1 -- src/ai/session.rs
```

Note: `goose_cli` and `kiro_cli` each have a test that writes a fake
executable and runs it. Under the full suite these sometimes fail with "Text
file busy (os error 26)" and pass when run alone. They are unrelated to this
change.

## Two-run check

`two-runs.sh` runs the same local review twice with one shared response
cache and counts, for each run, the requests that reached the model and the
cache hits. The model is `fake-gemini.py`, a scripted stand-in for the
Gemini API, so this needs no key and costs nothing. Only python3 is needed.

Build both binaries, then point the script at each. Any repository and range
work; this uses the Sashiko repository and a fixed commit, so both runs
review the same thing:

```
# before: upstream main
git checkout d0e8bfc && cargo build --release && cp target/release/sashiko /tmp/sashiko-main
# after: the fix branch
git checkout FETCH_HEAD && cargo build --release && cp target/release/sashiko /tmp/sashiko-fix

path/to/two-runs.sh /tmp/sashiko-main . 'd0e8bfc~1..d0e8bfc'
path/to/two-runs.sh /tmp/sashiko-fix  . 'd0e8bfc~1..d0e8bfc'
```

What we got:

```
main   run 1: exit 0, requests sent to the model: 14, cache hits: 0
       run 2: exit 0, requests sent to the model: 12, cache hits: 2
fix    run 1: exit 0, requests sent to the model: 14, cache hits: 0
       run 2: exit 0, requests sent to the model: 0, cache hits: 14
```

On main the two hits are pre-screen and planning, the stages without tools.
The logs, settings and reports of each run are kept in the output directory
the script prints.

## How this differs from a normal review

No code is changed for the "before" run. The differences are all in how it
is run:

- The model is `fake-gemini.py`, reached through `GEMINI_BASE_URL`, with
  `GEMINI_API_KEY=fake`. Every conversation gets the same few tool calls and
  then a minimal answer that fits the requested schema.
- `XDG_DATA_HOME` points into the output directory, so the cache starts
  empty and the two runs share it and nothing else.
- `settings.toml` turns on `response_cache` (off by default) and
  `log_turns`, and the review runs with `--debug`, so requests and cache
  hits are logged. The counts are `grep -c` of "Sending Gemini request" and
  "Cache hit" in each run's log.
