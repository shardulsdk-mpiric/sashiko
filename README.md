# Sashiko tool call trace, 2026-09-24

One local Sashiko review of one kernel patch, with every tool call recorded
per stage. Collected for the discussion on sashiko-dev/sashiko#492.

## What was run

- Sashiko: branch `mpiric/stage-instrumentation` of this fork, commit
  `f5b0734`, which is upstream main `06ba233` plus two commits: the stage
  name in the log context tag and the `SASHIKO_TOOL_TRACE` recorder, and
  keeping thinking tokens in the token totals. The duplicate tool call
  guard is upstream's, unchanged.
- Model: `gemini-3.1-pro-preview`, upstream's default `[ai]` settings
  (`settings.toml`). Logging on, response cache off.
- Patch: "mptcp: do not drop partial packets", stable commit
  `bb37498a99e4` (upstream `50c2d91c5dfa`), range
  `bb37498a99e4^..bb37498a99e4`.

## Files

`gemini-3.1-pro-preview_12427_guard-on/`

| file | what it is |
|---|---|
| `trace.jsonl` | one line per tool call (`event: tool_call`) and per finished stage (`event: stage_end`) |
| `analysis.txt` | output of `tooltrace-analyze.py` on this run |
| `sashiko.log` | the review's `--debug` log with `ai.log_turns`, one block per model turn |
| `report.txt` | the review's output |
| `spend.json` | token totals and a cost estimate at list prices, not a bill |
| `meta.json`, `settings.toml` | run metadata and the exact settings |

Tool call fields: `stage` and `stage_run` (one execution of one stage),
`batch` (the model turn within the stage), `seq` (call number within the
stage), `name`, `args` (as the model sent them), `repeat_of` and `prev_seq`
(earlier identical call in the same stage, if any), `refused` (the
duplicate guard answered it), `cache_hit` (the ToolBox cache answered it),
`tool_error`, `result_bytes`.

To recompute: `python3 tooltrace-analyze.py gemini-3.1-pro-preview_12427_guard-on`

## Limits

- One patch, one run, temperature 1.0. A data point, not a rate.
- Local paths in the log and settings are replaced with `<workdir>/`.
