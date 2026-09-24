#!/usr/bin/env python3
"""Analyse the tool trace of one or more tooltrace-run.sh runs.

  tooltrace-analyze.py RUN_DIR [RUN_DIR ...] [--json]

Everything below is computed from trace.jsonl alone, which records each
tool call with its stage run, batch and sequence number. What each figure
means:

  within-stage repeat   the same tool name and raw arguments as an earlier
                        call in the same execution of the same stage
  cross-stage duplicate a call no earlier call in its own stage matches, but
                        one in another stage of the same patch does
  refused               the duplicate guard answered it with an error
                        (arm A only; arm B has no guard)
  cache hit             the ToolBox answered from its result cache without
                        running the tool (normalized arguments, so a hit can
                        also come from a call whose raw arguments differ)

After a refusal, the next batch of the same stage is classified as: asks for
the refused call again, moves on to other calls, or there is no next batch
(the stage answered). A refused call counts as "cache had it" when an
earlier call with identical raw arguments ran without a tool error anywhere
in the same patch, so the ToolBox would have served it.
"""
import collections
import json
import os
import sys


def load(run):
    recs = [json.loads(l) for l in open(os.path.join(run, "trace.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(run, "meta.json"))) if os.path.exists(os.path.join(run, "meta.json")) else {}
    spend = json.load(open(os.path.join(run, "spend.json"))) if os.path.exists(os.path.join(run, "spend.json")) else {}
    return recs, meta, spend


def key(r):
    return (r["name"], json.dumps(r["args"], sort_keys=True))


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def bucket(b):
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20)):
        if lo <= b <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return "21+"


def analyse(run):
    recs, meta, spend = load(run)
    calls = [r for r in recs if r["event"] == "tool_call"]
    ends = [r for r in recs if r["event"] == "stage_end"]
    calls.sort(key=lambda r: (r["ts_ms"], r["stage_run"], r["seq"]))

    # Cross-stage: first time each key was seen per patch, and by which stage run.
    first_seen = {}
    cross = 0
    for r in calls:
        k = (r["patch"], key(r))
        if r["repeat_of"] is None and k in first_seen and first_seen[k] != r["stage_run"]:
            cross += 1
            r["_cross"] = True
        first_seen.setdefault(k, r["stage_run"])

    within = [r for r in calls if r["repeat_of"] is not None]
    refused = [r for r in calls if r["refused"]]
    hits = [r for r in calls if r["cache_hit"]]

    dist = collections.Counter(r["seq"] - r["prev_seq"] for r in within)
    by_batch = collections.defaultdict(lambda: [0, 0])
    for r in calls:
        b = bucket(r["batch"])
        by_batch[b][1] += 1
        by_batch[b][0] += r["repeat_of"] is not None
    by_tool = collections.defaultdict(lambda: [0, 0])
    for r in calls:
        by_tool[r["name"]][1] += 1
        by_tool[r["name"]][0] += r["repeat_of"] is not None

    # After a refusal: what the stage's next batch did.
    batches = collections.defaultdict(list)
    for r in calls:
        batches[(r["stage_run"], r["batch"])].append(r)
    after = collections.Counter()
    for r in refused:
        nxt = batches.get((r["stage_run"], r["batch"] + 1))
        if not nxt:
            after["no next batch (stage answered)"] += 1
        elif any(key(n) == key(r) for n in nxt):
            after["asked for the same call again"] += 1
        else:
            after["moved on to other calls"] += 1

    ran_ok = collections.defaultdict(list)
    for r in calls:
        if not r["refused"] and not r["tool_error"]:
            ran_ok[(r["patch"], key(r))].append(r["ts_ms"])
    cache_had = sum(
        1 for r in refused
        if any(ts <= r["ts_ms"] for ts in ran_ok.get((r["patch"], key(r)), []))
    )

    stages = []
    for e in sorted(ends, key=lambda e: e["stage_run"]):
        mine = [r for r in calls if r["stage_run"] == e["stage_run"]]
        stages.append({
            "stage": e["stage"], "run": e["stage_run"], "ok": e["ok"],
            "calls": len(mine),
            "repeats": sum(r["repeat_of"] is not None for r in mine),
            "refused": sum(r["refused"] for r in mine),
            "cache_hits": sum(r["cache_hit"] for r in mine),
            "batches": e["batches"], "tokens_total": e.get("tokens_total"),
            "error": (e.get("error") or "")[:120],
        })

    return {
        "run": os.path.basename(os.path.normpath(run)),
        "arm": meta.get("arm"), "model": meta.get("model"), "commit": meta.get("commit", "")[:12],
        "exit_code": meta.get("exit_code"), "usd_estimate": spend.get("usd"),
        "guard_tripped": spend.get("tripped"),
        "stages": len(ends), "stages_failed": sum(not e["ok"] for e in ends),
        "stages_without_tool_calls": sum(1 for s in stages if s["calls"] == 0),
        "tool_calls": len(calls),
        "within_stage_repeats": len(within),
        "cross_stage_duplicates": cross,
        "refused": len(refused),
        "cache_hits": len(hits),
        "cache_hits_on_within_stage_repeats": sum(r["cache_hit"] for r in within),
        "cache_hits_on_cross_stage": sum(r["cache_hit"] for r in calls if r.get("_cross")),
        "refused_that_cache_had": cache_had,
        "after_refusal": dict(after),
        "repeat_distance": dict(sorted(dist.items())),
        "repeat_rate_by_batch": {b: v for b, v in sorted(by_batch.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))},
        "repeat_rate_by_tool": dict(by_tool),
        "per_stage": stages,
    }


def report(a):
    n = a["tool_calls"]
    out = [f"## {a['run']}", "",
           f"arm {a['arm']}  model {a['model']}  commit {a['commit']}  exit {a['exit_code']}  "
           f"est ${a['usd_estimate']}  guard {a['guard_tripped'] or 'not tripped'}", "",
           f"stages {a['stages']} ({a['stages_failed']} failed, {a['stages_without_tool_calls']} with no tool calls)",
           f"tool calls {n}",
           f"  within-stage repeats   {a['within_stage_repeats']:5d}  {pct(a['within_stage_repeats'], n)}",
           f"  cross-stage duplicates {a['cross_stage_duplicates']:5d}  {pct(a['cross_stage_duplicates'], n)}",
           f"  refused by the guard   {a['refused']:5d}  {pct(a['refused'], n)}",
           f"  ToolBox cache hits     {a['cache_hits']:5d}  {pct(a['cache_hits'], n)}"
           f"  (within-stage {a['cache_hits_on_within_stage_repeats']}, cross-stage {a['cache_hits_on_cross_stage']})",
           f"  refused, cache had it  {a['refused_that_cache_had']:5d}  of {a['refused']}", ""]
    if a["after_refusal"]:
        out.append("after a refusal, the stage's next batch:")
        out += [f"  {v:5d}  {k}" for k, v in a["after_refusal"].items()]
        out.append("")
    if a["repeat_distance"]:
        out.append("within-stage repeat distance (calls back to the previous identical call):")
        cum, tot = 0, sum(a["repeat_distance"].values())
        for d, c in a["repeat_distance"].items():
            cum += c
            out.append(f"  {d:3d}: {c:4d}  {pct(c, tot):>6}  cumulative {pct(cum, tot)}")
        out.append("")
    out.append("within-stage repeat rate by batch (model turn with tool calls):")
    out += [f"  {b:>5}: {r}/{t}  {pct(r, t)}" for b, (r, t) in a["repeat_rate_by_batch"].items()]
    out.append("")
    out.append("within-stage repeat rate by tool:")
    out += [f"  {k:16s} {r}/{t}  {pct(r, t)}" for k, (r, t) in sorted(a["repeat_rate_by_tool"].items())]
    out.append("")
    out.append("per stage:")
    out.append("  run stage                      ok    calls rep  ref  hit  batches tokens_total")
    for s in a["per_stage"]:
        out.append(f"  {s['run']:3d} {s['stage'][:26]:26s} {str(s['ok']):5s} {s['calls']:5d} {s['repeats']:4d} "
                   f"{s['refused']:4d} {s['cache_hits']:4d} {s['batches']:7d} {s['tokens_total']}")
        if s["error"]:
            out.append(f"      error: {s['error']}")
    return "\n".join(out)


def compare(results):
    rows = ["tool_calls", "within_stage_repeats", "cross_stage_duplicates", "refused",
            "cache_hits", "refused_that_cache_had", "stages", "stages_failed",
            "stages_without_tool_calls", "usd_estimate", "exit_code"]
    w = max(len(r["run"]) for r in results)
    out = ["## side by side", "", " " * 28 + "".join(f"{r['arm'] or '?':>{w + 2}}" for r in results)]
    for row in rows:
        out.append(f"{row:28s}" + "".join(f"{str(r[row]):>{w + 2}}" for r in results))
    return "\n".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    if not args:
        sys.exit(__doc__)
    results = [analyse(r) for r in args]
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1))
    else:
        print("\n\n".join(report(r) for r in results))
        if len(results) > 1:
            print("\n\n" + compare(results))
