"""Query episode traces or compare paired benchmark success latency (stdlib only)."""
import argparse
import json
import math
from pathlib import Path
import statistics


def query(path, event=None, operation=None, request_id=None):
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if event and row.get("event") != event:
            continue
        if operation and row.get("operation") != operation:
            continue
        if request_id and request_id not in {row.get("request_id"), row.get("payload", {}).get("request_id")}:
            continue
        yield row


def episodes(path):
    path = Path(path)
    if path.is_dir():
        data = {}
        for episode_file in sorted(path.rglob("episode_[0-9]*.json")):
            row = json.loads(episode_file.read_text())
            data.setdefault(row["task"], {"episode_results": []})["episode_results"].append(row)
    else:
        data = json.loads(path.read_text())
    output = {}
    for task, result in data.items():
        if not isinstance(result, dict):
            continue
        for index, row in enumerate(result.get("episode_results", [])):
            # Never guess seeds for old runs: they remain explicitly unpairable.
            ep = row.get("episode", result.get("episode_start", 0) + index)
            seed = row.get("seed")
            if seed is None:
                continue
            key = (task, ep, seed)
            if key in output:
                raise ValueError(f"Duplicate episode identity: {key}")
            output[key] = row
    return output


def compare(baseline, hierarchy):
    left, right = episodes(baseline), episodes(hierarchy)
    pairs = []
    for key in sorted(left.keys() & right.keys()):
        old, new = left[key], right[key]
        a, b = old.get("success_wall_seconds"), new.get("success_wall_seconds")
        valid_times = all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in (a, b))
        comparable = (old.get("success") and new.get("success") and valid_times
                      and old.get("timing_protocol") == new.get("timing_protocol") == "episode_start_before_reset_v1"
                      and old.get("max_steps") is not None and old.get("max_steps") == new.get("max_steps"))
        pairs.append({"task": key[0], "episode": key[1], "seed": key[2],
                      "baseline_success": old.get("success"), "hierarchy_success": new.get("success"),
                      "baseline_reason": old.get("termination_reason"), "hierarchy_reason": new.get("termination_reason"),
                      "baseline_success_seconds": a, "hierarchy_success_seconds": b,
                      "baseline_success_step": old.get("success_step"), "hierarchy_success_step": new.get("success_step"),
                      "latency_comparable": bool(comparable),
                      "delta_seconds": b - a if comparable else None,
                      "ratio": b / a if comparable and a > 0 else None})
    deltas = [r["delta_seconds"] for r in pairs if r["latency_comparable"]]
    return {"paired_episodes": len(pairs), "baseline_unmatched_with_seed": len(left) - len(pairs),
            "hierarchy_unmatched_with_seed": len(right) - len(pairs), "paired_success_latency_count": len(deltas),
            "baseline_successes": sum(bool(r["baseline_success"]) for r in pairs),
            "hierarchy_successes": sum(bool(r["hierarchy_success"]) for r in pairs),
            "median_added_seconds": statistics.median(deltas) if deltas else None,
            "mean_added_seconds": statistics.mean(deltas) if deltas else None,
            "note": "Unmatched/missing-seed runs excluded; missing timing is never estimated. "
                    "Confirm identical hardware, model weights, simulator setup and load before causal interpretation.",
            "episodes": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("query")
    trace.add_argument("path")
    trace.add_argument("--event")
    trace.add_argument("--operation")
    trace.add_argument("--request-id")
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline")
    comparison.add_argument("hierarchy")
    args = parser.parse_args()
    if args.command == "query":
        for row in query(args.path, args.event, args.operation, args.request_id):
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(json.dumps(compare(args.baseline, args.hierarchy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
