"""Aggregate one-episode RoboCasa quick-eval shards.

The sharded eval layout is:

    <results-dir>/<Task>_ep<N>/<timestamp>/results_shard0.json

This script writes a compact aggregate_summary.json without copying the large
per-step action logs from each shard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default="aggregate_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    aggregate = _aggregate(results_dir)
    output_path = results_dir / args.output
    output_path.write_text(json.dumps(aggregate, indent=2))

    print(f"Wrote {output_path}")
    for task, task_summary in aggregate["tasks"].items():
        print(
            f"{task}: {task_summary['successes']}/{task_summary['episodes']} = "
            f"{100.0 * task_summary['success_rate']:.1f}%"
        )
    overall = aggregate["overall"]
    print(
        f"Overall: {overall['successes']}/{overall['episodes']} = "
        f"{100.0 * overall['success_rate']:.1f}%"
    )


def _aggregate(results_dir: Path) -> dict:
    task_records: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*/*/results_shard*.json")):
        if path.name == "aggregate_summary.json":
            continue
        payload = json.loads(path.read_text())
        shard = path.parents[1].name
        for task, task_result in payload.items():
            instruction = task_result.get("instruction", "")
            episode_start = int(task_result.get("episode_start", 0))
            for offset, episode in enumerate(task_result.get("episode_results", [])):
                record = {
                    "shard": shard,
                    "episode_index": episode_start + offset,
                    "success": bool(episode.get("success", False)),
                    "steps": int(episode.get("steps", 0)),
                    "instruction": instruction,
                    "path": str(path),
                }
                if "subgoals" in episode:
                    record["subgoals"] = episode["subgoals"]
                task_records.setdefault(task, []).append(record)

    tasks = {}
    total_successes = 0
    total_episodes = 0
    for task in sorted(task_records):
        records = sorted(task_records[task], key=lambda item: (item["episode_index"], item["shard"], item["path"]))
        successes = sum(1 for record in records if record["success"])
        episodes = len(records)
        total_successes += successes
        total_episodes += episodes
        tasks[task] = {
            "episode_results": records,
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else 0.0,
            "mean_steps": sum(record["steps"] for record in records) / episodes if episodes else 0.0,
        }

    return {
        "results_dir": str(results_dir),
        "tasks": tasks,
        "overall": {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        },
    }


if __name__ == "__main__":
    main()

