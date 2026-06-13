"""
Merge per-shard JSON results from run_robocasa_eval.py into a single summary table.

Usage:
    python scripts/eval/merge_results.py --results-dir runs/robocasa/2026-03-05_12-00-00/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tyro


def main(
    results_dir: str,
    output: Optional[str] = None,
):
    results_dir = Path(results_dir)
    shard_files = sorted(results_dir.glob("results_shard*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard result files found in {results_dir}")

    merged: dict[str, dict] = {}
    for f in shard_files:
        data = json.loads(f.read_text())
        merged.update(data)

    atomic_tasks    = {k: v for k, v in merged.items() if v.get("split") == "atomic" and "success_rate" in v}
    composite_tasks = {k: v for k, v in merged.items() if v.get("split") == "composite" and "success_rate" in v}
    errors          = {k: v for k, v in merged.items() if "error" in v}

    # Aggregate
    atomic_srs    = [v["success_rate"] for v in atomic_tasks.values()]
    composite_srs = [v["success_rate"] for v in composite_tasks.values()]
    composite_subtask_srs = [
        np.mean(ep["subtask_successes"])
        for v in composite_tasks.values()
        for ep in v.get("episode_results", [])
        if ep.get("subtask_successes")
    ]

    lines = []
    lines.append("=" * 64)
    lines.append("OpenPI \u00d7 RoboCasa 365 \u2014 Evaluation Summary")
    lines.append("=" * 64)
    lines.append("")

    lines.append("\u2500\u2500 Atomic tasks \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    lines.append(f"  Tasks evaluated : {len(atomic_tasks)}")
    if atomic_srs:
        lines.append(f"  Mean SR         : {np.mean(atomic_srs):.1%}")
        lines.append(f"  Median SR       : {np.median(atomic_srs):.1%}")
        lines.append("")
        lines.append(f"  {'Task':<45} SR")
        lines.append(f"  {'-'*45} ------")
        for task, result in sorted(atomic_tasks.items(), key=lambda x: -x[1]["success_rate"]):
            lines.append(f"  {task:<45} {result['success_rate']:.1%}")
    lines.append("")

    lines.append("\u2500\u2500 Composite tasks \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    lines.append(f"  Tasks evaluated : {len(composite_tasks)}")
    if composite_srs:
        lines.append(f"  Mean full SR    : {np.mean(composite_srs):.1%}")
        lines.append(f"  Median full SR  : {np.median(composite_srs):.1%}")
    if composite_subtask_srs:
        lines.append(f"  Mean subtask SR : {np.mean(composite_subtask_srs):.1%}")
    if composite_tasks:
        lines.append("")
        lines.append(f"  {'Task':<45} Full SR   Subtask SR")
        lines.append(f"  {'-'*45} --------- ----------")
        for task, result in sorted(composite_tasks.items(), key=lambda x: -x[1]["success_rate"]):
            ep_subtasks = [
                np.mean(ep["subtask_successes"])
                for ep in result.get("episode_results", [])
                if ep.get("subtask_successes")
            ]
            subtask_str = f"{np.mean(ep_subtasks):.1%}" if ep_subtasks else "  n/a"
            lines.append(f"  {task:<45} {result['success_rate']:.1%}    {subtask_str}")
    lines.append("")

    if errors:
        lines.append("\u2500\u2500 Errors (tasks skipped) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        for task, err in errors.items():
            lines.append(f"  {task}: {err['error']}")
        lines.append("")

    all_srs = atomic_srs + composite_srs
    if all_srs:
        lines.append(f"Overall mean SR (all {len(all_srs)} tasks): {np.mean(all_srs):.1%}")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)

    out_path = Path(output) if output else results_dir / "summary.txt"
    out_path.write_text(report)
    print(f"\nSummary written to {out_path}")

    # Also save merged JSON
    merged_path = results_dir / "results_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2))
    print(f"Merged JSON written to {merged_path}")


if __name__ == "__main__":
    tyro.cli(main)
