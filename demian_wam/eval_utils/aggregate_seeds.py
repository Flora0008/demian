"""
Aggregate per-seed merged results into a single multi-seed summary.

Reads `seed{42,43,...}/results_merged.json` under a results dir, then reports
per-task mean +/- std and overall averages across seeds, matching the reward-set
/ test-set protocol in workspace/openpi/docs/evaluation_protocol.md
(per-task SR = mean over seeds; overall = mean over tasks of per-task means).

Usage:
    python eval_utils/aggregate_seeds.py \\
        --results-dir runs/robocasa/reward_baseline_v2gbs256_iter35k

Optional: --seeds 42,43,44,45,46 (default auto-detects seed* subdirs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tyro


def main(
    results_dir: str,
    seeds: Optional[str] = None,
    output: Optional[str] = None,
):
    root = Path(results_dir)
    if seeds:
        seed_list = [int(s) for s in seeds.split(",")]
    else:
        seed_list = sorted(
            int(p.name[4:])
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith("seed") and p.name[4:].isdigit()
        )
    if not seed_list:
        raise FileNotFoundError(f"No seed* subdirs in {root}")

    # task -> seed -> success_rate
    per_task: dict[str, dict[int, float]] = {}
    missing: list[str] = []
    for s in seed_list:
        merged_path = root / f"seed{s}" / "results_merged.json"
        if not merged_path.exists():
            missing.append(f"seed{s}")
            continue
        data = json.loads(merged_path.read_text())
        for task, result in data.items():
            sr = result.get("success_rate")
            if sr is None:
                continue
            per_task.setdefault(task, {})[s] = float(sr)

    if missing:
        print(f"[WARN] Missing results_merged.json for: {', '.join(missing)}")

    lines = []
    lines.append("=" * 78)
    lines.append(f"Multi-seed reward-set summary  ({root.name})")
    lines.append(f"Seeds: {seed_list}")
    lines.append("=" * 78)
    lines.append("")
    header = f"  {'Task':<42} {'mean':>7}  {'std':>7}  {'per-seed values':<24}"
    lines.append(header)
    lines.append(f"  {'-'*42} {'-'*7}  {'-'*7}  {'-'*24}")

    task_means: list[float] = []
    task_stds: list[float] = []
    summary: dict[str, dict] = {}

    for task in sorted(per_task.keys()):
        sr_by_seed = per_task[task]
        srs = [sr_by_seed[s] for s in seed_list if s in sr_by_seed]
        if not srs:
            continue
        mean_sr = float(np.mean(srs))
        std_sr = float(np.std(srs, ddof=0))  # population std across the 5 seeds
        task_means.append(mean_sr)
        task_stds.append(std_sr)
        per_seed_str = " ".join(f"{sr_by_seed.get(s, float('nan')):.0%}".rjust(4) for s in seed_list)
        lines.append(f"  {task:<42} {mean_sr:>6.1%}  {std_sr:>6.1%}  {per_seed_str}")
        summary[task] = {
            "seeds": seed_list,
            "per_seed_sr": [sr_by_seed.get(s) for s in seed_list],
            "mean": mean_sr,
            "std": std_sr,
        }

    lines.append("")
    if task_means:
        avg_mean = float(np.mean(task_means))
        avg_std = float(np.mean(task_stds))
        lines.append(f"  Average per-task mean SR : {avg_mean:.1%}")
        lines.append(f"  Average per-task std SR  : {avg_std:.1%}  (across-seed variability)")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)

    out_txt = Path(output) if output else root / "multi_seed_summary.txt"
    out_txt.write_text(report)
    print(f"\nText summary written to {out_txt}")

    out_json = root / "multi_seed_summary.json"
    out_json.write_text(json.dumps(
        {
            "seeds": seed_list,
            "per_task": summary,
            "average_per_task_mean": float(np.mean(task_means)) if task_means else None,
            "average_per_task_std": float(np.mean(task_stds)) if task_stds else None,
        },
        indent=2,
    ))
    print(f"JSON summary written to {out_json}")


if __name__ == "__main__":
    tyro.cli(main)
