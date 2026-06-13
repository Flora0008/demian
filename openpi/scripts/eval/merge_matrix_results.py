"""Aggregate composite-matrix results across the 5 SLURM jobs.

Usage:
    python3 scripts/eval/merge_matrix_results.py \
        --jobs <bl-fix> <bl-dyn> <mx-fix> <mx-dyn> <mx-dyn-cap>
        [--out runs/robocasa/composite-matrix-summary]

Expects the same 5-config order as submit_composite_matrix.sh.

Produces a table:
    Task               | bl-fix  bl-dyn  mx-fix  mx-dyn  mx-dyn-cap
    -------------------+-----------------------------------------------
    PrepareCoffee      | overall SR
      phase1 (name)    | phase1 SR
      phase2 (name)    | phase2 SR
    ...
    TOTAL              | mean SR / mean phase1 / mean phase2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CONFIG_LABELS = ["bl-fix", "bl-dyn", "mx-fix", "mx-dyn", "mx-dyn-cap"]


def find_results(run_root: Path, job_id: int) -> Path | None:
    d = run_root / f"composite-{job_id}"
    if not d.exists():
        return None
    for root, _, files in os.walk(d):
        for f in files:
            if f.startswith("results_shard") and f.endswith(".json"):
                return Path(root) / f
    return None


def pct(hits: int, n: int) -> str:
    return f"{hits}/{n}" if n else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs=5, type=int, required=True,
                    help=f"Job IDs in order: {' '.join(CONFIG_LABELS)}")
    ap.add_argument("--run-root", default="runs/robocasa",
                    help="Directory containing composite-<JOB>/ result dirs")
    ap.add_argument("--out", default="runs/robocasa/composite-matrix-summary",
                    help="Output prefix (writes .txt and .json)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    datas: dict[str, dict] = {}  # config_label -> {task: task_result}
    for label, job_id in zip(CONFIG_LABELS, args.jobs):
        rf = find_results(run_root, job_id)
        if rf is None:
            print(f"WARNING: no results for {label} job {job_id}")
            datas[label] = {}
            continue
        datas[label] = json.loads(rf.read_text())

    # Union of tasks across configs (in stable order derived from first non-empty).
    task_order: list[str] = []
    for label in CONFIG_LABELS:
        for t in datas[label].keys():
            if t not in task_order:
                task_order.append(t)

    # Build table.
    lines = []
    header = f"{'Task':<26s} | " + "  ".join(f"{c:>10s}" for c in CONFIG_LABELS)
    lines.append(header)
    lines.append("-" * len(header))

    totals = {c: {"sr": [], "sg": {}} for c in CONFIG_LABELS}
    structured: dict[str, dict] = {}

    for task in task_order:
        row_sr = []
        for c in CONFIG_LABELS:
            t_res = datas[c].get(task)
            if not t_res:
                row_sr.append("   —  ")
                continue
            row_sr.append(pct(t_res.get("successes", 0), t_res.get("episodes", 0)))
            totals[c]["sr"].append(t_res.get("success_rate", 0.0))
        lines.append(f"{task:<26s} | " + "  ".join(f"{s:>10s}" for s in row_sr))

        # Per-subgoal rows (union of subgoal names across configs for this task).
        subgoal_names: list[str] = []
        for c in CONFIG_LABELS:
            sg = datas[c].get(task, {}).get("subgoal_success_rate", {})
            for n in sg.keys():
                if n not in subgoal_names:
                    subgoal_names.append(n)
        task_struct = {"overall": {}, "subgoals": {}}
        for c in CONFIG_LABELS:
            t_res = datas[c].get(task, {})
            task_struct["overall"][c] = {
                "success_rate": t_res.get("success_rate", 0.0),
                "successes": t_res.get("successes", 0),
                "episodes": t_res.get("episodes", 0),
            }
        for name in subgoal_names:
            row = []
            for c in CONFIG_LABELS:
                sg = datas[c].get(task, {}).get("subgoal_success_rate", {}).get(name)
                if not sg:
                    row.append("   —  ")
                    continue
                row.append(pct(sg.get("successes", 0), sg.get("episodes", 0)))
                totals[c]["sg"].setdefault(name, []).append(sg.get("success_rate", 0.0))
            lines.append(f"  {name:<24s} | " + "  ".join(f"{s:>10s}" for s in row))
            task_struct["subgoals"][name] = {
                c: datas[c].get(task, {}).get("subgoal_success_rate", {}).get(name)
                for c in CONFIG_LABELS
            }
        structured[task] = task_struct
        lines.append("")

    # Totals row.
    lines.append("=" * len(header))
    mean_row = []
    for c in CONFIG_LABELS:
        srs = totals[c]["sr"]
        mean_row.append(f"{(sum(srs)/len(srs)*100):.0f}%" if srs else "—")
    lines.append(f"{'Mean overall SR':<26s} | " + "  ".join(f"{s:>10s}" for s in mean_row))

    # Print + save.
    summary = "\n".join(lines)
    print(summary)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(summary + "\n")
    out.with_suffix(".json").write_text(json.dumps({
        "config_labels": CONFIG_LABELS,
        "job_ids": dict(zip(CONFIG_LABELS, args.jobs)),
        "tasks": structured,
    }, indent=2))
    print(f"\nSaved to {out.with_suffix('.txt')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
