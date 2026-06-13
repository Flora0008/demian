"""Generic replay of stored action logs to recompute the loose `place_anytime`
metric under whatever `_check_loose_success` is defined in run_robocasa_eval.py
RIGHT NOW (so a code change to the loose check is automatically picked up).

Generalisation of `replay_loose_check_ppcs.py`: takes `--task <name>` and works
for any task that:
  (a) has an entry in `_check_loose_success`'s task dispatch, and
  (b) stored `episode_results[i]["action_log"]` in the saved JSON.

Per-episode seed convention (from `run_robocasa_eval.py`):
    ep_seed = base_seed + task_index_in_shard * 10000 + ep
The cell's args[.json|_shard*.json] holds (base_seed, tasks, task_shard, num_shards);
the script reconstructs the per-shard ordering with `i % num_shards == task_shard`
to find the target task's index, then mirrors the eval's seed math.

Run inside the eval container (cosmos-wam-eval.sqsh). MuJoCo env replay is CPU-bound
(rendering not needed for state inspection — env.step still drives sim physics).

Usage:
    python eval_utils/replay_loose_check.py --task CloseBlenderLid \
        --results-glob 'runs/robocasa/reward17task_*/seed*/results_merged.json' \
        [--dry-run] [--filter <substr>]

The merged JSON is rewritten in place; a `.bak_loose_<task>` backup is created
the first time a cell is touched, so reruns are idempotent.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.run_robocasa_eval import (  # noqa: E402
    _to_gym_action_dict,
    _check_loose_success,
    build_env,
)


class _ReplayArgs:
    """Minimal stand-in for Args — only fields build_env reads."""
    def __init__(self, camera_height: int = 256, camera_width: int = 256):
        self.camera_height = camera_height
        self.camera_width = camera_width


def _resolve_seed(cell_dir: Path, task: str) -> tuple[int, int] | None:
    """Return (base_seed, task_index_in_shard) for `task` inside this cell.

    Tries `args.json` first (NUM_SHARDS=1 layout, single args file per cell),
    then falls back to args_shard*.json (multi-shard layout). Picks the shard
    whose filtered task list contains `task`.
    """
    candidates = []
    if (cell_dir / "args.json").exists():
        candidates.append(cell_dir / "args.json")
    candidates.extend(sorted(cell_dir.glob("args_shard*.json")))

    for args_path in candidates:
        try:
            with open(args_path) as f:
                args = json.load(f)
        except Exception:
            continue
        all_tasks = args.get("tasks", []) or []
        if task not in all_tasks:
            continue
        seed = args.get("seed")
        if seed is None:
            continue
        task_shard = int(args.get("task_shard", 0))
        num_shards = int(args.get("num_shards", 1))
        shard_tasks = [t for i, t in enumerate(all_tasks) if i % num_shards == task_shard]
        if task not in shard_tasks:
            continue
        return int(seed), shard_tasks.index(task)
    return None


def _replay_one_episode(env, action_log, seed):
    """Reset env with seed, step through action_log, return (place_anytime, place_first_step).
    Loose check is sticky-once-True (mirrors live eval)."""
    env.reset(seed=seed)
    place_anytime = False
    place_first_step = None
    for step, action in enumerate(action_log):
        a = np.asarray(action, dtype=np.float32)
        if a.shape[0] < 7:
            break
        env.step(_to_gym_action_dict(a[:7]))
        if not place_anytime and _check_loose_success(env):
            place_anytime = True
            place_first_step = step
            break
    return place_anytime, place_first_step


def _process_cell(merged_path: Path, task: str, dry_run: bool) -> dict:
    """Update one cell's task entry. Returns a summary dict."""
    summary = {"path": str(merged_path), "task": task,
               "n_eps": 0, "old_count": 0, "new_count": 0, "changes": []}
    with open(merged_path) as f:
        data = json.load(f)
    if task not in data:
        summary["note"] = f"no '{task}' entry"
        return summary

    resolved = _resolve_seed(merged_path.parent, task)
    if resolved is None:
        summary["note"] = "could not resolve seed (no args.json with this task)"
        return summary
    seed_base, task_index = resolved
    summary["seed_base"] = seed_base
    summary["task_index"] = task_index

    entry = data[task]
    eps = entry.get("episode_results", [])
    if not eps:
        summary["note"] = "no episode_results"
        return summary

    env = build_env(task, _ReplayArgs())
    try:
        for i, ep in enumerate(eps):
            action_log = ep.get("action_log", [])
            if not action_log:
                continue
            old_place = bool(ep.get("place_anytime"))
            ep_seed = seed_base + task_index * 10000 + i
            try:
                new_place, new_first = _replay_one_episode(env, action_log, ep_seed)
            except Exception as e:
                summary["changes"].append({"ep": i, "error": f"{type(e).__name__}: {e}"})
                continue
            summary["n_eps"] += 1
            if old_place:
                summary["old_count"] += 1
            if new_place:
                summary["new_count"] += 1
            if old_place != new_place or ep.get("place_first_step") != new_first:
                summary["changes"].append({"ep": i, "old": old_place, "new": new_place,
                                           "old_first": ep.get("place_first_step"), "new_first": new_first})
            if not dry_run:
                ep["place_anytime"] = new_place
                ep["place_first_step"] = new_first
    finally:
        env.close()

    if not dry_run and summary["n_eps"] > 0:
        n = entry.get("episodes", len(eps))
        # Recompute aggregate place_anytime from the (possibly updated) per-episode flags
        new_count = sum(1 for ep in eps if ep.get("place_anytime"))
        entry["place_anytime_count"] = new_count
        entry["place_anytime_rate"] = (new_count / n) if n else 0.0

        bak = merged_path.with_suffix(f".json.bak_loose_{task}")
        if not bak.exists():
            shutil.copy2(merged_path, bak)
        with open(merged_path, "w") as f:
            json.dump(data, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   help="Task name to replay (e.g. CloseBlenderLid, PickPlaceCounterToStove)")
    p.add_argument("--results-glob",
                   default="runs/robocasa/reward17task_*/seed*/results_merged.json",
                   help="Glob for results_merged.json files to process.")
    p.add_argument("--filter", default="",
                   help="Only process cells whose path contains this substring")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write back; just report what would change")
    p.add_argument("--skip-existing-backup", action="store_true",
                   help="Skip cells where a .bak_loose_<task> backup already exists "
                        "(i.e. that cell has been processed by an earlier replay run). "
                        "Use this to resume a partial replay sweep efficiently.")
    args = p.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.results_glob))
    if args.filter:
        paths = [p for p in paths if args.filter in str(p)]
    if args.skip_existing_backup:
        bak_suffix = f".json.bak_loose_{args.task}"
        before = len(paths)
        paths = [p for p in paths if not p.with_suffix(bak_suffix).exists()]
        print(f"[INFO] --skip-existing-backup: {before - len(paths)} cells already processed, "
              f"{len(paths)} remaining")
    print(f"[INFO] task={args.task}")
    print(f"[INFO] Found {len(paths)} candidate cells")
    print(f"[INFO] dry_run={args.dry_run}")

    n_cells = 0
    n_changed = 0
    total_old = total_new = 0
    for path in paths:
        try:
            summary = _process_cell(path, args.task, dry_run=args.dry_run)
        except Exception as e:
            print(f"  [ERR] {path}: {e}")
            traceback.print_exc()
            continue
        if summary.get("n_eps", 0) == 0:
            note = summary.get("note", "")
            if note:
                print(f"   {path}: skipped ({note})")
            continue
        n_cells += 1
        total_old += summary["old_count"]
        total_new += summary["new_count"]
        delta = summary["new_count"] - summary["old_count"]
        marker = "  *" if summary["changes"] else "   "
        print(f"{marker} {path}: {summary['old_count']}->{summary['new_count']} "
              f"(n_eps={summary['n_eps']}, delta={delta:+d})")
        if summary["changes"]:
            n_changed += 1
            for c in summary["changes"]:
                print(f"        {c}")

    print(f"\n[INFO] {args.task}: {n_cells} cells processed; {n_changed} changed.")
    print(f"[INFO] Old place_anytime total = {total_old}; new = {total_new}; delta = {total_new-total_old:+d}.")


if __name__ == "__main__":
    main()
