"""Replay stored PPCS action logs and recompute the loose `place_anytime` metric
under the new contact-free check (XY-distance to pan center only).

Reads every `runs/robocasa/sanity5task_*_loose/seed*/results_merged.json`, finds
PickPlaceCounterToStove entries, replays each episode by feeding the stored
action_log into a freshly-reset env (same per-episode seed convention as the
original eval), evaluates the new loose check at each step, and updates the
file in place. Strict `success` is left untouched.

Run inside the eval container (cosmos-wam-eval.sqsh) on CPU. No policy server,
no GPU, no rendering needed beyond what env.step does internally.
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

# Bring run_robocasa_eval helpers into scope.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.run_robocasa_eval import (  # noqa: E402
    _to_gym_action_dict,
    _check_loose_success,
    build_env,
)


def _resolve_seed(cell_dir: Path) -> tuple[int, int] | None:
    """Return (base_seed, ppcs_task_index_in_shard) for the cell's PPCS shard.

    The original eval computes per-episode seed as:
        ep_seed = base_seed + task_index_in_shard * 10000 + ep
    where `task_index_in_shard` is PPCS's position in the SHARD-FILTERED task
    list (run_robocasa_eval.py line ~660 filters via `i % num_shards == shard`).

    args_shard*.json stores the *full* tasks list passed via TASK env var, plus
    `task_shard` and `num_shards`; reconstruct the per-shard list with the same
    filter to recover the index PPCS occupied in `task_names`.
    """
    for args_path in sorted(cell_dir.glob("args_shard*.json")):
        try:
            with open(args_path) as f:
                args = json.load(f)
        except Exception:
            continue
        all_tasks = args.get("tasks", [])
        if "PickPlaceCounterToStove" not in all_tasks:
            continue
        seed = args.get("seed")
        if seed is None:
            continue
        task_shard = int(args.get("task_shard", 0))
        num_shards = int(args.get("num_shards", 1))
        shard_tasks = [t for i, t in enumerate(all_tasks) if i % num_shards == task_shard]
        if "PickPlaceCounterToStove" not in shard_tasks:
            continue  # this shard didn't run PPCS — try the next args_shard*.json
        return int(seed), shard_tasks.index("PickPlaceCounterToStove")
    return None


class _ReplayArgs:
    """Minimal stand-in for the Args dataclass — only fields build_env reads."""
    def __init__(self, camera_height: int = 256, camera_width: int = 256):
        self.camera_height = camera_height
        self.camera_width = camera_width


def _replay_one_episode(env, action_log, seed, verbose=False):
    """Reset env with seed, step through action_log, return (place_anytime, place_first_step)."""
    env.reset(seed=seed)
    raw = env.unwrapped
    place_anytime = False
    place_first_step = None
    min_dist = float("inf")
    min_dist_step = -1
    for step, action in enumerate(action_log):
        a = np.asarray(action, dtype=np.float32)
        if a.shape[0] < 7:
            break
        env.step(_to_gym_action_dict(a[:7]))
        obj_pos = np.asarray(raw.sim.data.body_xpos[raw.obj_body_id["obj"]][:2])
        pan_pos = np.asarray(raw.sim.data.body_xpos[raw.obj_body_id["container"]][:2])
        d = float(np.linalg.norm(obj_pos - pan_pos))
        if d < min_dist:
            min_dist = d
            min_dist_step = step
        if not place_anytime and _check_loose_success(env):
            place_anytime = True
            place_first_step = step
            break
    if verbose:
        try:
            obj_pos = np.asarray(raw.sim.data.body_xpos[raw.obj_body_id["obj"]])
            pan_pos = np.asarray(raw.sim.data.body_xpos[raw.obj_body_id["container"]])
            print(f"      [diag] min_xy_dist={min_dist:.4f}m @ step {min_dist_step};  "
                  f"final obj_xyz={obj_pos.round(3).tolist()};  pan_xyz={pan_pos.round(3).tolist()}")
        except Exception as e:
            print(f"      [diag] error: {e}")
    return place_anytime, place_first_step


def _process_cell(merged_path: Path, dry_run: bool, seed_base_override: int | None) -> dict:
    """Update PPCS entries in one results_merged.json. Returns a per-episode summary.

    The per-episode seed is read from this cell's args_shard*.json (the shard that
    ran PPCS) — i.e. the same `seed` arg the original eval received. With a 4-shard
    layout and the 5-task list, PPCS is task index 3 → runs on shard 3, so we look
    at args_shard3.json; fall back to any args_shard*.json that lists PPCS.
    """
    summary = {"path": str(merged_path), "n_eps": 0, "old_count": 0, "new_count": 0, "changes": []}
    with open(merged_path) as f:
        data = json.load(f)
    if "PickPlaceCounterToStove" not in data:
        summary["note"] = "no PPCS entry"
        return summary

    # Resolve per-cell (seed_base, ppcs_task_index).
    if seed_base_override is not None:
        seed_base, task_index = seed_base_override, 0
    else:
        resolved = _resolve_seed(merged_path.parent)
        if resolved is None:
            summary["note"] = "could not resolve seed"
            return summary
        seed_base, task_index = resolved
    summary["seed_base"] = seed_base
    summary["task_index"] = task_index
    ppcs = data["PickPlaceCounterToStove"]
    eps = ppcs.get("episode_results", [])
    if not eps:
        summary["note"] = "no episode_results"
        return summary

    env = build_env("PickPlaceCounterToStove", _ReplayArgs())
    try:
        for i, ep in enumerate(eps):
            action_log = ep.get("action_log", [])
            if not action_log:
                continue
            old_place = bool(ep.get("place_anytime"))
            ep_seed = seed_base + task_index * 10000 + i
            try:
                new_place, new_first = _replay_one_episode(env, action_log, ep_seed, verbose=dry_run)
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
        # Update aggregate place_anytime fields.
        n = ppcs.get("episodes", len(eps))
        ppcs["place_anytime_count"] = summary["new_count"]
        ppcs["place_anytime_rate"] = (summary["new_count"] / n) if n else 0.0
        # Backup + write.
        bak = merged_path.with_suffix(".json.bak_precontactdrop")
        if not bak.exists():
            shutil.copy2(merged_path, bak)
        with open(merged_path, "w") as f:
            json.dump(data, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/robocasa",
                   help="Search root for sanity5task_*_loose/seed*/results_merged.json")
    p.add_argument("--seed-base", type=int, default=None,
                   help="Override the per-cell seed lookup; default reads from args_shard*.json")
    p.add_argument("--dry-run", action="store_true", help="Don't write back; just report changes")
    p.add_argument("--filter", default="", help="Only process cells whose path contains this substring")
    args = p.parse_args()

    pattern = os.path.join(args.root, "sanity5task_*_loose", "seed*", "results_merged.json")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if args.filter:
        paths = [p for p in paths if args.filter in str(p)]
    print(f"[INFO] Found {len(paths)} cells under {args.root}")
    print(f"[INFO] dry_run={args.dry_run}  seed_base={args.seed_base}")

    n_changed = 0
    n_cells = 0
    total_old = total_new = 0
    for path in paths:
        try:
            summary = _process_cell(path, dry_run=args.dry_run, seed_base_override=args.seed_base)
        except Exception as e:
            print(f"  [ERR] {path}: {e}")
            traceback.print_exc()
            continue
        if summary.get("n_eps", 0) == 0:
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

    print(f"\n[INFO] {n_cells} PPCS cells processed; {n_changed} changed.")
    print(f"[INFO] Old place_anytime total = {total_old}; new = {total_new}; delta = {total_new-total_old:+d}.")


if __name__ == "__main__":
    main()
