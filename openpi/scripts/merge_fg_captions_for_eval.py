"""Merge per-task/subtask fine-grained caption files into a single JSONL.

Reads from the atomic_split directory structure:
    atomic_split/TaskName/SubtaskName/date/lerobot/meta/captions_fine_grain_*.jsonl
    atomic_split/TaskName/date/lerobot/meta/captions_fine_grain_*.jsonl  (single-subtask)

Produces a merged JSONL compatible with FineGrainedCaptionProvider, with:
    - tasks: [gym_task_name]  (e.g., "CoffeeSetupMug")
    - annotations_per_step: {step_str: {caption_key: text}}
    - For multi-subtask tasks, subtask 2 steps are offset by subtask 1's length

Usage:
    python scripts/merge_fg_captions_for_eval.py \
        --input /path/to/fsw/.../atomic_split \
        --output /path/to/fsw/.../fg_captions_eval.jsonl \
        --tasks CoffeeSetupMug OpenDrawer ...   # optional; defaults to all
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

CAPTION_FILENAME = "captions_fine_grain_Qwen3-VL-30B-A3B-Instruct.jsonl"

# Subtask ordering: lower number = earlier phase
SUBTASK_PRIORITY = {
    "ApproachToObject": 0, "ApproachToObject_0": 0, "ApproachToObject_1": 0,
    "GraspObject": 1, "GraspObject_0": 1, "GraspObject_1": 1,
    "PickObject": 2,
    "OpenDoor": 3, "OpenDoor_0": 3, "OpenDoor_1": 3,
    "PlaceObject": 4,
    "PullOpen": 4,
    "PushClose": 4,
    "TwistKnob": 4,
    "PressButton": 4,
    "CloseDoor": 4, "CloseDoor_0": 4, "CloseDoor_1": 4,
    "SlideRack": 4,
    "TurnSinkSpout": 4,
}


def find_caption_file(subtask_dir: Path) -> Path | None:
    """Find the caption JSONL file inside a subtask or date directory."""
    for p in subtask_dir.rglob(CAPTION_FILENAME):
        return p
    return None


def discover_task(task_dir: Path) -> list[tuple[str, Path]]:
    """Discover subtasks and their caption files for a task directory.

    Returns list of (subtask_name, caption_file_path) sorted by execution order.
    """
    subtasks = []

    for child in sorted(task_dir.iterdir()):
        if not child.is_dir():
            continue
        # Date directories (e.g., "20250820") indicate single-subtask tasks
        if child.name.isdigit():
            cap = find_caption_file(child)
            if cap:
                subtasks.append(("_single", cap))
        # Otherwise it's a subtask directory
        elif child.name not in ("huggingface", "lerobot"):
            cap = find_caption_file(child)
            if cap:
                subtasks.append((child.name, cap))

    # Sort by subtask priority
    subtasks.sort(key=lambda x: SUBTASK_PRIORITY.get(x[0], 99))
    return subtasks


def load_caption_file(path: Path) -> dict[int, dict[int, dict[str, str]]]:
    """Load a caption JSONL file.

    Returns: {episode_index: {step: {caption_key: text}}}
    """
    data: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            ep_idx = entry["episode_index"]
            step = entry["step"]
            data[ep_idx][step] = entry["annotations"]
    return dict(data)


def merge_subtasks(
    subtask_list: list[tuple[str, Path]],
    caption_stride: int,
) -> dict[int, dict[int, dict[str, str]]]:
    """Merge subtask captions by concatenating steps sequentially.

    For multi-subtask tasks, subtask 2's steps are offset by
    (subtask 1's max_step + caption_stride).

    Returns: {episode_index: {step: {caption_key: text}}}
    """
    if len(subtask_list) == 1:
        return load_caption_file(subtask_list[0][1])

    # Load all subtasks
    subtask_data = []
    for name, path in subtask_list:
        data = load_caption_file(path)
        subtask_data.append((name, data))

    # Find common episodes (same episode_index across subtasks)
    common_episodes = set(subtask_data[0][1].keys())
    for _, data in subtask_data[1:]:
        common_episodes &= set(data.keys())

    merged: dict[int, dict[int, dict[str, str]]] = {}

    for ep_idx in sorted(common_episodes):
        merged_steps: dict[int, dict[str, str]] = {}
        offset = 0

        for _name, data in subtask_data:
            ep_captions = data[ep_idx]
            max_step = 0
            for step, captions in sorted(ep_captions.items()):
                merged_steps[step + offset] = captions
                max_step = max(max_step, step)

            # Next subtask starts after this one
            offset += max_step + caption_stride

        merged[ep_idx] = merged_steps

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge FG captions for eval")
    parser.add_argument("--input", required=True, help="Path to atomic_split directory")
    parser.add_argument("--output", required=True, help="Output merged JSONL path")
    parser.add_argument("--tasks", nargs="*", help="Specific task names (default: all)")
    parser.add_argument("--caption-stride", type=int, default=4, help="Caption step stride in data")
    parser.add_argument("--max-episodes", type=int, default=10, help="Max episodes per task to include")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Discover tasks
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted([
            d.name for d in input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    global_ep_idx = 0
    total_entries = 0

    with open(args.output, "w") as out:
        for task_name in task_names:
            # Find task directory (try exact name and -split variant)
            task_dir = None
            for variant in [task_name, f"{task_name}-split"]:
                candidate = input_dir / variant
                if candidate.is_dir():
                    task_dir = candidate
                    break

            if task_dir is None:
                print(f"  SKIP: {task_name} (no directory found)")
                continue

            subtasks = discover_task(task_dir)
            if not subtasks:
                print(f"  SKIP: {task_name} (no caption files)")
                continue

            subtask_names = [s[0] for s in subtasks]
            merged = merge_subtasks(subtasks, args.caption_stride)

            ep_count = 0
            for ep_idx in sorted(merged.keys()):
                if ep_count >= args.max_episodes:
                    break

                annotations_per_step = {str(k): v for k, v in sorted(merged[ep_idx].items())}

                entry = {
                    "episode_index": global_ep_idx,
                    "tasks": [task_name],
                    "length": max(merged[ep_idx].keys()) + args.caption_stride,
                    "annotations_per_step": annotations_per_step,
                    "source_episode": ep_idx,
                    "subtasks": subtask_names,
                }
                out.write(json.dumps(entry) + "\n")
                global_ep_idx += 1
                ep_count += 1
                total_entries += 1

            print(f"  {task_name}: {ep_count} episodes, subtasks={subtask_names}, steps={len(annotations_per_step)} per ep")

    print(f"\nWrote {total_entries} entries to {args.output}")


if __name__ == "__main__":
    main()
