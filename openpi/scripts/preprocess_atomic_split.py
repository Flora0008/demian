"""Preprocess atomic_split RoboCasa dataset for LM-augmented training.

Converts per-subtask LeRobot datasets (161 subtask dirs) into a single merged
dataset with:
  - 7-dim arm-only actions (stripped from 12-dim LeRobot format)
  - Packed npz images (3 cameras per episode, decoded from mp4)
  - Reasoning text from Qwen3-VL captions stored in episodes.jsonl

Source layout:
    atomic_split/{Task}/{Subtask}/{date}/lerobot/
        data/chunk-000/episode_{ep:06d}.parquet
        videos/chunk-000/{camera_key}/episode_{ep:06d}.mp4
        meta/info.json, episodes.jsonl, tasks.jsonl
        meta/captions_{caption_model}.jsonl

Output layout:
    {output_dir}/
        data/chunk-{chunk:03d}/episode_{ep:06d}.parquet
        packed/episode_{ep:06d}.npz
        meta/info.json, episodes.jsonl, tasks.jsonl

Usage:
    python scripts/preprocess_atomic_split.py \
        --input_dir /path/to/atomic_split \
        --output_dir /path/to/atomic_split_processed \
        --num_workers 16
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

CAMERA_KEYS = [
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]
SHORT_NAMES = ["eye_in_hand", "agentview_left", "agentview_right"]


def find_subtask_lerobot_dirs(input_dir: str) -> list[tuple[str, str, str]]:
    """Discover all lerobot dirs under atomic_split.

    Handles two directory structures:
      - Multi-subtask: {Task}/{Subtask}/{date}/lerobot/
      - Single-task:   {Task}/{date}/lerobot/  (e.g., CloseOven)

    Returns list of (task_name, subtask_name, lerobot_dir_path).
    For single-task dirs, subtask_name equals task_name.
    """
    results = []
    for task in sorted(os.listdir(input_dir)):
        task_path = os.path.join(input_dir, task)
        if not os.path.isdir(task_path):
            continue

        # Check if this is a single-task dir: {Task}/{date}/lerobot/ exists directly.
        found_single = False
        for child in os.listdir(task_path):
            lerobot_dir = os.path.join(task_path, child, "lerobot")
            if os.path.isdir(lerobot_dir):
                results.append((task, task, lerobot_dir))
                found_single = True
                break

        if found_single:
            continue

        # Multi-subtask: {Task}/{Subtask}/{date}/lerobot/
        for subtask in sorted(os.listdir(task_path)):
            subtask_path = os.path.join(task_path, subtask)
            if not os.path.isdir(subtask_path):
                continue
            for date_dir in os.listdir(subtask_path):
                lerobot_dir = os.path.join(subtask_path, date_dir, "lerobot")
                if os.path.isdir(lerobot_dir):
                    results.append((task, subtask, lerobot_dir))
                    break  # One date dir per subtask
    return results


def episode_has_base_motion(parquet_path: str, threshold: float) -> bool:
    """Check if an episode has significant base motion in action[0:4]."""
    table = pq.read_table(parquet_path)
    actions = np.stack(table.column("action").to_pylist())
    return float(np.abs(actions[:, 0:4]).max()) > threshold


def strip_action_dims(action: np.ndarray) -> np.ndarray:
    """Strip base_motion[0:4] and control_mode[4], keep action[5:12] -> 7 dims."""
    return action[5:12].astype(np.float64)


def load_captions(meta_dir: str, caption_model: str) -> dict[int, dict]:
    """Load per-episode captions from captions JSONL file."""
    caption_file = os.path.join(meta_dir, f"captions_{caption_model}.jsonl")
    captions = {}
    if os.path.exists(caption_file):
        with open(caption_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                captions[entry["episode_index"]] = entry.get("annotations", {})
    return captions


def load_fine_grain_captions(meta_dir: str, caption_model: str) -> dict[int, dict[int, dict]]:
    """Load per-step fine-grained captions. Returns dict[ep_idx -> dict[step -> captions_dict]]."""
    caption_file = os.path.join(meta_dir, f"captions_fine_grain_{caption_model}.jsonl")
    captions: dict[int, dict[int, dict]] = {}
    if os.path.exists(caption_file):
        with open(caption_file) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                ep_idx = entry["episode_index"]
                step = entry["step"]
                if ep_idx not in captions:
                    captions[ep_idx] = {}
                captions[ep_idx][step] = entry.get("annotations", {})
    return captions


def extract_video_frames(video_path: str) -> np.ndarray:
    """Extract all frames from an mp4 video, returning (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.empty((0, 256, 256, 3), dtype=np.uint8)
    return np.stack(frames)


def process_episode(args: tuple) -> tuple[int, int, str | None]:
    """Process a single episode: read parquet, strip actions, extract video, write output.

    Returns (global_ep_idx, num_frames, error_string_or_None).
    """
    (
        global_ep_idx,
        src_lerobot_dir,
        src_ep_idx,
        src_chunks_size,
        new_task_idx,
        global_frame_start,
        output_dir,
        chunks_size,
    ) = args

    try:
        # Read source parquet.
        src_chunk_idx = src_ep_idx // src_chunks_size
        src_parquet = os.path.join(
            src_lerobot_dir, "data", f"chunk-{src_chunk_idx:03d}", f"episode_{src_ep_idx:06d}.parquet"
        )
        table = pq.read_table(src_parquet)
        df = table.to_pandas()
        num_frames = len(df)

        # Strip action dimensions: 12-dim -> 7-dim.
        df["action"] = df["action"].apply(strip_action_dims)

        # Re-index for merged dataset.
        df["episode_index"] = global_ep_idx
        df["frame_index"] = range(num_frames)
        df["index"] = range(global_frame_start, global_frame_start + num_frames)
        df["task_index"] = new_task_idx

        # Write output parquet with only the columns we need.
        keep_cols = ["observation.state", "action", "task_index", "episode_index", "frame_index", "index"]
        df_out = df[[c for c in keep_cols if c in df.columns]].copy()

        out_chunk_idx = global_ep_idx // chunks_size
        out_data_dir = os.path.join(output_dir, "data", f"chunk-{out_chunk_idx:03d}")
        os.makedirs(out_data_dir, exist_ok=True)
        out_parquet = os.path.join(out_data_dir, f"episode_{global_ep_idx:06d}.parquet")
        pq.write_table(pa.Table.from_pandas(df_out, preserve_index=False), out_parquet)

        # Extract video frames for all 3 cameras and pack into npz.
        arrays = {}
        for cam_key, short_name in zip(CAMERA_KEYS, SHORT_NAMES):
            video_path = os.path.join(
                src_lerobot_dir,
                "videos",
                f"chunk-{src_chunk_idx:03d}",
                cam_key,
                f"episode_{src_ep_idx:06d}.mp4",
            )
            if not os.path.exists(video_path):
                return (global_ep_idx, 0, f"Missing video: {video_path}")
            frames = extract_video_frames(video_path)
            if len(frames) != num_frames:
                return (
                    global_ep_idx,
                    0,
                    f"Frame mismatch ep {global_ep_idx}: video {cam_key} has {len(frames)}, parquet has {num_frames}",
                )
            arrays[short_name] = frames

        packed_dir = os.path.join(output_dir, "packed")
        os.makedirs(packed_dir, exist_ok=True)
        npz_path = os.path.join(packed_dir, f"episode_{global_ep_idx:06d}.npz")
        np.savez(npz_path, **arrays)

        return (global_ep_idx, num_frames, None)
    except Exception as e:
        return (global_ep_idx, 0, f"Error processing ep {global_ep_idx}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess atomic_split RoboCasa dataset for LM training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/path/to/data/robocasa/datasets/v1.0/pretrain/atomic_split",
        help="Root directory of atomic_split (contains {Task}/{Subtask}/ dirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the processed dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for video extraction",
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default="Qwen3-VL-30B-A3B-Instruct",
        help="Caption model name (matches filename captions_{model}.jsonl)",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="reasoning-short",
        help="Which caption field to use as reasoning text",
    )
    parser.add_argument(
        "--chunks_size",
        type=int,
        default=1000,
        help="Number of episodes per parquet chunk",
    )
    parser.add_argument(
        "--exclude_navigation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes with base_motion > threshold (default: True)",
    )
    parser.add_argument(
        "--nav_threshold",
        type=float,
        default=0.01,
        help="Max absolute base_motion value to allow when filtering navigation",
    )
    parser.add_argument(
        "--fine_grain",
        action="store_true",
        default=False,
        help="Load per-step fine-grained captions and store annotations_per_step in episodes.jsonl",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Shard index for distributed processing (0-based). If set, only processes episodes where global_ep_idx %% num_shards == shard_index.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for distributed processing",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    # --- Step 1: Discover all subtask datasets ---
    print(f"Scanning {input_dir} for subtask datasets...")
    subtask_dirs = find_subtask_lerobot_dirs(input_dir)
    print(f"Found {len(subtask_dirs)} subtask directories")

    # --- Step 2: Scan episodes, filter navigation, assign global indices ---
    # Each entry: (task, subtask, lerobot_dir, src_ep_idx, ep_length, task_desc,
    #              reasoning_text, src_chunks_size)
    all_episodes = []
    skipped_nav = 0
    total_source = 0

    for task_name, subtask_name, lerobot_dir in tqdm.tqdm(subtask_dirs, desc="Scanning subtasks"):
        meta_dir = os.path.join(lerobot_dir, "meta")

        with open(os.path.join(meta_dir, "info.json")) as f:
            info = json.load(f)
        src_chunks_size = info.get("chunks_size", 1000)

        # Load episode metadata.
        episodes_meta = []
        with open(os.path.join(meta_dir, "episodes.jsonl")) as f:
            for line in f:
                if line.strip():
                    episodes_meta.append(json.loads(line))
        total_source += len(episodes_meta)

        # Load tasks.
        tasks_map = {}
        tasks_file = os.path.join(meta_dir, "tasks.jsonl")
        if os.path.exists(tasks_file):
            with open(tasks_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        tasks_map[entry["task_index"]] = entry["task"]

        # Load captions.
        captions = load_captions(meta_dir, args.caption_model)
        fg_captions = load_fine_grain_captions(meta_dir, args.caption_model) if args.fine_grain else {}

        # Prefer original (composite) task description if available.
        original_task_desc = None
        orig_ep_file = os.path.join(meta_dir, "original_episodes.jsonl")
        if os.path.exists(orig_ep_file):
            with open(orig_ep_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    orig_entry = json.loads(first_line)
                    if orig_entry.get("tasks"):
                        original_task_desc = orig_entry["tasks"][0]

        for ep_meta in episodes_meta:
            ep_idx = ep_meta["episode_index"]
            ep_length = ep_meta["length"]
            task_desc = original_task_desc or (ep_meta["tasks"][0] if ep_meta.get("tasks") else tasks_map.get(0, f"{task_name}/{subtask_name}"))

            # Filter navigation episodes.
            if args.exclude_navigation:
                chunk_idx = ep_idx // src_chunks_size
                parquet_path = os.path.join(
                    lerobot_dir, "data", f"chunk-{chunk_idx:03d}", f"episode_{ep_idx:06d}.parquet"
                )
                if episode_has_base_motion(parquet_path, args.nav_threshold):
                    skipped_nav += 1
                    continue

            # Get reasoning text from captions.
            ep_captions = captions.get(ep_idx, {})
            reasoning_text = ep_captions.get(args.caption_key, "")

            # Get per-step fine-grained captions.
            annotations_per_step = {}
            if args.fine_grain and ep_idx in fg_captions:
                annotations_per_step = {str(step): caps for step, caps in fg_captions[ep_idx].items()}

            all_episodes.append({
                "task": task_name,
                "subtask": subtask_name,
                "lerobot_dir": lerobot_dir,
                "src_ep_idx": ep_idx,
                "length": ep_length,
                "task_desc": task_desc,
                "reasoning_text": reasoning_text,
                "annotations_per_step": annotations_per_step,
                "src_chunks_size": src_chunks_size,
            })

    print(f"\nFiltering summary:")
    print(f"  Total source episodes: {total_source}")
    print(f"  Skipped (navigation): {skipped_nav}")
    print(f"  Episodes to process: {len(all_episodes)}")

    # Assign global episode indices and compute global frame starts.
    global_frame_start = 0
    for i, ep in enumerate(all_episodes):
        ep["global_ep_idx"] = i
        ep["global_frame_start"] = global_frame_start
        global_frame_start += ep["length"]
    total_frames = global_frame_start

    # Build task index mapping.
    task_to_index = {}
    tasks_list = []
    for ep in all_episodes:
        desc = ep["task_desc"]
        if desc not in task_to_index:
            task_to_index[desc] = len(tasks_list)
            tasks_list.append({"task_index": len(tasks_list), "task": desc})
        ep["new_task_idx"] = task_to_index[desc]

    # --- Step 3: Process episodes in parallel ---
    # If sharding, only process this shard's episodes.
    is_sharded = args.shard_index is not None
    if is_sharded:
        shard_episodes = [ep for ep in all_episodes if ep["global_ep_idx"] % args.num_shards == args.shard_index]
        print(f"\nShard {args.shard_index}/{args.num_shards}: processing {len(shard_episodes)}/{len(all_episodes)} episodes")
    else:
        shard_episodes = all_episodes

    work_items = [
        (
            ep["global_ep_idx"],
            ep["lerobot_dir"],
            ep["src_ep_idx"],
            ep["src_chunks_size"],
            ep["new_task_idx"],
            ep["global_frame_start"],
            output_dir,
            args.chunks_size,
        )
        for ep in shard_episodes
    ]

    print(f"\nProcessing {len(work_items)} episodes with {args.num_workers} workers...")
    errors = []
    processed_frames = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_episode, w): w[0] for w in work_items}
        with tqdm.tqdm(total=len(work_items), desc="Processing") as pbar:
            for future in as_completed(futures):
                ep_idx, n_frames, error = future.result()
                if error:
                    errors.append(error)
                else:
                    processed_frames += n_frames
                pbar.update(1)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # --- Step 4: Write metadata (only shard 0 or non-sharded runs) ---
    if is_sharded and args.shard_index != 0:
        print(f"\nShard {args.shard_index} done. Processed {processed_frames} frames ({len(errors)} errors).")
        return

    total_episodes = len(all_episodes)
    total_chunks = (total_episodes + args.chunks_size - 1) // args.chunks_size

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_list),
        "total_chunks": total_chunks,
        "chunks_size": args.chunks_size,
        "fps": 20,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"dtype": "float64", "shape": [16]},
            "action": {"dtype": "float64", "shape": [7]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # tasks.jsonl
    with open(os.path.join(output_dir, "meta", "tasks.jsonl"), "w") as f:
        for task_entry in tasks_list:
            f.write(json.dumps(task_entry) + "\n")

    # episodes.jsonl — includes reasoning_text and provenance.
    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in all_episodes:
            entry = {
                "episode_index": ep["global_ep_idx"],
                "tasks": [ep["task_desc"]],
                "length": ep["length"],
                "reasoning_text": ep["reasoning_text"],
                "source_task": ep["task"],
                "source_subtask": ep["subtask"],
            }
            if ep["annotations_per_step"]:
                entry["annotations_per_step"] = ep["annotations_per_step"]
            f.write(json.dumps(entry) + "\n")

    # modality.json
    modality = {
        "state": {
            "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
            "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
            "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
            "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
            "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16},
        },
        "action": {
            "end_effector_position": {"original_key": "action", "start": 0, "end": 3},
            "end_effector_rotation": {"original_key": "action", "start": 3, "end": 6},
            "gripper_close": {"original_key": "action", "start": 6, "end": 7},
        },
    }
    with open(os.path.join(output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=4)

    print(f"\nDone! Processed dataset written to: {output_dir}")
    print(f"  Episodes: {total_episodes} ({len(errors)} errors)")
    print(f"  Frames: {total_frames}")
    print(f"  Tasks: {len(tasks_list)}")
    print(f"  Action dim: 7 (ee_pos[3] + ee_rot[3] + gripper[1])")
    print(f"  State dim: 16")
    episodes_with_reasoning = sum(1 for ep in all_episodes if ep["reasoning_text"])
    print(f"  Episodes with reasoning text: {episodes_with_reasoning}/{total_episodes}")


if __name__ == "__main__":
    main()
