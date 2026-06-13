#!/usr/bin/env python3
"""Prepare training data for the Qwen3.5-2B VLM instructor.

Reads the RoboCasa training episodes and extracts (images, task_description,
caption) tuples using the augmented per-task caption routing table.

For each episode, extracts frame-0 images (agentview_left, agentview_right,
eye_in_hand) from the packed npz files and saves them as JPEG.

Output layout:
    <output_dir>/
        train.jsonl          # {image_left, image_right, image_wrist, task_description, caption, ...}
        val.jsonl
        images/
            ep000000_left.jpg
            ep000000_right.jpg
            ep000000_wrist.jpg
            ...
        routing.json         # copy of routing table for provenance

Usage:
    python scripts/prepare_instructor_data.py
    python scripts/prepare_instructor_data.py --val-ratio 0.05 --output-dir data/instructor
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Defaults point at the released RoboCasa 365 dataset (data/robocasa365_v1; see
# data/README.md — run openpi/scripts/preprocess_atomic_split.py to materialize packed/).
DEFAULT_EPISODES_JSONL = "data/robocasa365_v1/meta/episodes.jsonl"
DEFAULT_PACKED_DIR = "data/robocasa365_v1/packed"
DEFAULT_ROUTING_JSON = "runs/robocasa/task_caption_routing_augmented.json"
DEFAULT_OUTPUT_DIR = "data/instructor_training_robocasa"

# Suffixes to strip from source_task to get the canonical eval task name.
_SPLIT_SUFFIXES = re.compile(r"-(DoubleDoor|SingleDoor|split)$")


def canonical_task(source_task: str) -> str:
    """Map source_task (e.g., 'CloseFridge-DoubleDoor') to eval task ('CloseFridge')."""
    return _SPLIT_SUFFIXES.sub("", source_task)


def sample_caption_type(weights: dict[str, float], rng: random.Random) -> str:
    """Weighted random sample from {caption_type: probability}."""
    types = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(types, weights=probs, k=1)[0]


def save_frame_images(ep_idx: int, packed_dir: str, images_dir: Path) -> dict[str, str] | None:
    """Extract frame-0 images from packed npz and save as JPEG.

    Returns dict of {camera: relative_path} or None if npz not found.
    """
    npz_path = Path(packed_dir) / f"episode_{ep_idx:06d}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    paths = {}
    for cam_key, suffix in [("agentview_left", "left"), ("agentview_right", "right"), ("eye_in_hand", "wrist")]:
        if cam_key not in data:
            return None
        frame = data[cam_key][0]  # frame 0, shape (256, 256, 3) uint8 RGB
        fname = f"ep{ep_idx:06d}_{suffix}.jpg"
        Image.fromarray(frame).save(str(images_dir / fname), quality=95)
        paths[f"image_{suffix}"] = f"images/{fname}"
    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare instructor training data")
    parser.add_argument("--episodes-jsonl", default=DEFAULT_EPISODES_JSONL)
    parser.add_argument("--packed-dir", default=DEFAULT_PACKED_DIR)
    parser.add_argument("--routing-json", default=DEFAULT_ROUTING_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load augmented routing table.
    routing = json.loads(Path(args.routing_json).read_text())
    print(f"Loaded routing for {len(routing)} tasks from {args.routing_json}")

    # Load episodes.
    episodes = []
    with open(args.episodes_jsonl) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes from {args.episodes_jsonl}")

    # Process episodes.
    examples = []
    stats = Counter()
    skipped_no_routing = 0
    skipped_no_caption = 0
    skipped_no_images = 0

    for i, ep in enumerate(episodes):
        source_task = ep.get("source_task", "")
        if not source_task:
            stats["no_source_task"] += 1
            continue

        canon = canonical_task(source_task)
        if canon not in routing:
            skipped_no_routing += 1
            continue

        weights = routing[canon]
        caption_type = sample_caption_type(weights, rng)

        # Get task description.
        tasks = ep.get("tasks", [])
        task_desc = tasks[0] if tasks else source_task

        # Get caption text.
        captions = ep.get("episode_captions", {})
        if caption_type == "None":
            caption_text = task_desc
        else:
            caption_text = captions.get(caption_type, "")

        if not caption_text:
            skipped_no_caption += 1
            continue

        # Extract frame-0 images.
        ep_idx = ep["episode_index"]
        image_paths = save_frame_images(ep_idx, args.packed_dir, images_dir)
        if image_paths is None:
            skipped_no_images += 1
            continue

        example = {
            **image_paths,
            "task_description": task_desc,
            "caption": caption_text,
            "caption_type": caption_type,
            "source_task": source_task,
            "canonical_task": canon,
            "episode_index": ep_idx,
        }
        examples.append(example)
        stats[f"type:{caption_type}"] += 1
        stats[f"task:{canon}"] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(episodes)} episodes, {len(examples)} examples so far")

    print(f"\nTotal examples: {len(examples)}")
    print(f"Skipped (no routing): {skipped_no_routing}")
    print(f"Skipped (no caption): {skipped_no_caption}")
    print(f"Skipped (no images):  {skipped_no_images}")
    print(f"\nCaption type distribution:")
    for k, v in sorted(stats.items()):
        if k.startswith("type:"):
            print(f"  {k[5:]:30s}: {v:5d} ({v / len(examples) * 100:.1f}%)")

    # Train/val split.
    rng.shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Save.
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    (out_dir / "routing.json").write_text(json.dumps(routing, indent=2))

    print(f"\nSaved {len(train_examples)} train, {len(val_examples)} val to {out_dir}")

    # Print sample.
    print(f"\n--- Sample training examples ---")
    for ex in train_examples[:3]:
        print(f"  [{ex['canonical_task']}] type={ex['caption_type']}")
        print(f"    task: {ex['task_description']}")
        print(f"    images: {ex['image_left']}, {ex['image_right']}, {ex['image_wrist']}")
        print(f"    caption: {ex['caption'][:100]}...")
        print()


if __name__ == "__main__":
    main()
