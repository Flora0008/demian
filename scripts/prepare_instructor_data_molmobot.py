#!/usr/bin/env python3
"""Prepare training data for the MolmoBot VLM instructor.

Iterates training episodes across 4 configs (Pick, PnP, NextTo, Color), merges
per-episode GT captions from backfill JSONL shards, samples one caption type
per episode from the routing distribution, extracts the first exo + wrist frame
from the MP4 videos, and writes a train/val JSONL.

Output row schema:
  {
    "image_exo": "images/<stem>_exo.jpg",
    "image_wrist": "images/<stem>_wrist.jpg",
    "task_description": "...",
    "caption": "..." | "",          # "" when caption_type == "none"
    "caption_type": "...",
    "source_config": "FrankaPickAndPlaceOmniCamConfig",
    "house": "house_10025",
    "batch": "batch_13_of_20",
    "episode_index": 0,
    "benchmark": "pnp_std",          # routing key used
    "reward_weight": 0.65             # reward_table[benchmark][caption_type]
  }

Usage:
    python3 scripts/prepare_instructor_data.py \\
        --routing-json runs/molmobot/reward_table_mix5/routing_table.json \\
        --reward-json  runs/molmobot/reward_table_mix5/reward_table.json \\
        --output-dir   data/instructor_training_mix5 \\
        --n 100                        # dry run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2


DEFAULT_DATA_ROOT = "/path/to/data/molmobot-data/subset_20pct"

# One representative benchmark (or a list, for averaged routing) per training config.
# For Pick — training data doesn't encode scene category (Std/Hard/OOD), so
# average the three Pick benchmark routings to get a balanced distribution.
CONFIG_TO_BENCHMARKS = {
    "FrankaPickOmniCamConfig": ["pick_std", "pick_hard", "pick_ood"],
    "FrankaPickAndPlaceOmniCamConfig": ["pnp_std"],
    "FrankaPickAndPlaceNextToOmniCamConfig": ["pnp_nextto"],
    "FrankaPickAndPlaceColorOmniCamConfig": ["pnp_color"],
}

# task_type field → primary benchmark (used for reward_weight lookup)
TASK_TYPE_TO_PRIMARY_BENCH = {
    "pick": "pick_hard",  # use Hard as representative Pick reward
    "pick_and_place": "pnp_std",
    "pick_and_place_next_to": "pnp_nextto",
    "pick_and_place_color": "pnp_color",
}


def load_routing(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text())


def load_reward(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text())


def average_routing(routings: list[dict[str, float]]) -> dict[str, float]:
    """Average multiple caption_type→prob distributions, re-normalize."""
    if len(routings) == 1:
        return dict(routings[0])
    keys = set()
    for r in routings:
        keys.update(r.keys())
    avg = {k: sum(r.get(k, 0.0) for r in routings) / len(routings) for k in keys}
    total = sum(avg.values())
    if total > 0:
        avg = {k: v / total for k, v in avg.items()}
    return avg


def build_caption_lookup(data_root: Path, config: str) -> dict[tuple, dict]:
    """Merge all caption JSONL shards for one config.

    Returns dict mapping (house, batch, episode_index) → entry dict where
    entry.captions is the union of caption keys seen across shards.
    """
    lookup: dict[tuple, dict] = {}
    shards = sorted((data_root / "captions" / config).glob("*.jsonl"))
    for sh in shards:
        with open(sh) as f:
            for line in f:
                e = json.loads(line)
                key = (e["house"], e["batch"], e["episode_index"])
                if key not in lookup:
                    # Shallow-copy the entry; we'll merge captions dict in place.
                    lookup[key] = {
                        "house": e["house"],
                        "batch": e["batch"],
                        "episode_index": e["episode_index"],
                        "task_description": e.get("task_description", ""),
                        "task_type": e.get("task_type", ""),
                        "annotations": dict(e.get("annotations", {})),
                    }
                else:
                    # Merge caption keys; first non-empty wins per key
                    merged = lookup[key]["annotations"]
                    for k, v in e.get("annotations", {}).items():
                        if v and k not in merged:
                            merged[k] = v
    return lookup


def sample_caption_type(weights: dict[str, float], rng: random.Random) -> str:
    types = list(weights.keys())
    probs = [weights[t] for t in types]
    return rng.choices(types, weights=probs, k=1)[0]


def extract_first_frame(video_path: Path, out_path: Path, quality: int = 95) -> bool:
    """Write first frame of video to out_path as JPEG. Returns True on success."""
    if out_path.exists():
        return True
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    ok = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bool(ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--routing-json", required=True)
    p.add_argument("--reward-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=None,
                   help="Max total episodes (for dry run; default = all).")
    p.add_argument("--max-per-config", type=int, default=None,
                   help="Max episodes per config (uniformly subsampled). "
                        "Default = all.")
    p.add_argument("--configs", default=",".join(CONFIG_TO_BENCHMARKS.keys()),
                   help="Comma-separated configs to include.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    routing = load_routing(Path(args.routing_json))
    reward = load_reward(Path(args.reward_json))
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    # Pre-compute config-level routing (averaged across representative benchmarks)
    config_routing: dict[str, dict[str, float]] = {}
    for cfg in configs:
        if cfg not in CONFIG_TO_BENCHMARKS:
            raise ValueError(f"Unknown config {cfg}")
        benches = CONFIG_TO_BENCHMARKS[cfg]
        rs = [routing[b] for b in benches if b in routing]
        if not rs:
            raise ValueError(f"Config {cfg} has no benchmarks in routing: {benches}")
        config_routing[cfg] = average_routing(rs)
        print(f"[{cfg}] routing (avg over {benches}):")
        for k, v in sorted(config_routing[cfg].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<20} {v:.3f}")

    examples = []
    stats = Counter()
    # Per-type caption-missing stats (e.g., we sample agent_arm_state but the
    # episode's caption merge doesn't have it → skip this episode)
    missing_caption_stats = Counter()
    frame_failures = 0

    t0 = time.time()
    total_considered = 0

    for cfg in configs:
        print(f"\n=== {cfg} ===")
        lookup = build_caption_lookup(data_root, cfg)
        print(f"  merged caption entries: {len(lookup)}")

        weights = config_routing[cfg]

        # Deterministic ordering, then optional uniform sub-sample per config
        keys_ordered = sorted(lookup.keys())
        if args.max_per_config is not None and len(keys_ordered) > args.max_per_config:
            # Subsample deterministically — use a config-scoped rng so the
            # global caption-type rng isn't perturbed.
            sub_rng = random.Random(args.seed + hash(cfg) % 1_000_000)
            keys_ordered = sub_rng.sample(keys_ordered, args.max_per_config)
            keys_ordered.sort()
            print(f"  subsampled to {len(keys_ordered)} episodes (max-per-config)")

        examples_before = len(examples)
        for key in keys_ordered:
            entry = lookup[key]
            if args.n is not None and total_considered >= args.n:
                break
            total_considered += 1

            house = entry["house"]
            batch = entry["batch"]
            ep_idx = entry["episode_index"]
            task_desc = entry["task_description"]
            task_type = entry["task_type"]

            # Sample caption type from the routing distribution
            caption_type = sample_caption_type(weights, rng)

            if caption_type == "none":
                caption_text = ""
            else:
                caption_text = entry["annotations"].get(caption_type, "")
                if not caption_text:
                    # This episode's merge doesn't have the sampled type — skip
                    missing_caption_stats[caption_type] += 1
                    continue

            # Reward weight: use primary benchmark for task_type, or first cfg bench
            primary_bench = TASK_TYPE_TO_PRIMARY_BENCH.get(
                task_type, CONFIG_TO_BENCHMARKS[cfg][0],
            )
            reward_weight = reward.get(primary_bench, {}).get(caption_type, 0.5)

            # Extract first frames
            stem = f"{cfg}__{house}__{batch}__ep{ep_idx:06d}"
            exo_src = data_root / cfg / "part0" / "train" / house / \
                f"episode_{ep_idx:08d}_droid_shoulder_light_randomization_{batch}.mp4"
            wrist_src = data_root / cfg / "part0" / "train" / house / \
                f"episode_{ep_idx:08d}_wrist_camera_zed_mini_{batch}.mp4"
            exo_dst = images_dir / f"{stem}_exo.jpg"
            wrist_dst = images_dir / f"{stem}_wrist.jpg"

            if not exo_src.exists() or not wrist_src.exists():
                frame_failures += 1
                continue
            if not extract_first_frame(exo_src, exo_dst) or \
               not extract_first_frame(wrist_src, wrist_dst):
                frame_failures += 1
                continue

            examples.append({
                "image_exo": f"images/{exo_dst.name}",
                "image_wrist": f"images/{wrist_dst.name}",
                "task_description": task_desc,
                "caption": caption_text,
                "caption_type": caption_type,
                "source_config": cfg,
                "house": house,
                "batch": batch,
                "episode_index": ep_idx,
                "benchmark": primary_bench,
                "reward_weight": round(reward_weight, 4),
            })
            stats[f"type:{caption_type}"] += 1
            stats[f"bench:{primary_bench}"] += 1
            stats[f"cfg:{cfg}"] += 1

            if len(examples) % 1000 == 0:
                dt = time.time() - t0
                rate = len(examples) / dt if dt > 0 else 0
                print(f"  [{len(examples):6d} examples, {rate:.1f} ex/s, "
                      f"{total_considered:6d} considered]")

        if args.n is not None and total_considered >= args.n:
            break

    # Report
    dt = time.time() - t0
    print(f"\nTotal: {len(examples)} examples from {total_considered} considered "
          f"in {dt:.1f}s ({len(examples)/max(dt,1):.1f} ex/s)")
    print(f"Frame-extraction failures: {frame_failures}")
    if missing_caption_stats:
        print(f"Skipped due to missing caption for sampled type:")
        for k, v in missing_caption_stats.most_common():
            print(f"  {k}: {v}")

    print("\nCaption-type distribution:")
    type_counts = {k[5:]: v for k, v in stats.items() if k.startswith("type:")}
    for t, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        type_rewards = [ex["reward_weight"] for ex in examples if ex["caption_type"] == t]
        avg_r = sum(type_rewards) / len(type_rewards) if type_rewards else 0
        print(f"  {t:<22} {n:6d}  (avg reward_weight={avg_r:.3f})")

    print("\nBenchmark (reward key) distribution:")
    for k, v in sorted(stats.items()):
        if k.startswith("bench:"):
            print(f"  {k[6:]:<14} {v:6d}")

    print("\nConfig distribution:")
    for k, v in sorted(stats.items()):
        if k.startswith("cfg:"):
            print(f"  {k[4:]:<40} {v:6d}")

    # Shuffle + split + write
    rng.shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(out_dir / "val.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    (out_dir / "routing_table.json").write_text(json.dumps(routing, indent=2))
    (out_dir / "reward_table.json").write_text(json.dumps(reward, indent=2))

    print(f"\nSaved {len(train_examples)} train, {len(val_examples)} val → {out_dir}")
    print(f"Images → {images_dir}")


if __name__ == "__main__":
    main()
