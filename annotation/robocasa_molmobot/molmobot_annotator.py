#!/usr/bin/env python3
"""Annotation the MolmoBot full-existing training trajectories.

Default target pass:
  - FrankaPickAndPlaceOmniCamConfig/part1/train
  - FrankaPickAndPlaceNextToOmniCamConfig/part2/train
  - FrankaPickAndPlaceColorOmniCamConfig/part0/train

Output:
  <DATA_ROOT>/captions/<config>/captions_<model>_full_existing_<part>_shard<K>.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


_IN_CONTAINER = os.path.isdir("/workspace")
_BASE = (
    "/workspace"
    if _IN_CONTAINER
    else "/path/to"
)

HOST_PREFIX = "/path/to"
CONTAINER_PREFIX = "/workspace"

DEFAULT_DATA_ROOT = f"{_BASE}/data/molmobot-data"

CAMERA = "droid_shoulder_light_randomization"
REQUIRED_ANNOTATION_TYPES = ("current_action", "environment-simple")

TARGET_SPECS: dict[str, dict[str, int | str]] = {
    "FrankaPickAndPlaceOmniCamConfig": {
        "part": "part1",
        "expected": 221_035,
        "shards": 32,
    },
    "FrankaPickAndPlaceNextToOmniCamConfig": {
        "part": "part2",
        "expected": 37_392,
        "shards": 6,
    },
    "FrankaPickAndPlaceColorOmniCamConfig": {
        "part": "part0",
        "expected": 28_177,
        "shards": 4,
    },
}

CONFIGS = tuple(TARGET_SPECS.keys())
AnnotationKey = tuple[str, str, int]


def rewrite_path(path: str) -> str:
    """Map host paths into the Slurm container mount when needed."""
    if _IN_CONTAINER and path.startswith(HOST_PREFIX):
        return CONTAINER_PREFIX + path[len(HOST_PREFIX):]
    return path


def parse_batch_from_h5_subpath(subpath: str) -> str:
    match = re.search(r"trajectories_(batch_\d+_of_\d+)\.h5$", subpath)
    if not match:
        raise ValueError(f"Could not parse batch from {subpath}")
    return match.group(1)


def parse_traj_idx(traj_key: str) -> int:
    return int(traj_key.split("_", 1)[1])


def house_sort_key(house: str) -> int:
    return int(house.split("_", 1)[1])


def load_valid_trajectory_index(train_dir: Path) -> dict[str, dict[str, dict[str, int]]]:
    index_path = train_dir / "valid_trajectory_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing training index: {index_path}")
    with open(index_path) as f:
        return json.load(f)


def read_h5_metadata(h5_path: Path, traj_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Read task metadata for a batch H5. Missing/bad groups return blanks."""
    import h5py

    metadata: dict[str, dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as f:
        for traj_key in traj_keys:
            if traj_key not in f:
                metadata[traj_key] = {
                    "task_description": "",
                    "task_type": "",
                    "length": 0,
                    "metadata_error": "missing_traj_group",
                }
                continue

            traj = f[traj_key]
            task_desc = ""
            task_type = ""
            try:
                obs_scene_raw = traj["obs_scene"][()]
                if isinstance(obs_scene_raw, bytes):
                    obs_scene_raw = obs_scene_raw.decode("utf-8").rstrip("\x00")
                obs_scene = json.loads(obs_scene_raw)
                task_desc = obs_scene.get("task_description", "") or ""
                task_type = obs_scene.get("task_type", "") or ""
            except Exception as exc:
                metadata[traj_key] = {
                    "task_description": "",
                    "task_type": "",
                    "length": 0,
                    "metadata_error": f"obs_scene_error: {exc}",
                }
                continue

            length = 0
            try:
                length = int(traj["success"].shape[0])
            except Exception:
                pass

            metadata[traj_key] = {
                "task_description": task_desc,
                "task_type": task_type,
                "length": length,
            }
    return metadata


def discover_target_records(
    data_root: str,
    config: str,
    part: str,
    camera: str,
    include_metadata: bool,
    check_video: bool,
) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    """Build the target records from valid_trajectory_index.json."""
    train_dir = Path(data_root) / config / part / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    valid_index = load_valid_trajectory_index(train_dir)
    records: list[dict[str, Any]] = []
    problems: Counter[str] = Counter()
    examples: list[str] = []

    for house in sorted(valid_index, key=house_sort_key):
        files = valid_index[house]
        for h5_subpath, traj_lens in files.items():
            batch = parse_batch_from_h5_subpath(h5_subpath)
            h5_path = train_dir / h5_subpath
            traj_keys = list(traj_lens.keys())
            batch_metadata: dict[str, dict[str, Any]] = {}

            if include_metadata:
                if h5_path.exists():
                    try:
                        batch_metadata = read_h5_metadata(h5_path, traj_keys)
                    except Exception as exc:
                        problems["h5_metadata_error"] += len(traj_keys)
                        if len(examples) < 8:
                            examples.append(f"{h5_subpath}: metadata error: {exc}")
                else:
                    problems["missing_h5"] += len(traj_keys)
                    if len(examples) < 8:
                        examples.append(f"{h5_subpath}: missing h5")

            for traj_key, indexed_len in traj_lens.items():
                episode_index = parse_traj_idx(traj_key)
                video_path = (
                    train_dir
                    / house
                    / f"episode_{episode_index:08d}_{camera}_{batch}.mp4"
                )
                video_exists = video_path.exists() if check_video else None
                if not video_exists:
                    if check_video:
                        problems["missing_video"] += 1
                        if len(examples) < 8:
                            examples.append(f"{house}/{batch}/ep{episode_index}: missing video")

                meta = batch_metadata.get(traj_key, {})
                if meta.get("metadata_error"):
                    problems[str(meta["metadata_error"])] += 1
                    if len(examples) < 8:
                        examples.append(f"{house}/{batch}/{traj_key}: {meta['metadata_error']}")

                records.append({
                    "config": config,
                    "part": part,
                    "house": house,
                    "batch": batch,
                    "episode_index": episode_index,
                    "task_description": meta.get("task_description", ""),
                    "task_type": meta.get("task_type", ""),
                    "length": int(meta.get("length") or indexed_len or 0),
                    "video_path": rewrite_path(str(video_path)),
                    "video_exists": video_exists,
                    "h5_subpath": h5_subpath,
                })

    return records, problems, examples


def hydrate_records(
    data_root: str,
    records: list[dict[str, Any]],
) -> tuple[Counter[str], list[str]]:
    """Fill metadata and video existence for only the records this worker owns."""
    problems: Counter[str] = Counter()
    examples: list[str] = []
    if not records:
        return problems, examples

    config = records[0]["config"]
    part = records[0]["part"]
    train_dir = Path(data_root) / config / part / "train"

    by_h5: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_h5.setdefault(rec["h5_subpath"], []).append(rec)

    for h5_subpath, batch_records in by_h5.items():
        h5_path = train_dir / h5_subpath
        traj_keys = [f"traj_{int(rec['episode_index'])}" for rec in batch_records]
        metadata: dict[str, dict[str, Any]] = {}
        if h5_path.exists():
            try:
                metadata = read_h5_metadata(h5_path, traj_keys)
            except Exception as exc:
                problems["h5_metadata_error"] += len(batch_records)
                if len(examples) < 8:
                    examples.append(f"{h5_subpath}: metadata error: {exc}")
        else:
            problems["missing_h5"] += len(batch_records)
            if len(examples) < 8:
                examples.append(f"{h5_subpath}: missing h5")

        for rec in batch_records:
            traj_key = f"traj_{int(rec['episode_index'])}"
            meta = metadata.get(traj_key, {})
            if meta.get("metadata_error"):
                problems[str(meta["metadata_error"])] += 1
                if len(examples) < 8:
                    examples.append(
                        f"{rec['house']}/{rec['batch']}/{traj_key}: {meta['metadata_error']}"
                    )
            rec["task_description"] = meta.get("task_description", "")
            rec["task_type"] = meta.get("task_type", "")
            rec["length"] = int(meta.get("length") or rec.get("length") or 0)

            video_exists = os.path.exists(rec["video_path"])
            rec["video_exists"] = video_exists
            if not video_exists:
                problems["missing_video"] += 1
                if len(examples) < 8:
                    examples.append(
                        f"{rec['house']}/{rec['batch']}/ep{rec['episode_index']}: missing video"
                    )

    return problems, examples


def annotation_key(entry: dict[str, Any]) -> AnnotationKey:
    return (entry["house"], entry["batch"], int(entry["episode_index"]))


def is_valid_annotation_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and not stripped.startswith("ERROR:")
    return True


def load_existing_annotations(annotations_dir: str) -> tuple[dict[AnnotationKey, dict[str, Any]], Counter[str], list[str]]:
    """Union valid annotations across every JSONL shard in a config directory."""
    merged: dict[AnnotationKey, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    examples: list[str] = []

    for path in sorted(glob.glob(os.path.join(annotations_dir, "*.jsonl"))):
        stats["files"] += 1
        try:
            with open(path) as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    stats["lines"] += 1
                    try:
                        entry = json.loads(line)
                        key = (
                            entry["house"],
                            entry["batch"],
                            int(entry["episode_index"]),
                        )
                    except Exception as exc:
                        stats["bad_lines"] += 1
                        if len(examples) < 8:
                            examples.append(f"{path}:{lineno}: bad line: {exc}")
                        continue

                    dst = merged.setdefault(key, {})
                    for subkey, value in (entry.get("annotations", {}) or {}).items():
                        if not is_valid_annotation_value(value):
                            stats["invalid_annotation_values"] += 1
                            continue
                        if subkey in dst and dst[subkey] != value:
                            stats["conflicting_values"] += 1
                            if len(examples) < 8:
                                examples.append(
                                    f"{path}:{lineno}: conflicting {subkey} for "
                                    f"{key[0]}/{key[1]}/ep{key[2]}"
                                )
                            continue
                        dst[subkey] = value
        except Exception as exc:
            stats["unreadable_files"] += 1
            if len(examples) < 8:
                examples.append(f"{path}: unreadable: {exc}")

    return merged, stats, examples


def missing_required(annotations: dict[str, Any], required_keys: tuple[str, ...]) -> list[str]:
    return [key for key in required_keys if key not in annotations]


def summarize_coverage(
    records: list[dict[str, Any]],
    existing: dict[AnnotationKey, dict[str, Any]],
    required_keys: tuple[str, ...],
) -> dict[str, Any]:
    complete = 0
    present_by_key: Counter[str] = Counter()
    missing_by_key: Counter[str] = Counter()
    missing_examples: list[str] = []

    for rec in records:
        caps = existing.get(annotation_key(rec), {})
        missing = missing_required(caps, required_keys)
        for key in required_keys:
            if key in caps:
                present_by_key[key] += 1
            else:
                missing_by_key[key] += 1
        if not missing:
            complete += 1
        elif len(missing_examples) < 8:
            missing_examples.append(
                f"{rec['config']}/{rec['part']} {rec['house']}/{rec['batch']}/"
                f"ep{rec['episode_index']}: missing {','.join(missing)}"
            )

    total = len(records)
    pct = 100.0 * complete / total if total else 100.0
    return {
        "total": total,
        "complete": complete,
        "pct": pct,
        "required_keys": required_keys,
        "present_by_key": present_by_key,
        "missing_by_key": missing_by_key,
        "missing_examples": missing_examples,
    }


def print_coverage_block(label: str, summary: dict[str, Any], expected: int | None = None):
    total = summary["total"]
    suffix = ""
    if expected is not None:
        suffix = " OK" if total == expected else f" EXPECTED {expected:,}"
    print(
        f"{label}: {summary['complete']:,}/{total:,} complete "
        f"({summary['pct']:.2f}%){suffix}"
    )
    present_by_key = summary["present_by_key"]
    missing_by_key = summary["missing_by_key"]
    for key in summary.get("required_keys", REQUIRED_ANNOTATION_TYPES):
        print(
            f"  {key}: present {present_by_key[key]:,}, "
            f"missing {missing_by_key[key]:,}"
        )
    if summary["missing_examples"]:
        print("  Missing examples:")
        for example in summary["missing_examples"]:
            print(f"    - {example}")


def resolve_part(config: str, requested_part: str | None, parser: argparse.ArgumentParser) -> str:
    expected_part = str(TARGET_SPECS[config]["part"])
    if requested_part is None:
        return expected_part
    if requested_part != expected_part:
        parser.error(
            f"{config} target part is {expected_part}; got {requested_part}. "
            "This script is restricted to the current full-existing target set."
        )
    return requested_part


def ensure_safe_paths(data_root: str, annotations_dir: str, parser: argparse.ArgumentParser):
    normalized_root = str(Path(data_root))
    normalized_out = str(Path(annotations_dir))
    if normalized_root.endswith("/subset_20pct"):
        parser.error("--data_root must be the full molmobot-data root, not subset_20pct")
    if "/subset_20pct/captions" in normalized_out:
        parser.error("Refusing to write into subset_20pct/captions")


def annotation_type_slug(annotation_types: list[str]) -> str:
    raw = "_".join(annotation_types)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def run_coverage_report(args: argparse.Namespace, parser: argparse.ArgumentParser):
    configs = list(CONFIGS) if args.config in (None, "all") else [args.config]
    global_records: list[dict[str, Any]] = []
    global_existing: dict[tuple[str, AnnotationKey], dict[str, Any]] = {}
    required_keys = tuple(
        key.strip() for key in args.coverage_annotation_type.split(",") if key.strip()
    )
    if not required_keys:
        parser.error("--coverage_annotation_type must include at least one annotation subkey")

    print(f"Data root: {args.data_root}")
    print(f"Required: {', '.join(required_keys)}")
    print()

    for config in configs:
        part = resolve_part(config, args.part, parser)
        annotations_dir = args.output_dir or os.path.join(args.data_root, "captions", config)
        ensure_safe_paths(args.data_root, annotations_dir, parser)

        records, problems, problem_examples = discover_target_records(
            args.data_root,
            config,
            part,
            args.camera,
            include_metadata=False,
            check_video=False,
        )
        existing, stats, load_examples = load_existing_annotations(annotations_dir)
        summary = summarize_coverage(records, existing, required_keys)

        print_coverage_block(
            f"{config}/{part}",
            summary,
            expected=int(TARGET_SPECS[config]["expected"]),
        )
        print(f"  annotations_dir: {annotations_dir}")
        print(f"  existing files: {stats['files']:,}, jsonl lines: {stats['lines']:,}")
        if stats["conflicting_values"]:
            print(f"  conflicting annotation values ignored: {stats['conflicting_values']:,}")
        if problems:
            print(f"  target discovery warnings: {dict(problems)}")
        for example in (problem_examples + load_examples)[:8]:
            print(f"    - {example}")
        print()

        global_records.extend(records)
        for key, annotations in existing.items():
            global_existing[(config, key)] = annotations

    if len(configs) > 1:
        complete = 0
        present_by_key: Counter[str] = Counter()
        missing_by_key: Counter[str] = Counter()
        missing_examples: list[str] = []
        for rec in global_records:
            caps = global_existing.get((rec["config"], annotation_key(rec)), {})
            missing = missing_required(caps, required_keys)
            for key in required_keys:
                if key in caps:
                    present_by_key[key] += 1
                else:
                    missing_by_key[key] += 1
            if not missing:
                complete += 1
            elif len(missing_examples) < 8:
                missing_examples.append(
                    f"{rec['config']}/{rec['part']} {rec['house']}/{rec['batch']}/"
                    f"ep{rec['episode_index']}: missing {','.join(missing)}"
                )

        total = len(global_records)
        print_coverage_block(
            "GLOBAL",
            {
                "total": total,
                "complete": complete,
                "pct": 100.0 * complete / total if total else 100.0,
                "required_keys": required_keys,
                "present_by_key": present_by_key,
                "missing_by_key": missing_by_key,
                "missing_examples": missing_examples,
            },
            expected=sum(int(spec["expected"]) for spec in TARGET_SPECS.values()),
        )


def build_client_and_prompt_helpers(args: argparse.Namespace):
    import openai
    from dense_annotator import SplitContext, build_messages, call_api

    client = openai.OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    model = args.vllm_model
    video_content_type = "video_url"

    return client, model, video_content_type, SplitContext, build_messages, call_api


def main():
    parser = argparse.ArgumentParser(
        description="Annotation MolmoBot full-existing training trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", choices=("all",) + CONFIGS, default=None)
    parser.add_argument("--part", default=None, help="Must match this config's target part")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <data_root>/captions/<config>)")
    parser.add_argument("--camera", default=CAMERA)

    parser.add_argument("--annotation_type", default=",".join(REQUIRED_ANNOTATION_TYPES),
                        help="Comma-separated annotation subkeys to generate")
    parser.add_argument("--prompts_json",
                        default=os.path.join(os.path.dirname(__file__), "prompts.json"))
    parser.add_argument("--rerun", action="store_true", help="Append regenerated annotations")
    parser.add_argument("--coverage_only", action="store_true",
                        help="Only report target coverage and exit")
    parser.add_argument("--coverage_annotation_type",
                        default=",".join(REQUIRED_ANNOTATION_TYPES),
                        help="Comma-separated annotation subkeys required by --coverage_only")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--n", type=int, default=None,
                        help="Max appended records per worker (for testing)")

    parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--vllm_model", default="Qwen3-VL-30B-A3B-Instruct")

    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)

    args = parser.parse_args()

    if args.coverage_only:
        run_coverage_report(args, parser)
        return

    if args.config in (None, "all"):
        parser.error("--config is required for annotation generation")
    if args.worker_id < 0 or args.worker_id >= args.n_workers:
        parser.error("--worker_id must be in [0, n_workers)")

    part = resolve_part(args.config, args.part, parser)
    annotations_dir = args.output_dir or os.path.join(args.data_root, "captions", args.config)
    ensure_safe_paths(args.data_root, annotations_dir, parser)
    os.makedirs(annotations_dir, exist_ok=True)

    prompts_config = json.load(open(args.prompts_json))
    prompts = prompts_config["prompts"]
    annotation_types = [ct.strip() for ct in args.annotation_type.split(",") if ct.strip()]
    for ct in annotation_types:
        if ct not in prompts:
            parser.error(f"Unknown annotation type: {ct}")

    model = args.vllm_model

    records, discovery_problems, discovery_examples = discover_target_records(
        args.data_root,
        args.config,
        part,
        args.camera,
        include_metadata=False,
        check_video=False,
    )
    expected = int(TARGET_SPECS[args.config]["expected"])
    if len(records) != expected:
        print(
            f"[warn] target record count mismatch for {args.config}/{part}: "
            f"discovered {len(records):,}, expected {expected:,}",
            file=sys.stderr,
        )

    model_slug = model.split("/")[-1]
    out_file = os.path.join(
        annotations_dir,
        f"captions_{model_slug}_full_existing_{part}_"
        f"{annotation_type_slug(annotation_types)}_shard{args.worker_id}.jsonl",
    )

    my_records = [rec for i, rec in enumerate(records) if i % args.n_workers == args.worker_id]
    existing, existing_stats, existing_examples = load_existing_annotations(annotations_dir)
    config_summary = summarize_coverage(records, existing, REQUIRED_ANNOTATION_TYPES)
    shard_summary = summarize_coverage(my_records, existing, REQUIRED_ANNOTATION_TYPES)

    pending_records: list[dict[str, Any]] = []
    for rec in my_records:
        existing_caps = existing.get(annotation_key(rec), {})
        to_run = [
            ct for ct in annotation_types
            if args.rerun or ct not in existing_caps
        ]
        if to_run:
            rec["_to_run"] = to_run
            pending_records.append(rec)

    count_skipped_done = len(my_records) - len(pending_records)
    process_records = pending_records
    count_deferred_limit = 0
    if args.n is not None:
        process_records = pending_records[:args.n]
        count_deferred_limit = max(0, len(pending_records) - len(process_records))

    shard_problems, shard_examples = hydrate_records(args.data_root, process_records)
    discovery_problems.update(shard_problems)
    discovery_examples.extend(shard_examples)

    print(f"Config       : {args.config}")
    print(f"Part         : {part}/train")
    print(f"Data root    : {args.data_root}")
    print(f"Output       : {out_file}")
    print(f"Model        : {model}")
    print(f"Camera       : {args.camera}")
    print(f"Annotation types: {annotation_types}")
    print(f"Shard        : {args.worker_id}/{args.n_workers} ({len(my_records):,} records)")
    print(f"Existing     : {existing_stats['files']:,} files, {existing_stats['lines']:,} lines")
    print(f"Pending      : {len(pending_records):,} records needing {annotation_types}")
    print_coverage_block("Config coverage before", config_summary, expected=expected)
    print_coverage_block("Shard coverage before", shard_summary)
    if discovery_problems:
        print(f"Discovery warnings: {dict(discovery_problems)}")
    for example in (discovery_examples + existing_examples)[:8]:
        print(f"  [example] {example}")
    if args.n is not None:
        print(f"Append limit : {args.n}")
    print()

    count_appended = 0
    count_missing_video = 0
    count_generated_subkeys = 0

    if process_records:
        client, model, video_content_type, SplitContext, build_messages, call_api = (
            build_client_and_prompt_helpers(args)
        )
    else:
        client = video_content_type = SplitContext = build_messages = call_api = None

    if process_records:
        fout = open(out_file, "a", buffering=1)
    else:
        fout = None

    try:
        for idx, rec in enumerate(process_records):
            key = annotation_key(rec)
            existing_caps = existing.get(key, {})
            to_run = rec["_to_run"]

            if not rec["video_exists"] or not os.path.exists(rec["video_path"]):
                count_missing_video += 1
                if count_missing_video <= 8:
                    print(
                        f"  [missing-video] {rec['house']}/{rec['batch']}/"
                        f"ep{rec['episode_index']} -> {rec['video_path']}"
                    )
                continue

            print(
                f"  [{rec['house']}/{rec['batch']}/ep{rec['episode_index']}] "
                f"{rec['task_description'][:60]} -> {to_run}"
            )

            split_ctx = SplitContext(
                original_task_name=rec["task_type"],
                original_descs=[rec["task_description"]] if rec["task_description"] else [],
                primitive_ctx=None,
            )

            entry = {
                "config": rec["config"],
                "part": rec["part"],
                "house": rec["house"],
                "batch": rec["batch"],
                "episode_index": rec["episode_index"],
                "task_description": rec["task_description"],
                "task_type": rec["task_type"],
                "annotations": {},
            }

            for ct in to_run:
                prompt_cfg = prompts[ct]
                try:
                    assert SplitContext is not None
                    assert build_messages is not None
                    assert call_api is not None
                    messages = build_messages(
                        rec["video_path"],
                        is_composite=False,
                        task_desc=rec["task_description"],
                        split_ctx=split_ctx,
                        prompt_text=prompt_cfg["prompt"],
                        constraint=prompt_cfg.get("constraint", ""),
                        inject_primitive_ctx=False,
                        video_content_type=video_content_type,
                        no_context=False,
                        json_key="annotation",
                    )
                    assert client is not None
                    raw = call_api(client, model, messages, args.max_tokens)
                    try:
                        annotation = json.loads(raw)["annotation"]
                    except (json.JSONDecodeError, KeyError):
                        annotation = raw
                    entry["annotations"][ct] = annotation
                    if is_valid_annotation_value(annotation):
                        existing.setdefault(key, {})[ct] = annotation
                    count_generated_subkeys += 1
                    print(f"    [{ct}] {str(annotation)[:100]}")
                except Exception as exc:
                    print(f"    [{ct}] ERROR: {exc}")
                    entry["annotations"][ct] = f"ERROR: {exc}"

            assert fout is not None
            fout.write(json.dumps(entry) + "\n")
            count_appended += 1

            if count_appended % 50 == 0:
                print(
                    f"  ... appended {count_appended}, generated {count_generated_subkeys} subkeys, "
                    f"skipped {count_skipped_done}, missing-video {count_missing_video}, "
                    f"idx={idx + 1}/{len(process_records)}"
                )
    finally:
        if fout is not None:
            fout.close()

    shard_after = summarize_coverage(my_records, existing, REQUIRED_ANNOTATION_TYPES)
    print()
    print(
        f"Worker {args.worker_id} done. Appended records: {count_appended}, "
        f"generated subkeys: {count_generated_subkeys}, already complete: {count_skipped_done}, "
        f"missing video: {count_missing_video}, deferred by --n: {count_deferred_limit}"
    )
    print_coverage_block("Shard coverage after local appends", shard_after)


if __name__ == "__main__":
    main()
