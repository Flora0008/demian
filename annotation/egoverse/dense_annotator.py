#!/usr/bin/env python3
"""Dense annotation for EgoVerse human egocentric video data.

Annotates each annotation segment of an episode by sampling frames from the
zarr store and sending them as images to a vLLM (OpenAI-compatible) server.

Usage:
    # Annotate one episode
    python dense_annotator.py --vllm_base_url http://localhost:8000/v1 --vllm_model MODEL --episode 69a6625fb29228ae08b91423

    # Process all episodes (with frame sampling options)
    python dense_annotator.py --vllm_base_url http://localhost:8000/v1 --vllm_model MODEL --max_frames 12 --sample_strategy uniform

    # Specific annotation types
    python dense_annotator.py --vllm_base_url http://localhost:8000/v1 --vllm_model MODEL --annotation_type current_action,reasoning

    # Dry run (load data, print what would be sent, skip API calls)
    python dense_annotator.py --dry_run --episode 69a6625fb29228ae08b91423
"""

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import openai
import zarr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPATIAL_ANNOTATION_TYPES = {"spatial_start", "spatial_end", "spatial_motion"}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BASE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else "/path/to"
)
DEFAULT_DATASET_DIR = os.path.join(_BASE, "egoverse_dataset")
DEFAULT_PROMPTS_JSON = os.path.join(os.path.dirname(__file__), "prompts.json")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode(episode_dir: str) -> dict:
    """Load zarr episode: metadata, annotations, available cameras."""
    store = zarr.open(episode_dir, mode="r")

    with open(os.path.join(episode_dir, "zarr.json")) as f:
        meta = json.load(f)["attributes"]

    raw_annotations = store["annotations"][:]
    annotations = []
    for a in raw_annotations:
        text = a if isinstance(a, str) else a.decode("utf-8")
        annotations.append(json.loads(text))

    cameras = sorted(k for k in store.keys() if k.startswith("images."))

    return {
        "meta": meta,
        "store": store,
        "annotations": annotations,
        "cameras": cameras,
        "total_frames": meta["total_frames"],
        "episode_dir": episode_dir,
    }


def decode_frame(store, camera: str, idx: int) -> np.ndarray:
    """Decode a single JPEG frame from zarr into BGR numpy array."""
    raw = store[camera][idx]
    while isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    jpeg_bytes = bytes(raw)
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def encode_frame_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frame_indices(start: int, end: int, max_frames: int,
                         strategy: str = "uniform") -> list:
    """Sample frame indices from [start, end).

    Strategies:
        uniform:  evenly spaced across the segment
        first_last: first frame, last frame, and uniform in between
        stride:   every Nth frame (N = ceil(segment_len / max_frames))
    """
    segment_len = end - start
    if segment_len <= 0:
        return []
    if segment_len <= max_frames:
        return list(range(start, end))

    if strategy == "uniform":
        indices = np.linspace(start, end - 1, max_frames, dtype=int).tolist()
    elif strategy == "first_last":
        if max_frames <= 2:
            indices = [start, end - 1][:max_frames]
        else:
            middle = np.linspace(start, end - 1, max_frames, dtype=int).tolist()
            indices = [start] + middle[1:-1] + [end - 1]
    elif strategy == "stride":
        step = max(1, segment_len // max_frames)
        indices = list(range(start, end, step))[:max_frames]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Spatial fact extraction (Strategy SG)
# ---------------------------------------------------------------------------

def _grasp_aperture(keypoints_flat: np.ndarray) -> float:
    """Thumb-tip (joint 4) to index-tip (joint 8) distance in cm."""
    kp = keypoints_flat.reshape(21, 3)
    return float(np.linalg.norm(kp[4] - kp[8]) * 100)


def compute_spatial_facts(store, start_idx: int, end_idx: int) -> str:
    """Compute programmatic spatial facts from proprioceptive data.

    Returns a human-readable text block that can be injected into the VLM
    prompt alongside the sampled frames.
    """
    r_ee = store["right.obs_ee_pose"][start_idx:end_idx]   # (T, 7)
    l_ee = store["left.obs_ee_pose"][start_idx:end_idx]
    r_kp = store["right.obs_keypoints"][start_idx:end_idx]  # (T, 63)
    l_kp = store["left.obs_keypoints"][start_idx:end_idx]
    head = store["obs_head_pose"][start_idx:end_idx]         # (T, 7)

    first, last = 0, len(r_ee) - 1

    # Positions in cm
    r_pos_s, r_pos_e = r_ee[first, :3] * 100, r_ee[last, :3] * 100
    l_pos_s, l_pos_e = l_ee[first, :3] * 100, l_ee[last, :3] * 100
    head_s, head_e = head[first, :3] * 100, head[last, :3] * 100

    # Displacements
    r_disp = r_pos_e - r_pos_s
    l_disp = l_pos_e - l_pos_s
    r_disp_cm = float(np.linalg.norm(r_disp))
    l_disp_cm = float(np.linalg.norm(l_disp))

    # Inter-hand distance
    ih_s = float(np.linalg.norm(r_ee[first, :3] - l_ee[first, :3]) * 100)
    ih_e = float(np.linalg.norm(r_ee[last, :3] - l_ee[last, :3]) * 100)

    # Grasp apertures
    r_ga_s = _grasp_aperture(r_kp[first])
    r_ga_e = _grasp_aperture(r_kp[last])
    l_ga_s = _grasp_aperture(l_kp[first])
    l_ga_e = _grasp_aperture(l_kp[last])

    # Convergence / divergence
    if ih_e < ih_s - 1.0:
        ih_trend = "converging"
    elif ih_e > ih_s + 1.0:
        ih_trend = "diverging"
    else:
        ih_trend = "stable"

    def _grasp_trend(ga_s, ga_e):
        if ga_e < ga_s - 0.3:
            return "closing"
        elif ga_e > ga_s + 0.3:
            return "opening"
        return "stable"

    def _fmt_pos(p):
        return f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"

    def _fmt_disp(d):
        return f"({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})"

    n_frames = end_idx - start_idx
    lines = [
        "=== PROPRIOCEPTIVE SPATIAL DATA ===",
        "",
        "--- Start of Segment (first frame) ---",
        f"Right hand position: {_fmt_pos(r_pos_s)} cm",
        f"Left hand position:  {_fmt_pos(l_pos_s)} cm",
        f"Head position:       {_fmt_pos(head_s)} cm",
        f"Inter-hand distance: {ih_s:.1f} cm",
        f"Right grasp aperture (thumb-index): {r_ga_s:.1f} cm",
        f"Left grasp aperture (thumb-index):  {l_ga_s:.1f} cm",
        "",
        "--- End of Segment (last frame) ---",
        f"Right hand position: {_fmt_pos(r_pos_e)} cm",
        f"Left hand position:  {_fmt_pos(l_pos_e)} cm",
        f"Head position:       {_fmt_pos(head_e)} cm",
        f"Inter-hand distance: {ih_e:.1f} cm",
        f"Right grasp aperture (thumb-index): {r_ga_e:.1f} cm",
        f"Left grasp aperture (thumb-index):  {l_ga_e:.1f} cm",
        "",
        "--- Motion Over Segment ({} frames) ---".format(n_frames),
        f"Right hand displacement: {r_disp_cm:.1f} cm, vector {_fmt_disp(r_disp)} cm",
        f"Left hand displacement:  {l_disp_cm:.1f} cm, vector {_fmt_disp(l_disp)} cm",
        f"Inter-hand distance: {ih_s:.1f} → {ih_e:.1f} cm ({ih_trend})",
        f"Right grasp: {r_ga_s:.1f} → {r_ga_e:.1f} cm ({_grasp_trend(r_ga_s, r_ga_e)})",
        f"Left grasp:  {l_ga_s:.1f} → {l_ga_e:.1f} cm ({_grasp_trend(l_ga_s, l_ga_e)})",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotations I/O
# ---------------------------------------------------------------------------

def load_existing_annotations(path: str) -> dict:
    """Load annotations keyed by segment_index."""
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                result[entry["segment_index"]] = entry
    return result


def save_annotations(path: str, annotations: dict):
    """Save annotations sorted by segment_index."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx in sorted(annotations.keys()):
            f.write(json.dumps(annotations[idx]) + "\n")


# ---------------------------------------------------------------------------
# VLM message building
# ---------------------------------------------------------------------------

def build_messages(
    frames: list,
    task_description: str,
    scene_desc: str,
    objects: list,
    segment_annotation: str,
    segment_index: int,
    total_segments: int,
    prev_annotation: str,
    next_annotation: str,
    prompt_text: str,
    constraint: str = "",
    json_key: str = "annotation",
    spatial_context: str = "",
) -> list:
    """Build system + user messages with sampled frames for one VLM call."""
    n_frames = len(frames)

    system_parts = [
        f"You are an expert analyst of human egocentric manipulation videos. "
        f"You will be shown {n_frames} frames sampled from a video segment of a person performing a task, "
        f"captured from a head-mounted camera. "
        f'Respond with a JSON object containing a single key "{json_key}" whose value is the requested description. '
        f'Example: {{"{json_key}": "The person reaches for the bottle with their right hand."}} '
        "Output valid JSON only — no markdown, no extra keys, no text outside the JSON object."
    ]
    if constraint:
        system_parts.append(f'The "{json_key}" value must be {constraint}.')

    # Task context
    task_context = f"TASK: {task_description}\n"
    if scene_desc:
        task_context += f"SCENE: {scene_desc}\n"
    if objects:
        task_context += f"OBJECTS: {', '.join(objects)}\n"

    task_context += (
        f"\nCURRENT SEGMENT: {segment_index + 1} of {total_segments}\n"
        f"CURRENT ACTION: {segment_annotation}\n"
    )
    if prev_annotation:
        task_context += f"PREVIOUS ACTION: {prev_annotation}\n"
    if next_annotation:
        task_context += f"NEXT ACTION: {next_annotation}\n"

    if spatial_context:
        task_context += f"\n{spatial_context}\n"

    task_context += f"\nINSTRUCTION: {prompt_text}"

    # Build image content blocks
    user_content = []
    for frame in frames:
        data_url = f"data:image/jpeg;base64,{encode_frame_base64(frame)}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })
    user_content.append({"type": "text", "text": task_context})

    return [
        {"role": "system", "content": " ".join(system_parts)},
        {"role": "user", "content": user_content},
    ]


def call_api(client: openai.OpenAI, model: str, messages: list,
             max_tokens: int = 256) -> str:
    """Call the VLM API and return the response text."""
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_episode(
    episode: dict,
    prompts_config: dict,
    annotation_types: list,
    client: openai.OpenAI,
    model: str,
    camera: str,
    max_frames: int,
    sample_strategy: str,
    rerun: bool,
    max_tokens: int,
    dry_run: bool,
    output_dir: str,
) -> int:
    """Process one episode: annotation each annotation segment. Returns count of segments annotationed."""
    meta = episode["meta"]
    store = episode["store"]
    annotations = episode["annotations"]
    episode_id = meta.get("episode_id", Path(episode["episode_dir"]).name)

    if not annotations:
        print(f"  [skip] no annotations")
        return 0

    # Output path
    model_slug = model.split("/")[-1]
    annotations_file = os.path.join(output_dir, episode_id, f"captions_{model_slug}.jsonl")
    existing = load_existing_annotations(annotations_file)

    prompts = prompts_config["prompts"]
    task_desc = meta.get("task_description", "")
    scene_desc = meta.get("scene_desc", "")
    objects = meta.get("objects", [])

    count = 0

    for seg_idx, ann in enumerate(annotations):
        entry = existing.get(seg_idx, {
            "segment_index": seg_idx,
            "start_idx": ann["start_idx"],
            "end_idx": ann["end_idx"],
            "annotation": ann["text"],
            "annotations": {},
        })

        done_cts = set(entry.get("annotations", {}).keys())
        to_run = [ct for ct in annotation_types if ct in prompts and (rerun or ct not in done_cts)]

        if not to_run:
            continue

        # Sample frames
        indices = sample_frame_indices(
            ann["start_idx"], ann["end_idx"], max_frames, sample_strategy
        )
        if not indices:
            print(f"  [seg {seg_idx}] empty segment, skipping")
            continue

        frames = []
        for idx in indices:
            frame = decode_frame(store, camera, idx)
            if frame is not None:
                frames.append(frame)
        if not frames:
            print(f"  [seg {seg_idx}] no decodable frames, skipping")
            continue

        # Context: previous/next annotations
        prev_ann = annotations[seg_idx - 1]["text"] if seg_idx > 0 else ""
        next_ann = annotations[seg_idx + 1]["text"] if seg_idx + 1 < len(annotations) else ""

        print(f"  [seg {seg_idx}] frames {ann['start_idx']}–{ann['end_idx']} "
              f"({len(frames)} sampled)  {ann['text'][:60]}...")

        # Compute spatial facts once per segment if any spatial types requested
        spatial_facts = ""
        if SPATIAL_ANNOTATION_TYPES & set(to_run):
            try:
                spatial_facts = compute_spatial_facts(
                    store, ann["start_idx"], ann["end_idx"]
                )
            except Exception as exc:
                print(f"  [seg {seg_idx}] spatial facts error: {exc}")

        if dry_run:
            print(f"    [dry_run] would annotation: {to_run}")
            if spatial_facts:
                print(f"    [dry_run] spatial facts:\n{spatial_facts}")
            continue

        if "annotations" not in entry:
            entry["annotations"] = {}

        for ct in to_run:
            prompt_cfg = prompts[ct]
            try:
                json_key = "reasoning" if ct == "reasoning" else "annotation"
                messages = build_messages(
                    frames,
                    task_desc, scene_desc, objects,
                    ann["text"], seg_idx, len(annotations),
                    prev_ann, next_ann,
                    prompt_cfg["prompt"],
                    constraint=prompt_cfg.get("constraint", ""),
                    json_key=json_key,
                    spatial_context=spatial_facts if ct in SPATIAL_ANNOTATION_TYPES else "",
                )
                raw = call_api(client, model, messages, max_tokens)
                try:
                    annotation = json.loads(raw)[json_key]
                except (json.JSONDecodeError, KeyError):
                    annotation = raw
                entry["annotations"][ct] = annotation
                print(f"    [{ct}] {annotation[:100]}...")
            except Exception as exc:
                print(f"    [{ct}] ERROR: {exc}")
                entry["annotations"][ct] = f"ERROR: {exc}"

        existing[seg_idx] = entry
        count += 1
        save_annotations(annotations_file, existing)

    if count > 0:
        print(f"  Saved → {annotations_file}  ({count} segments annotationed)")

    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dense annotation for EgoVerse egocentric video data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                        help="Root directory containing episode folders")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for annotations (default: same as data_dir)")
    parser.add_argument("--episode", default=None,
                        help="Process only this episode ID")
    parser.add_argument("--n", type=int, default=None,
                        help="Process at most N episodes")
    parser.add_argument("--camera", default="images.front_1",
                        help="Camera to use for frame extraction")

    # Frame sampling
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Maximum frames to sample per segment")
    parser.add_argument("--sample_strategy", default="uniform",
                        choices=["uniform", "first_last", "stride"],
                        help="Frame sampling strategy")

    # Prompts & annotations
    parser.add_argument("--prompts_json", default=DEFAULT_PROMPTS_JSON,
                        help="Prompts config JSON file")
    parser.add_argument("--annotation_type", default=None,
                        help="Comma-separated annotation types (default: all in prompts.json)")
    parser.add_argument("--rerun", action="store_true",
                        help="Overwrite existing annotations")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens for VLM response")

    # Backend (vLLM, OpenAI-compatible server)
    parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--vllm_api_key", default="EMPTY",
                        help="vLLM API key")
    parser.add_argument("--vllm_model", default="Qwen3-VL-30B-A3B-Instruct",
                        help="vLLM model name")

    # Parallel sharding
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker index for parallel runs")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Total number of parallel workers")

    # Debug
    parser.add_argument("--dry_run", action="store_true",
                        help="Load data and print plan, skip API calls")

    args = parser.parse_args()

    # Load prompts
    with open(args.prompts_json) as f:
        prompts_config = json.load(f)
    all_annotation_types = list(prompts_config["prompts"].keys())
    annotation_types = (
        [ct.strip() for ct in args.annotation_type.split(",")]
        if args.annotation_type else all_annotation_types
    )

    # Build API client (vLLM, OpenAI-compatible; skip for dry_run)
    client = None
    model = "dry_run"
    if not args.dry_run:
        client = openai.OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        model = args.vllm_model

    data_path = Path(args.dataset_dir)
    output_dir = args.output_dir or str(data_path)

    if not data_path.exists():
        print(f"[error] data_dir does not exist: {data_path}")
        return

    # Discover episodes
    if args.episode:
        episode_ids = [args.episode]
    else:
        episode_ids = sorted(
            d.name for d in data_path.iterdir()
            if d.is_dir() and (d / "zarr.json").exists()
        )

    # Worker sharding
    if args.n_workers > 1:
        episode_ids = [
            eid for i, eid in enumerate(episode_ids)
            if i % args.n_workers == args.worker_id
        ]

    if args.n is not None:
        episode_ids = episode_ids[:args.n]

    print(f"Data dir       : {data_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Model          : {model}")
    print(f"Annotation types  : {annotation_types}")
    print(f"Frame sampling : max_frames={args.max_frames}, strategy={args.sample_strategy}")
    print(f"Episodes       : {len(episode_ids)}")
    if args.n_workers > 1:
        print(f"Worker         : {args.worker_id} / {args.n_workers}")
    if args.dry_run:
        print(f"Mode           : DRY RUN")
    print()

    total_segments = 0
    for ep_id in episode_ids:
        ep_dir = str(data_path / ep_id)
        print(f"[{ep_id}]")

        try:
            episode = load_episode(ep_dir)
        except Exception as exc:
            print(f"  [error] failed to load: {exc}")
            continue

        meta = episode["meta"]
        print(f"  Task: {meta.get('task_description', 'N/A')[:80]}")
        print(f"  Frames: {episode['total_frames']} | Segments: {len(episode['annotations'])}")

        if args.camera not in episode["cameras"]:
            print(f"  [skip] camera '{args.camera}' not found (available: {episode['cameras']})")
            continue

        n = process_episode(
            episode, prompts_config, annotation_types,
            client, model, args.camera,
            args.max_frames, args.sample_strategy,
            args.rerun, args.max_tokens, args.dry_run,
            output_dir,
        )
        total_segments += n

    print(f"\nDone. {total_segments} segments annotationed across {len(episode_ids)} episodes.")


if __name__ == "__main__":
    main()
