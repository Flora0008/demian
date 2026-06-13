# Data Pipeline

## Installation

The pipeline runs on Python 3.10+. Install the data-pipeline deps:

```bash
# Inside a container session (srun --container-image=… --pty bash).
pip install --upgrade 'zarr>=3.0.0'
pip install opencv-python-headless openai
```

Dependencies (`requirements.txt`):
- `zarr>=3.0` — reading EgoVerse Zarr v3 datasets (container ships 2.x; upgrade needed)
- `opencv-python-headless>=4.0` — frame decoding, video writing, image encoding
- `openai>=1.0` — VLM API client (used for vLLM, Gemini, and Anthropic backends)
- `numpy>=1.24` — ships with most PyTorch containers

For the vLLM backend, run the VLM server separately (in a container or a GPU job): start `vllm serve <model>` and point `dense_annotator.py --vllm_base_url` at its localhost endpoint.

## Downloading EgoVerse data

EgoVerse is a public dataset — download it from the official repository and follow its setup / download instructions:

**https://github.com/GaTech-RL2/EgoVerse**

Once you have a local copy, point the annotation scripts at it with `--dataset_dir <path>` (the dataset root holds one per-episode Zarr store per directory — see *EgoVerse Dataset Overview* below). The *Dense Annotation* section has the annotation commands.

## EgoVerse Dataset Overview

### Statistics
- Episodes are single- or multi-camera: single-camera episodes have
  `images.front_1` only; multi-camera episodes add `images.left_wrist` +
  `images.right_wrist`. The large majority are single-camera.
- Annotations are per-episode (shared across all camera views).

### Format
Zarr v3 format. Each episode is a directory named by episode ID (e.g., `69a6625fb29228ae08b91423/`).

### Visualization

Use `visualize_episode.py` to inspect episodes. It decodes JPEG frames from zarr, overlays annotation text and frame counters, and outputs annotated videos or frame grids.

```bash
# Grid of 16 sampled frames (fast preview)
python visualize_episode.py --episode 69a6625fb29228ae08b91423 --save_grid --no_video

# Full annotated MP4 video
python visualize_episode.py --episode 69a6625fb29228ae08b91423

# Video with frame skip (faster, smaller file)
python visualize_episode.py --episode 69a6625fb29228ae08b91423 --skip 3

# Custom data directory and output location
python visualize_episode.py --episode EPISODE_ID --data_dir /path/to/data --output_dir vis_output

# Specific camera (for multi-camera episodes)
python visualize_episode.py --episode EPISODE_ID --camera images.left_wrist
```

Options:
- `--episode`: episode ID (required)
- `--data_dir`: root directory containing episode folders (default: `data/`)
- `--output_dir`: where to save outputs (default: `vis_output/`)
- `--camera`: specific camera to visualize (default: all available)
- `--fps`: output video FPS (default: 30)
- `--skip`: frame skip for video (default: 1)
- `--save_grid`: save a grid of sampled frames as JPEG
- `--grid_samples`: number of frames in grid (default: 16)
- `--no_video`: skip video generation

## Dense Annotation

`dense_annotator.py` generates dense annotations for each annotation segment by sampling frames and sending them to a VLM.

### Batch annotation (sharded)

For large batches, run a VLM server (Qwen3-VL-30B-A3B-Instruct via `vllm serve`) and shard the work across N parallel `dense_annotator.py` processes with
`--worker_id <i> --n_workers <N>` — each worker handles the episodes where `ep_idx % n_workers == worker_id`, so no coordination is needed.

```bash
# one worker of N, against a running vLLM server:
python dense_annotator.py \
    --dataset_dir /path/to/egoverse_dataset \
    --vllm_base_url http://localhost:8000/v1 --vllm_model Qwen3-VL-30B-A3B-Instruct \
    --worker_id 0 --n_workers 16
```

The single-episode `python dense_annotator.py …` flow below is for ad-hoc work.


### How it works

For each annotation segment (e.g., frames 279-453: "inject liquid into flask"), it samples up to N frames, sends them as images to the VLM with task context (task description, scene, objects, previous/next annotations), and generates multiple annotation types.

### Annotation types

Defined in `prompts.json`:
- `current_action` — what the person's hands are physically doing
- `environment` — scene and workspace description
- `hand_start_state` / `hand_end_state` — hand poses at segment boundaries
- `reasoning` — why this action, what comes before/after
- `object_state_before` / `object_state_after` / `object_state_delta` — task-relevant object state at boundaries and what changed
- `spatial_start` / `spatial_end` / `spatial_motion` — proprio-grounded spatial descriptions
- `dense` — a single robot/agent-centric subtask description tuned to feed an action expert (target objects + verbs + directions + stage cues)

### Output

`{output_dir}/{episode_id}/annotations_{model}.jsonl` — one JSON line per segment, each containing:
```json
{
  "segment_index": 0,
  "start_idx": 0,
  "end_idx": 279,
  "annotation": "place syringe into small bottle, extract liquid from small bottle using syringe",
  "annotations": {
    "current_action": "The person's left hand holds a small bottle steady while ...",
    "reasoning": "The person is extracting liquid from a small bottle using a syringe ..."
  }
}
```

### Usage

```bash
# Dry run (no API calls, just verify data loading)
python dense_annotator.py --dry_run --dataset_dir /path/to/data --episode 69a6625fb29228ae08b91423

# Annotate one episode (vLLM, OpenAI-compatible server)
python dense_annotator.py \
    --vllm_base_url http://localhost:8000/v1 \
    --vllm_model Qwen3-VL-30B-A3B-Instruct \
    --dataset_dir /path/to/data \
    --episode 69a6625fb29228ae08b91423

# Frame sampling options
python dense_annotator.py --vllm_base_url http://localhost:8000/v1 \
    --vllm_model MODEL --max_frames 12 --sample_strategy first_last

# Specific annotation types only
python dense_annotator.py --vllm_base_url http://localhost:8000/v1 \
    --vllm_model MODEL --annotation_type current_action,reasoning

# Parallel processing (e.g., 4 workers)
python dense_annotator.py --vllm_base_url http://localhost:8000/v1 \
    --vllm_model MODEL --worker_id 0 --n_workers 4
```

### Starting the vLLM server

The vLLM backend requires a running server. To start one with Qwen3-VL-235B on 8 H100 GPUs:

```bash
vllm serve /path/to/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --limit-mm-per-prompt '{"image": 15}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --served-model-name Qwen3-VL-30B-A3B-Instruct \
    --port 8000

# Check readiness
curl -sf http://localhost:8000/health && echo "READY"

# Stop later
kill $(lsof -ti :8000)
```

### Options
- `--dataset_dir`: root directory containing episode folders (default: `egoverse_dataset/`)
- `--output_dir`: output directory for annotations (default: same as dataset_dir)
- `--episode`: process only this episode ID
- `--n`: process at most N episodes
- `--camera`: camera to use (default: `images.front_1`)
- `--max_frames`: max frames per segment (default: 10)
- `--sample_strategy`: `uniform` (default), `first_last`, or `stride`
- `--annotation_type`: comma-separated types (default: all)
- `--rerun`: overwrite existing annotations
- `--dry_run`: print plan without API calls
- `--worker_id` / `--n_workers`: parallel sharding
- `--vllm_base_url`: vLLM server URL (default: `http://localhost:8000/v1`)
- `--vllm_model`: model name served by vLLM

### Example output

Episode `69a6625fb29228ae08b91423` (Qwen3-VL-30B, 8 frames per segment, `current_action` + `reasoning`):

```
Segment 0 (frames 0-279): "place syringe into small bottle, extract liquid"
  [current_action] The person's left hand holds a small bottle steady while the right hand
                   inserts a syringe into it and draws liquid upward. The right hand then
                   lifts the syringe, now filled, while the left hand continues to support
                   the bottle.
  [reasoning]      The person is extracting liquid from a small bottle using a syringe,
                   likely to precisely measure or transfer it to another container. This is
                   a preparatory step for mixing ingredients, possibly for perfume or
                   cosmetic formulation.

Segment 5 (frames 1212-1509): "pour liquid from small cylindrical flask into beaker"
  [current_action] The person's right hand holds a small cylindrical flask and pours its
                   liquid into a beaker held in the left hand. The left hand tilts the
                   beaker slightly to receive the liquid while the right hand lowers the
                   flask after pouring.
  [reasoning]      The person is pouring liquid from a small flask into a beaker to combine
                   ingredients, likely as part of mixing a custom perfume or solution. This
                   is a preparatory step following the use of a syringe to transfer liquid,
                   and precedes stirring with a hand mixer to homogenize the mixture.

Segment 7 (frames 2079-2245): "place hand mixer on table, place beaker on table"
  [current_action] The person places the hand mixer down on the table with their right hand
                   and sets the beaker down with their left hand, both movements directed
                   downward onto the glass counter.
  [reasoning]      The person has finished mixing the liquid and is now placing the hand
                   mixer and beaker down, indicating the completion of the main manipulation
                   phase. This is a cleanup or concluding step after blending the mixture.
```

## Appendix: EgoVerse Dataset Details

### Metadata (`zarr.json`)
Top-level attributes per episode:
- `embodiment`: e.g., `MECKA_BIMANUAL`
- `total_frames`: number of frames (e.g., 2245)
- `fps`: 30
- `task_name`: task identifier
- `task_description`: natural-language task summary
- `environment_id` / `scene_id` / `scene_desc`: environment context
- `objects`: list of objects in the scene (e.g., `["bottle", "measuring_cup", "liquid"]`)
- `intrinsics`: camera intrinsics (focal length, principal point, distortion coefficients)
- `duration`: episode duration in seconds
- `episode_id`, `user_id`

### Data Arrays

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `images.front_1` | `(T,)` | JPEG bytes | Egocentric RGB frames (640x360x3, stored as compressed JPEG). Present in all episodes. |
| `images.left_wrist` | `(T,)` | JPEG bytes | Left wrist camera frames. Present in ~4% of episodes (917/24303). |
| `images.right_wrist` | `(T,)` | JPEG bytes | Right wrist camera frames. Present in ~4% of episodes (917/24303). |
| `annotations` | `(N,)` | JSON bytes | Temporal action annotations (see below) |
| `left.obs_ee_pose` | `(T, 7)` | float64 | Left end-effector pose (position xyz + quaternion xyzw) |
| `right.obs_ee_pose` | `(T, 7)` | float64 | Right end-effector pose |
| `left.obs_wrist_pose` | `(T, 7)` | float64 | Left wrist pose |
| `right.obs_wrist_pose` | `(T, 7)` | float64 | Right wrist pose |
| `left.obs_keypoints` | `(T, 63)` | float64 | Left hand keypoints (21 joints x 3D) |
| `right.obs_keypoints` | `(T, 63)` | float64 | Right hand keypoints |
| `obs_head_pose` | `(T, 7)` | float64 | Head pose |

Where `T` = `total_frames`, `N` = number of annotation segments.

### Annotations
Each annotation is a JSON object with:
- `text`: action description (e.g., `"place syringe into small bottle, extract liquid from small bottle using syringe"`)
- `start_idx`: start frame index (inclusive)
- `end_idx`: end frame index (exclusive)

Annotations partition the full episode into consecutive, non-overlapping segments. Example (episode `69a6625fb29228ae08b91423`, T=2245, 75s):
```
[0]    0-279:   "place syringe into small bottle, extract liquid from small bottle using syringe"
[1]  279-453:   "inject liquid into small cylindrical flask using syringe"
[2]  453-702:   "place syringe into small bottle, extract liquid from small bottle using syringe"
[3]  702-969:   "inject liquid into small cylindrical flask using syringe"
[4]  969-1212:  "place syringe on table, pick up small cylindrical flask"
[5] 1212-1509:  "pour liquid from small cylindrical flask into beaker"
[6] 1509-2079:  "pick up hand mixer, stir liquid in bottle using hand mixer"
[7] 2079-2245:  "place hand mixer on table, place beaker on table"
```
