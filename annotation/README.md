# DeMiAn Dense Annotation Pipeline

This directory holds the dense-annotation pipeline used to generate the per-segment, multi-aspect annotations that DeMiAn trains and evaluates on.

## Layout

```
annotation/
├── robocasa_molmobot/        # RoboCasa + MolmoBot annotation (Qwen3-VL-30B-A3B-Instruct)
│   ├── dense_annotator.py     # RoboCasa: per-episode / per-segment annotation
│   ├── molmobot_annotator.py  # MolmoBot
│   ├── prompts.json           # aspect prompts (preserved verbatim — exact prompts used)
│   └── prompts_atomic.json    # aspect prompts for the atomic RoboCasa split
└── egoverse/                 # EgoVerse human egocentric annotation (Qwen3-VL-30B-A3B-Instruct)
    ├── dense_annotator.py     # per-segment annotation from the Zarr episode store
    ├── prompts.json           # aspect prompts (preserved verbatim)
    ├── visualize_episode.py   # overlay annotation text on decoded frames for inspection
    ├── requirements.txt
    └── README.md              # EgoVerse data layout + usage detail
```

## Annotation aspects

DeMiAn's four annotation aspects (paper name → prompt key in `prompts.json`):

| Paper aspect        | Prompt key            |
|---------------------|-----------------------|
| Physical motion     | `current_action`      |
| Scene composition   | `environment-simple`  |
| Arm pose            | `agent_arm_state`     |
| Reasoning           | `reasoning-short`     |

pass `--annotation_type` to select the subset to generate.

## Labeling models

- RoboCasa / MolmoBot: `Qwen3-VL-30B-A3B-Instruct`
- EgoVerse: `Qwen3-VL-30B-A3B-Instruct`

Both annotators use a vLLM (OpenAI-compatible) backend (`--vllm_base_url ... --vllm_model ...`)

## Note on the output field name

These annotators were renamed from "caption" to "annotation" to match the paper's terminology, so they now emit JSONL with an `annotations` field and `annotations_<model>.jsonl` filenames. The **already-released** annotation artifacts and the downstream training/eval readers (`molmobot_pi0`, the `generate_gt_*` scripts) still use the legacy `captions` field / `captions_*.jsonl` naming. If you regenerate annotations with this pipeline, either keep the `captions` field for compatibility or rename it in the readers as well.
