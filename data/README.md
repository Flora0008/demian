# Datasets

DeMiAn re-annotates **existing** public datasets; no new demonstrations are collected. This repo ships **code only** — the dense annotations,
processed datasets, and model checkpoints are hosted on Hugging Face. Download them into a `data/` (and `checkpoints/`) directory alongside the repo, then point the launchers' env vars at them.

| Artifact | Hugging Face repo |
|----------|-------------------|
| Model checkpoints | [`PEARLS-Lab/DeMiAn`](https://huggingface.co/PEARLS-Lab/DeMiAn) (model) |
| Dense annotations + RoboCasa 365 | [`PEARLS-Lab/DeMiAn_data`](https://huggingface.co/datasets/PEARLS-Lab/DeMiAn_data) (dataset) |

## Model checkpoints

```bash
hf download PEARLS-Lab/DeMiAn --local-dir checkpoints
```

Layout (weights only): `DeMiAn-VLA/{molmobot/40000,openpi}` (Pi0.5 VLA policies —
MolmoBot mix5 + RoboCasa), `DeMiAn-WAM/{molmo,robocasa}` (Cosmos WAM policies),
`EgoVerse/iter_000030000/model` (WAM mid-training, DCP), and
`instructor/{molmospaces,robocasa}` (the Qwen3.5-2B instructors).

## DeMiAn dense annotations

```bash
hf download PEARLS-Lab/DeMiAn_data --repo-type dataset \
    --include "molmobot-captions/*" --local-dir data
```

Dense annotations for the MolmoBot configs (`*.jsonl`, one per config; `prompts.json`, `manifest.json`), generated with Qwen3-VL-30B-A3B-Instruct. Each JSONL record stores its per-aspect text under an `annotations` field; the data files and directories retain the legacy `captions` name (e.g. `data/molmobot-captions`, `captions_*.jsonl`). The MolmoSpaces per-bench GT annotation JSONs are derived from these by `scripts/generate_gt_*.py` (point `MLSPACES_CAPTIONS_DIR` / the scripts' `CAPTIONS_DIR` at `data/molmobot-captions`).

## RoboCasa 365 training set (DeMiAn-VLA / DeMiAn-WAM on RoboCasa)

```bash
hf download PEARLS-Lab/DeMiAn_data --repo-type dataset \
    --include "robocasa365_v1/*" --local-dir data
```

LeRobot-format RoboCasa 365 atomic split (the `atomic_split_processed/lerobot` layout the configs expect). Point `DATA_ROOT` (VLA) / `DATASET_PATH` (WAM) at `data/robocasa365_v1`. RoboCasa **atomic** GT annotations are built from this set with `openpi/scripts/merge_fg_captions_for_eval.py`; **composite** GT annotations are shipped at `openpi/scripts/eval/gt_captions_composite.json`.

*Note:* This dataset ships only the *non-image* artifacts: the 7-dim arm action parquet (`data/`), per-episode metadata + reasoning text (`meta/episodes.jsonl`, `info.json`, `tasks.jsonl`, `modality.json`), and normalization stats (`assets/`, `dataset_statistics*.json`). The bulk of the dataset — the per-episode packed camera frames (`packed/episode_*.npz`, three 256×256 cameras), which the loader reads — is reproducible from the original RoboCasa 365 demonstrations.

**Reproducing the packed images.** The packed arrays are derived entirely from the public RoboCasa 365 data. Convert the original RoboCasa 365 atomic split
(per-subtask LeRobot dirs with mp4 videos, annotated with the RoboCasa annotator in `annotation/robocasa_molmobot/`) into the trainable layout with:

```bash
python openpi/scripts/preprocess_atomic_split.py \
    --input_dir  /path/to/robocasa/.../pretrain/atomic_split \
    --output_dir data/robocasa365_v1 \
    --caption_model <annotation-model> --caption_key reasoning-short \
    --num_workers 16
```

This decodes the source mp4s into `packed/episode_*.npz`, strips actions to the 7-dim arm format, and writes the per-episode reasoning text used by the
`pi05_robocasa_lm` LM-loss training. (The shipped `data/`, `meta/`, and `assets/` above are the output of exactly this step, sans `packed/`.)

## MolmoBot subset (DeMiAn-VLA / DeMiAn-WAM post-training)

The MolmoBot subset is the post-training corpus for the MolmoBot embodiment. It is a 20% random sample of the public MolmoBot dataset and can be reconstructed from that [source](https://github.com/allenai/MolmoBot).

## EgoVerse (WAM mid-training)

Used only for the EgoVerse mid-training stage. Obtain EgoVerse from its source and point `EGOVERSE_DATASET_DIR` (+ `EGOVERSE_LATENT_DIR_*`) at it; the DeMiAn annotations are applied by the pipeline in the paper.
