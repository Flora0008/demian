# DeMiAn-WAM — overlay onto Cosmos-Predict 2.5

DeMiAn-WAM is a world-action model: a **Cosmos-Predict 2.5 (2B) video DiT** backbone with a GR00T-style action head, trained with rectified-flow action matching and conditioned on the DeMiAn dense annotations.
The Cosmos-Predict 2.5 framework (`cosmos_predict2`, `cosmos-oss`, `cosmos-cuda`, Cosmos-Reason 1) is **not redistributed here** — it is an external dependency. This directory contains only the **DeMiAn-specific code** that is layered on top of it, as an *overlay*.

## External dependency

- Framework: NVIDIA **Cosmos-Predict 2.5** — <https://github.com/nvidia-cosmos/cosmos-predict2> (the `cosmos_predict2` package + the `cosmos-oss` / `cosmos-cuda` workspace packages; Apache-2.0).
- Reason 1 VL encoder (`cosmos_predict2/_src/reason1`) is upstream, used as-is.

## DeMiAn modifications

```
demian_wam/
├── cosmos_predict2/_src/predict2/
│   ├── action/                       # the DeMiAn action head + experiment (entire subtree)
│   │   ├── networks/                 #   action_head, {flow,joint,joint_flow,vlm}_action_expert,
│   │   │                             #   joint_action_video_dit, ar_language_head, ...
│   │   ├── models/                   #   *_video2world_model.py (action-conditioned / rectified-flow)
│   │   ├── configs/action_conditioned/   # config.py + experiment/exp_2B_*
│   │   ├── datasets/ , inference/
│   ├── networks/minimal_v4_dit.py    # PATCH: use_crossattn_projection
│   ├── schedulers/rectified_flow.py  # PATCH: `gaussian_bell` loss weighting option
│   └── datasets/local_datasets/dataset_egoverse{,_latent,_latent_random}.py  # EgoVerse datasets
├── cosmos_predict2/experiments/base/ # experiment-registration loaders:
│                                     #   action.py
│                                     #   egoverse.py
├── configs/                          # action-head experiment YAMLs
├── groot_dreams/                     # DeMiAn dataloader/transforms
├── eval_utils/                       # RoboCasa eval harness + policy server/client
└── scripts/
    ├── train_demian_wam.slurm          # action-head training (robocasa | molmobot)
    └── train_demian_wam_egoverse.slurm # EgoVerse video2world mid-training
```

The two files under `predict2/{networks,schedulers}/` are **patches** that must overwrite their upstream counterparts; everything under `predict2/action/`, `predict2/datasets/local_datasets/dataset_egoverse*`, `cosmos_predict2/experiments/`, and the top-level dirs are new. (`dataset_video.py` and `dataset_utils.py` in `local_datasets/` are upstream — not vendored.)

The experiment loader (`cosmos_predict2/experiments/base/action.py`) registers `dreamdojo_<stem>` for every `configs/*.yaml`; `2b_256_robocasa_action_head.yaml` and `2b_256_molmobot_action_head.yaml` give the RoboCasa365 and MolmoBot/MolmoSpaces training experiments. MolmoBot reuses DeMiAn-VLA's `MlSpacesDataset` (via the `MlSpacesVideoActionDataset` adapter in `groot_dreams`), so it additionally needs `molmobot_pi0` importable — point `MOLMOBOT_PI0_PATH` at the repo root (which holds `molmobot_pi0`).

### Use upstream as-is

- `predict2/conditioner.py`
- `predict2/networks/attention.py`
- `scripts/train.py` — was a 5-line shim into `cosmos_oss.scripts.train`; the
  launcher calls `python -m cosmos_oss.scripts.train` directly instead.
- `cosmos_predict2/_src/{imaginaire,reason1}`, the rest of `predict2/`, and the `cosmos-oss`/`cosmos-cuda`/`cosmos-gradio` packages — all upstream.

## Applying the overlay

```bash
# 1. clone + install the framework (see the Cosmos-Predict2.5 README)
git clone https://github.com/nvidia-cosmos/cosmos-predict2 cosmos-predict2.5
cd cosmos-predict2.5
pip install -e ./packages/cosmos-oss -e ./packages/cosmos-cuda -e .

# 2. overlay the DeMiAn code: predict2/action + experiments/ + the two patched
#    files (under cosmos_predict2/), the experiment YAMLs (configs/), and the
#    top-level packages.
cp -r /path/to/release/demian_wam/cosmos_predict2 .
cp -r /path/to/release/demian_wam/{configs,groot_dreams,eval_utils,scripts} .
# (groot_dreams / eval_utils import as top-level packages; PROJECT_DIR is on
#  PYTHONPATH. The experiment loader reads <repo-root>/configs/*.yaml.)
```

## Launchers

Pass node/GPU/partition on the `sbatch` line; paths come from env (see each launcher's header and the Reproduce sections below). Paths below are relative to `demian_wam/`. `<aspect>` ∈ `baseline | physical_motion | scene_composition | arm_pose | reasoning | mix5`.

| Stage | Suite | Launcher |
|---|---|---|
| Train DeMiAn-WAM | RoboCasa / MolmoBot | `scripts/train_demian_wam.slurm <robocasa\|molmobot> <aspect>` |
| Train DeMiAn-WAM (mid-train) | EgoVerse | `scripts/train_demian_wam_egoverse.slurm <baseline\|dense>` |
| Eval WAM (RoboCasa) | RoboCasa | `scripts/eval_demian_wam_robocasa.slurm` |
| Eval WAM (MolmoSpaces) | MolmoSpaces | `molmobot_pi0/eval/eval.py` via `eval_cosmos_wam_action_head` (repo root) |

## Reproduce — Train DeMiAn-WAM

`scripts/train_demian_wam.slurm <embodiment> <aspect>` trains one annotation-aspect ablation (paper Table 1) on one embodiment. The DiT backbone, visual data, and action targets are fixed across aspects; only the caption conditioning changes. RoboCasa selects captions via the `caption_mode` Hydra override (`ep_*` keys); MolmoBot reuses Pi0's `MlSpacesDataset` and selects captions via the `MLSPACES_CAPTION_KEY` env (Pi0 keys).

| Aspect (paper)    | launcher arg        | `robocasa` (caption_mode) | `molmobot` (MLSPACES_CAPTION_KEY) |
|-------------------|---------------------|---------------------------|-----------------------------------|
| Baseline          | `baseline`          | `none`                    | *(none)*                          |
| Physical Motion   | `physical_motion`   | `ep_action`               | `current_action`                  |
| Scene Composition | `scene_composition` | `ep_env_simple`           | `environment-simple`              |
| Arm Pose          | `arm_pose`          | `ep_arm_state`            | `agent_arm_state`                 |
| Reasoning         | `reasoning`         | `ep_reasoning`            | `reasoning-short`                 |
| mix5 (instructor base) | `mix5`              | `random:none,ep_action,…` | `random:none,current_action,…`    |

```bash
# node count / GPUs / partition / account passed at submit time (not hardcoded).
# Common env: PROJECT_DIR (overlaid clone), DATASET_PATH, PRETRAIN_CKPT, TOKENIZER_CKPT.

# RoboCasa365:
DATASET_PATH=<robocasa_lerobot> sbatch --nodes=1 --gpus-per-node=8 \
    scripts/train_demian_wam.slurm robocasa mix5

# MolmoBot/MolmoSpaces (additionally: MOLMOBOT_PI0_PATH=<repo root>,
#   and for non-baseline aspects MLSPACES_CAPTIONS_DIR=<data/molmobot-captions>):
DATASET_PATH=<molmobot_subset> MOLMOBOT_PI0_PATH=<repo root> \
MLSPACES_CAPTIONS_DIR=<data/molmobot-captions> \
    sbatch --nodes=1 --gpus-per-node=8 scripts/train_demian_wam.slurm molmobot reasoning
```

Entrypoint: `python -m cosmos_oss.scripts.train --config=…/action_conditioned/config.py -- experiment=dreamdojo_2b_256_{robocasa,molmobot}_action_head …`. Defaults: 100K iters, global batch 256 (4/GPU × 8 GPU × grad-accum 8 on 1 node), action-head LR 5e-5, 3K warmup, save every 500, auto-resume from the latest complete checkpoint + auto-resubmit on wall-time signal. If `CONTAINER` is unset the job runs in the current environment; if set, under `srun --container-image`.

### Architecture variants

DeMiAn-WAM uses the **joint-flow** action expert (`use_joint_flow_action_expert`, 4 layers tapping DiT blocks `[6,13,20,27]` = K=4), FFN ratio 4, and the Cosmos-Reason1 **VLM-image** prefix. The configs are vendored under `configs/`; select one by overriding `EXPERIMENT` (the loader registers `dreamdojo_<yaml-stem>`):

```bash
EXPERIMENT=dreamdojo_2b_256_robocasa_action_head_full_jflow_ffn4_vlmimg \
    sbatch --nodes=1 --gpus-per-node=8 scripts/train_demian_wam.slurm robocasa mix5

EXPERIMENT=dreamdojo_2b_256_molmobot_action_head_jflow_ffn4_vlmimg_full \
DATASET_PATH=<molmobot_subset> MOLMOBOT_PI0_PATH=<repo root> \
MLSPACES_CAPTIONS_DIR=<data/molmobot-captions> \
    sbatch --nodes=1 --gpus-per-node=8 scripts/train_demian_wam.slurm molmobot mix5
```

## Reproduce — EgoVerse mid-training

The scaling experiment (RQ4 Fig A in the paper) mid-trains the 2B video DiT on 50K EgoVerse human-egocentric clips with a **video-prediction** objective (no action head); the DeMiAn dense annotations enter as language conditioning. This uses the upstream **video2world** config plus the vendored `experiments/base/egoverse.py` (which self-registers the experiments) and the `dataset_egoverse*` datasets.

```bash
# variant: baseline (annotation only) vs dense (DeMiAn dense annotations).
# Set PROJECT_DIR (overlaid clone), PRETRAIN_CKPT (2B base ema_bf16.pt),
# TOKENIZER_CKPT, and the EGOVERSE_* data dirs via env.
EGOVERSE_DATASET_DIR=<egoverse_root> \
    sbatch --nodes=8 --gpus-per-node=8 \
    scripts/train_demian_wam_egoverse.slurm dense
```

Entrypoint: `python -m cosmos_oss.scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_egoverse_{annotation,dense_human,…}`. The EgoVerse data roots are read from `EGOVERSE_DATASET_DIR` / `EGOVERSE_LATENT_DIR_*` by `experiments/base/egoverse.py`. Pass any other registered variant directly as the launcher argument or via `EXPERIMENT=`.

## Reproduce — RoboCasa eval (DeMiAn-WAM)

Same harness/server shape as the VLA side, but the policy server wraps the WAM:
- `scripts/convert_dcp_to_regular.py` — converts the trained DCP checkpoint to a regular `.pt` (once).
- `socket_test_cosmos_wam.py` — serves the WAM policy over a socket.
- `eval_utils/run_robocasa_eval.py` (+ `robocasa_task_list.py`, `policy_client.py`, `merge_results.py`, `aggregate_seeds.py`) — the RoboCasa sim harness.

Launcher: `scripts/eval_demian_wam_robocasa.slurm` (set `EXPERIMENT` = the trained WAM experiment, `CKPT_DCP` = the DCP iter dir, `TOKENIZER_CKPT`, `ROBOCASA_PATH`, `ROBOSUITE_PATH`). It converts → serves → runs the harness; inference-time GT caption injection via `GT_REASONING_FILE` + `CAPTION_KEY`.
