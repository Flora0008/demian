# [DeMiAn] How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning

<p align="center">
  <a href="https://arxiv.org/pdf/2605.17077"><img src="https://img.shields.io/badge/arXiv-2605.17077-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/PEARLS-Lab/DeMiAn"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-FFD21E?logoColor=black" alt="Hugging Face"></a>
  <a href="https://pearls-lab.github.io/demian"><img src="https://img.shields.io/badge/Project-Website-2ea44f?logo=github&logoColor=white" alt="Website"></a>
</p>

<p align="center">
  <img src="docs/assets/demian_intro.gif" width="100%" alt="DeMiAn overview">
</p>

---

DeMiAn is instantiated on two policy architectures and two simulators:

Policy architectures:

- **DeMiAn-VLA** — `openpi` 0.5 / Pi0.5 backbone. Code: [`openpi`](https://github.com/Physical-Intelligence/openpi) + [`molmobot_pi0`](https://github.com/allenai/MolmoBot/tree/main/MolmoBot-Pi0).
- **DeMiAn-WAM** — Cosmos-Predict 2.5 video DiT + action head. Code:
  `demian_wam` (overlay onto external Cosmos-Predict 2.5).

Environments:

- **[RoboCasa 365](https://robocasa.ai/)**: 17 atomic + composite tasks
- **[MolmoSpaces](https://github.com/allenai/molmospaces)**: bench-v2, 9 benches

```
(image, task description) ──► Instructor (Qwen3.5-2B) ──► annotation
                                                              │
              task_description + annotation ──► Policy (DeMiAn-VLA / DeMiAn-WAM) ──► actions
```

## Layout

```
demian/                             # repo root
├── README.md                       # ← you are here
├── docker/                         # Dockerfile (VLA env) + Dockerfile.instructor
├── openpi/                         # vendored Pi0.5 PyTorch backbone (no jax model code)
│                                   #   + scripts/eval (RoboCasa harness, serve_policy, instructor server)
├── molmobot_pi0/                   # DeMiAn-VLA policy/dataset code + Hydra eval entrypoint
├── demian_wam/                     # DeMiAn-WAM overlay onto Cosmos-Predict 2.5 (action head, egoverse, eval)
├── config/eval/                    # Hydra eval config
├── runs/{molmobot,robocasa}/       # instructor SFT and per-task annotation routing tables
├── scripts/                        # scripts/slurm
├── docs/                           # GitHub Pages project page (index.md)
└── data/README.md                  # dataset download (HF)
```

## Setup

### 1. Environment (Docker)

You can build the environments from the provided Dockerfiles:

```bash
docker build -f docker/Dockerfile            -t demian-vla:latest .       # VLA train + eval (openpi 0.5 / Pi0.5)
docker build -f docker/Dockerfile.instructor -t demian-instructor:latest . # Qwen3.5-2B instructor server (async eval)
```

- `docker/Dockerfile` — torch 2.7.1 + JAX (CPU tree-utils only) + `transformers==4.53.2`
  with the PaliGemma/Gemma patches pre-applied; covers VLA post-training and the
  policy side of eval.
- `docker/Dockerfile.instructor` — newer `transformers` for the `qwen3_5` instructor
  (kept separate from the policy's pinned 4.53.2; the two co-host at eval over HTTP).
- **DeMiAn-WAM** uses the upstream **Cosmos-Predict 2.5** environment (build it from
  that repo's own Dockerfile), then apply the overlays in `demian_wam/` — see
  `demian_wam/README.md`.

### 2. Code + external repos

- [`openpi`](https://github.com/Physical-Intelligence/openpi) — vendored Pi0.5
  PyTorch backbone, adapted from Physical Intelligence's **openpi** (no separate
  clone needed; `PYTHONPATH=openpi/src:openpi/packages/openpi-client/src:.`, where
  `.` = repo root).
- [`molmobot_pi0`](https://github.com/allenai/MolmoBot/tree/main/MolmoBot-Pi0) —
  DeMiAn-VLA policy / dataset / eval code, adapted from AllenAI's **MolmoBot**
  (`MolmoBot-Pi0`).
- **Simulators**: [`molmospaces`](https://github.com/allenai/molmospaces) and [`robocasa`](https://github.com/robocasa/robocasa) + [`robosuite`](https://github.com/ARISE-Initiative/robosuite).
- [Cosmos-Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) for DeMiAn-WAM.

### 3. Datasets + checkpoints

The dataset and checkpoints are hosted on Hugging Face. See `data/README.md`
for exact commands. In short:

| Artifact | Location |
|---|---|
| Checkpoints (VLA/WAM policies + instructors) | HF [`PEARLS-Lab/DeMiAn`](https://huggingface.co/PEARLS-Lab/DeMiAn) → `checkpoints/` |
| DeMiAn dense annotations | HF [`PEARLS-Lab/DeMiAn_data`](https://huggingface.co/datasets/PEARLS-Lab/DeMiAn_data) `molmobot-captions/` → `data/molmobot-captions/` |
| RoboCasa 365 training set | HF [`PEARLS-Lab/DeMiAn_data`](https://huggingface.co/datasets/PEARLS-Lab/DeMiAn_data) `robocasa365_v1/` → `data/robocasa365_v1/` |
| MolmoBot post-training set | [`allenai/MolmoBot`](https://github.com/allenai/MolmoBot) |

### 4. Slurm Scripts

The launcher scripts are listed in each component's README:
- DeMiAn-VLA (train + eval): [`molmobot_pi0/README.md`](molmobot_pi0/README.md)
- DeMiAn-WAM (train + eval): [`demian_wam/README.md`](demian_wam/README.md)
- Instructor (train): [`scripts/instructor/README.md`](scripts/instructor/README.md)

## Train DeMiAn-VLA (MolmoBot policy)

The annotation-aspect ablations (paper Table 1) are trained with `scripts/slurm/train_demian_vla.slurm`, which fine-tunes the Pi0.5 PyTorch backbone (`openpi`, see its README) via `molmobot_pi0/train_openpi_pytorch.py`. The aspect is passed as an argument and selects the appended-caption type; the policy, visual data, and action targets are held fixed across aspects.

| Aspect (paper)      | launcher arg        | caption key (data sub-key) |
|---------------------|---------------------|----------------------------|
| Baseline            | `baseline`          | *(none; task only)*        |
| Physical Motion     | `physical_motion`   | `current_action`           |
| Scene Composition   | `scene_composition` | `environment-simple`       |
| Arm Pose            | `arm_pose`          | `agent_arm_state`          |
| Reasoning           | `reasoning`         | `reasoning-short`          |
| mix5 (instructor base) | `mix5`           | `random:none,current_action,reasoning-short,environment-simple,agent_arm_state` |

```bash
# node count / GPUs-per-node / partition / account are passed at submit time
# Set DATA_ROOT, PRETRAINED_CKPT, CONTAINER via env first.
sbatch --nodes=2 --gpus-per-node=8 scripts/slurm/train_demian_vla.slurm mix5
```

Defaults: 40K steps (the checkpoint evaluated in Table 1), global batch 256.

## Train DeMiAn-VLA (RoboCasa)

The RoboCasa365 VLA uses the same Pi0.5 backbone via a parallel launcher, `scripts/slurm/train_demian_vla_robocasa.slurm`, fine-tuning on the RoboCasa365 atomic split (`data/robocasa365_v1`; see `data/README.md` to obtain it and reproduce the packed images). Unlike the MolmoBot recipe, the annotated aspects add the **auxiliary LM loss** (λ_LM = 0.1, config `pi05_robocasa_lm`); the baseline uses `pi05_robocasa` with no LM loss. The aspect argument selects both the config and the appended-caption key:

| Aspect (paper)      | launcher arg        | config             | caption key (`--data.caption-key`) |
|---------------------|---------------------|--------------------|------------------------------------|
| Baseline            | `baseline`          | `pi05_robocasa`    | *(none; no LM loss)*               |
| Physical Motion     | `physical_motion`   | `pi05_robocasa_lm` | `current_action`                   |
| Scene Composition   | `scene_composition` | `pi05_robocasa_lm` | `environment-simple`               |
| Arm Pose            | `arm_pose`          | `pi05_robocasa_lm` | `agent_arm_state`                  |
| Reasoning           | `reasoning`         | `pi05_robocasa_lm` | `reasoning-short`                  |
| mix5 (instructor base) | `mix5`              | `pi05_robocasa_lm` | `random:none,current_action,reasoning-short,environment-simple,agent_arm_state` |

```bash
# Set DATA_ROOT (RoboCasa365 LeRobot root, e.g. data/robocasa365_v1),
# PRETRAINED_CKPT, and CONTAINER via env first.
sbatch --nodes=1 --gpus-per-node=8 scripts/slurm/train_demian_vla_robocasa.slurm reasoning
```

## Evaluation

- **DeMiAn-VLA** — eval launchers (RoboCasa, RoboCasa + instructor, composite tasks, MolmoSpaces + instructor) in [`molmobot_pi0/README.md`](molmobot_pi0/README.md); the RoboCasa eval (websocket policy server + sim) is detailed in [`openpi/README.md`](openpi/README.md).
- **DeMiAn-WAM** — eval launchers and the RoboCasa eval in [`demian_wam/README.md`](demian_wam/README.md).

## Citation

```bibtex
@misc{kim2026instructrobotdenselanguage,
      title={How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning}, 
      author={Bosung Kim and Ruiyi Wang and David Acuna and Jaehun Jung and Alexander Trevithick and Brandon Cui and Yejin Choi and Prithviraj Ammanabrolu},
      year={2026},
      eprint={2605.17077},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.17077}, 
}
```
