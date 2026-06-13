# DeMiAn Instructor

Trains the Qwen3.5-2B model that supplies the task-appropriate annotation at inference (released checkpoints: `checkpoints/instructor/{molmospaces,robocasa}`). There are two embodiment-specific trainers — MolmoBot/MolmoSpaces conditions on 2 camera views (exo + wrist), RoboCasa on 3 (agentview L/R + eye-in-hand). Each launcher's header documents its env vars; paths below are relative to the repo root.

| Embodiment | Data prep | Trainer | Launcher |
|---|---|---|---|
| MolmoBot / MolmoSpaces | `scripts/prepare_instructor_data_molmobot.py` | `scripts/instructor/train_instructor_molmobot.py` | `scripts/slurm/instructor/train_instructor_sft_*.slurm` |
| RoboCasa | `scripts/prepare_instructor_data_robocasa.py` (routing: `runs/robocasa/task_caption_routing_augmented.json`) | `scripts/instructor/train_instructor_robocasa.py` | `scripts/slurm/instructor/train_instructor_sft_robocasa.slurm` |

The RoboCasa data prep reads the released `data/robocasa365_v1` episodes + packed frames (see `data/README.md`) and the per-task routing table, builds `(3 images, task_description, caption)` SFT examples, then the trainer LoRA-finetunes Qwen3.5-2B. Training needs `peft`/`trl`/`datasets` (the launcher pip-installs them; `docker/Dockerfile.instructor` is the serving image, not the training one).
