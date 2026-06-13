# DeMiAn-VLA (`molmobot_pi0`)

DeMiAn-VLA policy / dataset / eval code (adapted from AllenAI's MolmoBot [`MolmoBot-Pi0`](https://github.com/allenai/MolmoBot/tree/main/MolmoBot-Pi0)) on the vendored `openpi` Pi0.5 PyTorch backbone.

Pass node/GPU/partition on the `sbatch` line; dataset/checkpoint paths come from env (each launcher's header documents its vars). Paths below are relative to the repo root. `<aspect>` ∈ `baseline | physical_motion | scene_composition | arm_pose | reasoning | mix5`.

| Stage | Suite | Launcher |
|---|---|---|
| Train DeMiAn-VLA | MolmoBot/MolmoSpaces | `scripts/slurm/train_demian_vla_molmobot.slurm <aspect>` |
| Train DeMiAn-VLA | RoboCasa365 | `scripts/slurm/train_demian_vla_robocasa.slurm <aspect>` |
| Eval VLA (RoboCasa) | RoboCasa | `scripts/slurm/eval_demian_vla_robocasa.slurm <aspect>` |
| Eval VLA (RoboCasa, instructor) | RoboCasa | `scripts/slurm/eval_demian_vla_robocasa_async.slurm <async\|sync>` |
| Eval VLA (composite tasks) | RoboCasa | `scripts/slurm/eval_demian_vla_composite.slurm <fix\|dynamic\|dynamic-gtcap\|dynamic-instructor>` |
| Eval VLA (MolmoSpaces, instructor) | MolmoSpaces | `scripts/slurm/eval_demian_vla_molmospaces_async.slurm <async\|sync>` |
