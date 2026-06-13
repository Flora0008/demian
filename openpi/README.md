# DeMiAn-VLA

This directory is based on the `openpi` [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi), `pyproject` version `0.1.0`. It provides the model + training pipeline used by DeMiAn-VLA post-training.

## Layout / PYTHONPATH

```
openpi/
├── src/openpi/                  # the `openpi` package
├── packages/openpi-client/src/  # `openpi_client` (pure-numpy dep of transforms)
└── scripts/train_pytorch.py     # PyTorch (DDP) training entrypoint
```

Run with:

```bash
export PYTHONPATH=openpi/src:openpi/packages/openpi-client/src:.   # '.' = repo root (molmobot_pi0)
python openpi/scripts/train_pytorch.py <config_name> --exp_name <run>
```

DeMiAn training configs (e.g. `pi05_mlspaces_caption_finetune`) are registered into `openpi.training.config._CONFIGS_DICT` by `molmobot_pi0/config_openpi.py`.


## RoboCasa eval

RoboCasa eval serves a trained policy over a websocket and runs the sim harness against it:
- `src/openpi/serving/websocket_policy_server.py` + `scripts/serve_policy.py` — the policy server.
- `scripts/eval/run_robocasa_eval.py` (+ `robocasa_task_list.py`, `composite_subgoals.py`, `aggregate_robocasa_quick_shards.py`) — the RoboCasa sim harness (uses `openpi_client.WebsocketClientPolicy`; needs the external robocasa/robosuite sim).

Launchers (all in `scripts/slurm/`):
- `eval_demian_vla_robocasa.slurm <aspect>` — GT-caption injection (`--gt-reasoning-file` + `--caption-key`: `current_action` / `environment-simple` / `agent_arm_start_state` / `reasoning-short`; baseline injects none).
- `eval_demian_vla_robocasa_async.slurm [async|sync]` — the learned instructor (paper RQ2 / async-vs-sync): also starts `scripts/eval/qwen_instructor_server.py` (the Qwen3.5-2B instructor) and runs the harness with `--instructor-type qwen --instructor-model-path http://localhost:<port>`; `sync` adds `--sync-instructor`.
- `eval_demian_vla_composite.slurm <fix|dynamic|dynamic-gtcap|dynamic-instructor>`
  — composite-task eval (paper RQ3) over the 18 composite tasks (those with subgoal success-checkers in `composite_subgoals.py`): `--split composite` with subgoal-driven prompt swapping and, for `dynamic-instructor`, the live instructor.

The **MolmoSpaces** side supports the same async instructor: `qwen_instructor_server.py` also drives `molmobot_pi0`'s `PiPolicy` (set `policy.instructor_url=http://localhost:<port>`, `policy.instructor_sync` for the sync baseline) — PiPolicy snapshots the initial exo+wrist frames, fires a non-blocking annotation request, and splices it at the next action-chunk boundary. Launcher: `scripts/slurm/eval_demian_vla_molmospaces_async.slurm [async|sync]`. (The instructor needs qwen3_5-capable `transformers`, newer than the pi05 policy's 4.53.2 — see the launcher's `INSTRUCTOR_PY` note.) The server accepts a variable image count (3 for RoboCasa, 2 for MolmoBot/MolmoSpaces).

Set `ACTION_MODEL`, `POLICY_CONFIG`, `INSTRUCTOR_MODEL` (async/instructor), `ROBOCASA_PATH`, `ROBOSUITE_PATH` via env.

**GT caption files.** MolmoSpaces per-bench GT caption JSONs are *generated* from the released `data/molmobot-captions/*.jsonl` by the `scripts/generate_gt_*.py` tools. RoboCasa **atomic** GT captions are built by `scripts/merge_fg_captions_for_eval.py` from the RoboCasa dataset's per-task `captions_*.jsonl` (ship with the external robocasa dataset). RoboCasa **composite** GT captions are shipped directly: `scripts/eval/gt_captions_composite.json`.
