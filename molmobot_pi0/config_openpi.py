import openpi.training.config as _config
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer

from molmobot_pi0.dataset_openpi import MlSpacesDatasetConfigFactory
from molmobot_pi0.prompt_templates import DEFAULT_PROMPT_TEMPLATES


for _train_cfg in [
    _config.TrainConfig(
        name="pi05_mlspaces_caption_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=MlSpacesDatasetConfigFactory(
            joint_pos_actions=False,
            prompt_templates=None,
            captions_dir=None,  # override via CLI: --data.captions-dir <path>
            caption_key="current_action",
            assets=_config.AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        pytorch_weight_path=None,
        weight_loader=_config.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=2_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_000,
        batch_size=1024,
        log_interval=100,
        save_interval=2000,
        keep_period=10_000,
        num_workers=48,
    ),
    # Variant of pi05_mlspaces_caption_finetune with lm_loss_weight=0.1 enabled
    # (matches openpi reference caption configs e.g. pi05_robocasa_v01_cap).
    # Adds an autoregressive LM loss on prefix tokens including the appended caption.
    _config.TrainConfig(
        name="pi05_mlspaces_caption_finetune_lm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            lm_loss_weight=0.1,
        ),
        data=MlSpacesDatasetConfigFactory(
            joint_pos_actions=False,
            prompt_templates=None,
            captions_dir=None,  # override via CLI: --data.captions-dir <path>
            caption_key="current_action",
            assets=_config.AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        pytorch_weight_path=None,
        weight_loader=_config.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=2_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_000,
        batch_size=1024,
        log_interval=100,
        save_interval=2000,
        keep_period=10_000,
        num_workers=48,
    ),
    _config.TrainConfig(
        name="molmobot_pi0_droid",
        model=pi0_config.Pi0Config(
            action_dim=8,
            action_horizon=16,
        ),
        data=MlSpacesDatasetConfigFactory(
            # must override repo_id
            joint_pos_actions=True,
            prompt_templates=DEFAULT_PROMPT_TEMPLATES,
            trim_episode_length=10,
        ),
        pytorch_weight_path=None,
        weight_loader=_config.weight_loaders.PaliGemmaWeightLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=300_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_001,
        batch_size=256,
        log_interval=100,
        save_interval=1000,
        keep_period=10_000,
        num_workers=48,
        freeze_filter="vision_tower",
    ),
    _config.TrainConfig(
        name="pi05_mlspaces_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=MlSpacesDatasetConfigFactory(
            # must override repo_id
            joint_pos_actions=False,
            prompt_templates=None,
            assets=_config.AssetsConfig(
                # reuse droid stats
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        pytorch_weight_path=None,
        weight_loader=_config.weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=2_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_000,
        batch_size=256,
        log_interval=100,
        save_interval=1000,
        keep_period=10_000,
        num_workers=48,
    ),
]:
    cfg_name = _train_cfg.name
    assert cfg_name not in _config._CONFIGS_DICT, f"Train config {cfg_name} already exists"
    _config._CONFIGS_DICT[cfg_name] = _train_cfg
