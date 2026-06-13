import dataclasses
import pathlib
from typing import Literal
import json
import logging

from etils import epath
import torch
from typing_extensions import override
import numpy as np
import jax

from openpi.training.data_loader import TorchDataLoader, DataLoaderImpl, DataLoader, transform_dataset, Dataset
from openpi.policies.droid_policy import DroidInputs, DroidOutputs
from openpi.training.config import AssetsConfig, DataConfig, DataConfigFactory, ModelTransformFactory
import openpi.training.config as _config
import openpi.models.model as _model
import openpi.transforms as _transforms

from molmobot_pi0.dataset import MlSpacesDataset
from molmobot_pi0.utils import DistributedWeightedDatasetSampler


class NumpifyTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # Use dlpack to avoid broken torch↔numpy bridge in NGC containers with numpy 2.x
                data[k] = np.from_dlpack(v.cpu().detach().contiguous())
            elif isinstance(v, jax.Array):
                data[k] = np.array(v)
        return data


class UnchunkObservationTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        for k, v in data.items():
            if k.startswith("observation"):
                assert v.shape[0] == 1, "Observation horizon must be 1"
                data[k] = v[0]
        return data


class ProprioSplitTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        state = data.pop("observation.state")
        joint_positions = state[:7]
        gripper_positions = state[7:]
        data["observation/joint_position"] = joint_positions
        data["observation/gripper_position"] = gripper_positions
        return data


class GripperStateTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        grip = data["observation/gripper_position"]
        data["observation/gripper_position"] = np.clip(grip[..., :1] / 0.824033, 0, 1)
        return data


class GripperActionTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" in data:
            actions = data["actions"]
            assert actions.shape[-1] == 8
            actions[..., 7] = actions[..., 7] / 255.0
        return data


class JointDeltaToVelTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" in data:
            actions = data["actions"]
            assert actions.shape[-1] == 8
            actions[..., :7] = actions[..., :7] / 0.2
        return data


class PreserveRawActionsTransform(_transforms.DataTransformFn):
    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" in data:
            # Preserve MolmoBot action units after robot-specific transforms
            # and before OpenPI z-score / quantile normalization.
            data["actions_raw"] = np.asarray(data["actions"], dtype=np.float32).copy()
        return data


@dataclasses.dataclass(frozen=True)
class MlSpacesDatasetConfigFactory(DataConfigFactory):
    joint_pos_actions: bool = True  # whether to treat actions as joint positions or deltas
    img_size: tuple[int, int] = (360, 640)
    augment_images: bool = True
    wrist_camera: str = "wrist_camera"
    exo_camera: str = "exo_camera_1"
    prompt_templates: dict[str, list[list[str]]] | None = None
    prompt_sampling_prob_threshold: float = 0.15
    prompt_sampling_temperature: float = 4.0
    trim_episode_length: int = 0
    captions_dir: str | None = None
    caption_key: str = "current_action"

    auxiliary_dataset_paths: dict[str, float] = dataclasses.field(default_factory=lambda: {})

    # Override assets to use a relative asset_id instead of the absolute repo_id path
    assets: AssetsConfig = dataclasses.field(default_factory=lambda: AssetsConfig(asset_id="molmobot"))

    def __setstate__(self, state: dict) -> None:
        # Handle schema evolution: add defaults for any new fields not in the pickled state
        for field in dataclasses.fields(self):
            if field.name not in state:
                if field.default is not dataclasses.MISSING:
                    state[field.name] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    state[field.name] = field.default_factory()
        self.__dict__.update(state)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                NumpifyTransform(),
                UnchunkObservationTransform(),
                ProprioSplitTransform(),  # split state into arm and gripper
                _transforms.RepackTransform({
                    f"observation/exterior_image_1_left": f"observation.image.{self.exo_camera}",
                    f"observation/wrist_image_left": f"observation.image.{self.wrist_camera}",
                    "actions": "action",
                    "prompt": "task",
                    "observation/joint_position": "observation/joint_position",
                    "observation/gripper_position": "observation/gripper_position",
                }),
            ]
        )
        action_specific_trfs = [JointDeltaToVelTransform()] if not self.joint_pos_actions else []
        data_transform = _transforms.Group(
            inputs=[
                GripperStateTransform(),  # rescale gripper state
                GripperActionTransform(),  # rescale gripper action
                *action_specific_trfs,
                DroidInputs(model_config.model_type),
                PreserveRawActionsTransform(),
            ],
            outputs=[
                DroidOutputs(),
            ]
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transform,
            model_transforms=model_transforms,
        )

    @override
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        # if outputting joint vel, use droid stats
        if not self.joint_pos_actions:
            return super()._load_norm_stats(assets_dir, asset_id)

        stats_file = f"{self.repo_id}/aggregated_stats.json"
        if not epath.Path(stats_file).exists():
            return None

        with open(stats_file, "r") as f:
            aggregated_stats = json.load(f)

        norm_stats = {
            "actions": _transforms.NormStats(
                mean=np.concatenate([aggregated_stats["actions/joint_pos/arm"]["mean"], [0.5]]),
                std=np.concatenate([aggregated_stats["actions/joint_pos/arm"]["std"], [0.5]]),
                q01=np.concatenate([aggregated_stats["actions/joint_pos/arm"]["min"], [0.0]]),
                q99=np.concatenate([aggregated_stats["actions/joint_pos/arm"]["max"], [1.0]]),
            ),
            "state": _transforms.NormStats(
                mean=np.concatenate([aggregated_stats["obs/agent/qpos/arm"]["mean"], [0.5]]),
                std=np.concatenate([aggregated_stats["obs/agent/qpos/arm"]["std"], [0.5]]),
                q01=np.concatenate([aggregated_stats["obs/agent/qpos/arm"]["min"], [0.0]]),
                q99=np.concatenate([aggregated_stats["obs/agent/qpos/arm"]["max"], [1.0]]),
            ),
        }
        return norm_stats


@dataclasses.dataclass(frozen=True)
class ConcatDatasetInfo:
    dataset_sizes: list[int]
    dataset_weights: list[float]


def create_dataset(
    data_config: DataConfig, data_config_factory: DataConfigFactory, action_horizon: int, model_config: _model.BaseModelConfig
) -> tuple[Dataset, int, ConcatDatasetInfo | None]:
    """Returns (dataset, epoch_size, concat_dataset_info).

    When there are no auxiliary datasets, concat_dataset_info is None.
    """
    assert isinstance(data_config_factory, MlSpacesDatasetConfigFactory)
    main_dataset = create_dataset_from_path(data_config.repo_id, data_config_factory, action_horizon, model_config)

    if not data_config_factory.auxiliary_dataset_paths:
        return main_dataset, len(main_dataset), None

    datasets = [main_dataset]
    main_dataset_weight = 1.0 - sum(data_config_factory.auxiliary_dataset_paths.values())
    assert 0 < main_dataset_weight < 1, "Main dataset weight must be in (0, 1)"
    dataset_weights = [main_dataset_weight]

    for dataset_path, weight in data_config_factory.auxiliary_dataset_paths.items():
        dataset = create_dataset_from_path(dataset_path, data_config_factory, action_horizon, model_config)
        datasets.append(dataset)
        dataset_weights.append(weight)
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    dataset_sizes = [len(ds) for ds in datasets]

    # we define an epoch as the number of samples after which we expect to see the whole main dataset
    samples_per_epoch = int(np.ceil(len(main_dataset) / main_dataset_weight))
    info = ConcatDatasetInfo(dataset_sizes=dataset_sizes, dataset_weights=dataset_weights)
    return concat_dataset, samples_per_epoch, info


def create_dataset_from_path(dataset_path: str, data_config_factory: DataConfigFactory, action_horizon: int, model_config: _model.BaseModelConfig) -> Dataset:
    assert isinstance(data_config_factory, MlSpacesDatasetConfigFactory)
    if data_config_factory.joint_pos_actions:
        selected_actions = ["joint_pos.arm", "joint_pos.gripper"]
    else:
        selected_actions = ["joint_pos_rel.arm", "joint_pos.gripper"]
    randomize_prompts = data_config_factory.prompt_templates is not None

    # Resolve per-config captions subdirectory from the dataset path
    # e.g., captions_dir="/path/captions", dataset_path=".../FrankaPickOmniCamConfig/part0/train"
    # -> resolved_captions_dir="/path/captions/FrankaPickOmniCamConfig"
    resolved_captions_dir = None
    if data_config_factory.captions_dir is not None:
        dataset_path_obj = pathlib.Path(dataset_path)
        for parent in [dataset_path_obj] + list(dataset_path_obj.parents):
            if parent.name.endswith("Config"):
                config_name = parent.name
                candidate = pathlib.Path(data_config_factory.captions_dir) / config_name
                if candidate.exists():
                    resolved_captions_dir = candidate
                break

    dataset = MlSpacesDataset(
        data_root=pathlib.Path(dataset_path),
        house_idxs=None,
        selected_states=["qpos.arm", "qpos.gripper"],
        selected_actions=selected_actions,
        selected_observations=[data_config_factory.wrist_camera, data_config_factory.exo_camera],
        selected_env_states=None,
        img_size=data_config_factory.img_size,
        action_chunking=True,
        obs_horizon=1,
        action_horizon=action_horizon,
        drop_n_last_frames=action_horizon - 1 + data_config_factory.trim_episode_length,
        batch_image_proc=False,
        augment_images=data_config_factory.augment_images,
        randomize_prompts=randomize_prompts,
        prompt_templates=data_config_factory.prompt_templates,
        prompt_sampling_prob_threshold=data_config_factory.prompt_sampling_prob_threshold,
        prompt_sampling_temperature=data_config_factory.prompt_sampling_temperature,
        prompt_sampling_randomize_casing=randomize_prompts,
        prompt_sampling_randomize_punctuation=randomize_prompts,
        captions_dir=resolved_captions_dir,
        caption_key=data_config_factory.caption_key,
    )
    return dataset


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    model_config = config.model
    action_horizon = config.model.action_horizon
    batch_size = config.batch_size
    seed = config.seed
    num_workers = config.num_workers

    dataset, epoch_size, concat_dataset_info = create_dataset(data_config, config.data, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if concat_dataset_info is None:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            else:
                sampler = DistributedWeightedDatasetSampler(
                    concat_dataset_info.dataset_sizes,
                    concat_dataset_info.dataset_weights,
                    epoch_size,
                )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
            if concat_dataset_info is not None:
                sampler = DistributedWeightedDatasetSampler(
                    concat_dataset_info.dataset_sizes,
                    concat_dataset_info.dataset_weights,
                    epoch_size,
                    num_replicas=1,
                    rank=0,
                )
    else:
        local_batch_size = batch_size // jax.process_count()
        assert concat_dataset_info is None, "Weighted sampling is not supported for JAX"

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)
