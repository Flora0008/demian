# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

from groot_dreams.dataloader import MultiVideoActionDataset

# bridge dataset path
base_path = "datasets/bridge/"

train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


# experiment for next-frame prediction
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)
bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)

# experiment for action-sequence video prediction
bridge_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)

dreamdojo_13frame_480_640_train_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=["datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"],
)
dreamdojo_13frame_480_640_val_dataset = L(MultiVideoActionDataset)(
    num_frames=13,
    dataset_path=["datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"],
)


# create dataloader for each dataset
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def build_webdataset(webdataset_instance, **kwargs):
    """Helper function to build WebDataset from a WebDataset instance.

    WebDatasets need to call build_dataset() to get the actual iterable dataset
    that can be used with DataLoader.

    Args:
        webdataset_instance: An instantiated WebDataset object.
        **kwargs: Additional parameters to override on the webdataset instance
            before building. This allows experiment configs to override parameters
            like gripper_rescale_factor, num_action_per_chunk, etc.
    """
    # Apply any parameter overrides to the webdataset instance
    for key, value in kwargs.items():
        if hasattr(webdataset_instance, key):
            setattr(webdataset_instance, key, value)
    return webdataset_instance.build_dataset()


def _worker_init_fn(worker_id):
    """Seed Python random and numpy per worker to avoid duplicate data."""
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def get_dataloader_with_sampler(dataset, batch_size=1, drop_last=True,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True, prefetch_factor=4,
                                **kwargs):
    """
    Create a DataLoader with a DistributedSampler, ensuring the dataset is only instantiated once.

    This function takes the dataset (which will already be instantiated by the LazyCall system)
    and creates both the sampler and DataLoader together, avoiding double instantiation that would
    occur if dataset is passed separately to both DataLoader and get_sampler.
    """
    # Create sampler with the instantiated dataset
    sampler = get_sampler(dataset)

    # persistent_workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None  # not supported with num_workers=0

    # Create DataLoader with the instantiated dataset and sampler
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        **kwargs,
    )


bridge_train_dataloader = L(get_dataloader_with_sampler)(
    dataset=bridge_train_dataset,
    batch_size=1,
    drop_last=True,
)
bridge_val_dataloader = L(get_dataloader_with_sampler)(
    dataset=bridge_val_dataset,
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dataloader = L(get_dataloader_with_sampler)(
    dataset=bridge_13frame_480_640_train_dataset,
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dataloader = L(get_dataloader_with_sampler)(
    dataset=bridge_13frame_480_640_val_dataset,
    batch_size=1,
    drop_last=True,
)

dreamdojo_13frame_480_640_train_dataloader = L(get_dataloader_with_sampler)(
    dataset=dreamdojo_13frame_480_640_train_dataset,
    batch_size=1,
    drop_last=True,
)
dreamdojo_13frame_480_640_val_dataloader = L(get_dataloader_with_sampler)(
    dataset=dreamdojo_13frame_480_640_val_dataset,
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )

    # 13 frame 480 640
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_13frame_480_640_train",
        node=bridge_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_13frame_480_640_val",
        node=bridge_13frame_480_640_val_dataloader,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="dreamdojo_13frame_480_640_train",
        node=dreamdojo_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="dreamdojo_13frame_480_640_val",
        node=dreamdojo_13frame_480_640_val_dataloader,
    )
