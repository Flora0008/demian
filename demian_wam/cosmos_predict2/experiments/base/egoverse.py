# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EgoVerse mid-training experiments for Cosmos Predict 2.5 (Video2World).

Two caption variants are registered as separate Hydra experiments:

- ``predict2_video2world_training_2b_egoverse_annotation`` —
  baseline, the text input to the encoder is the EgoVerse ``annotation``.
- ``predict2_video2world_training_2b_egoverse_augmented`` —
  the text input is ``annotation. current_action environment
  hand_start_state hand_end_state`` (space-joined, empty fields skipped,
  trailing period appended to annotation only if not already punctuated).

Both use the same model/trainer knobs as the RoboCasa experiments so training
curves are directly comparable.
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_egoverse import (
    EgoVerseVideoDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_egoverse_latent import (
    EgoVerseLatentDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_egoverse_latent_random import (
    EgoVerseLatentRandomCaptionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

import os

# EgoVerse dataset locations — set via env (the training launcher exports these).
EGOVERSE_DATASET_DIR = os.environ.get("EGOVERSE_DATASET_DIR", "egoverse_dataset")

# Precomputed (VAE latent + Cosmos Reason 1 text embedding) shards, one dir per
# caption variant.
EGOVERSE_LATENT_DIR_ANNOTATION = os.environ.get("EGOVERSE_LATENT_DIR_ANNOTATION", "egoverse_cosmos_stage1_annotation")
EGOVERSE_LATENT_DIR_AUGMENTED = os.environ.get("EGOVERSE_LATENT_DIR_AUGMENTED", "egoverse_cosmos_stage1")


def _make_dataloader(variant: str, batch_size: int = 1):
    dataset = L(EgoVerseVideoDataset)(
        dataset_dir=EGOVERSE_DATASET_DIR,
        num_frames=93,
        video_size=(480, 640),
        variant=variant,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=batch_size,
        drop_last=True,
        # EgoVerse frames live as per-JPEG entries in a Zarr store, so each
        # sample forces 93 JPEG decodes on CPU. With bs=8 per GPU and only
        # 4 workers that saturates the CPU and starves the H100s (~27s/iter
        # data-bound vs ~7s expected). Widen the pool and keep workers warm
        # across epochs.
        num_workers=16,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
    )


dataloader_train_egoverse_annotation = _make_dataloader("annotation")
dataloader_train_egoverse_augmented = _make_dataloader("annotation_plus_dense")
# Single dense field: caption = "{annotation}. {captions['dense']}"
dataloader_train_egoverse_dense_only = _make_dataloader("annotation_plus_dense_only")
# Single human-perspective dense field: caption = "{annotation}. {captions['dense_human']}"
dataloader_train_egoverse_dense_human = _make_dataloader("annotation_plus_dense_human")


def _make_latent_dataloader(dataset_dir: str, batch_size: int = 1):
    """Dataloader that feeds precomputed VAE latents + Reason 1 embeddings.

    Enables the fast path in ``text2world_model_rectified_flow.forward`` —
    both the Wan2.1 VAE encode and the Cosmos Reason 1 text encode are
    skipped, relying on the offline-precomputed tensors instead.
    """
    dataset = L(EgoVerseLatentDataset)(dataset_dir=dataset_dir)
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )


dataloader_train_egoverse_annotation_precomputed = _make_latent_dataloader(EGOVERSE_LATENT_DIR_ANNOTATION)
dataloader_train_egoverse_augmented_precomputed = _make_latent_dataloader(EGOVERSE_LATENT_DIR_AUGMENTED)


# NRT resume path: keep the stage2 experiment names/checkpoint directories,
# but feed existing raw EgoVerse data instead of the missing stage2 precompute.
dataloader_train_egoverse_annotation_stage2 = _make_dataloader("annotation")
dataloader_train_egoverse_dense_human_stage2 = _make_dataloader("annotation_plus_dense_human")


def _make_random_dense_dataloader(batch_size: int = 1):
    """Dataloader that feeds precomputed latents + random dense caption per step.

    Latent comes from upstream's shards (same VAE output regardless of text).
    Caption is `annotation + " " + <current_action|environment|reasoning>`,
    with the dense field drawn uniformly at random per ``__getitem__`` call.
    Online text encoder still runs each iter.
    """
    dataset = L(EgoVerseLatentRandomCaptionDataset)(
        latent_dataset_dir=EGOVERSE_LATENT_DIR_ANNOTATION,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )


dataloader_train_egoverse_random_dense_precomputed = _make_random_dense_dataloader()


def _make_experiment(dataloader, name: str) -> dict:
    """Shared experiment config for EgoVerse caption variants."""
    return dict(
        defaults=[
            f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
            {"override /data_train": "mock"},
            {"override /data_val": "mock"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict_v2p5",
            group="video2world",
            name=name,
        ),
        dataloader_train=dataloader,
        checkpoint=dict(
            save_iter=1000,
            load_path=DEFAULT_CHECKPOINT.s3.uri,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.001,
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        trainer=dict(
            logging_iter=100,
            max_iter=100_000,
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=200, save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(every_n=1000, save_s3=False),
                every_n_sample_ema=dict(every_n=1000, save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
    )


predict2_video2world_training_2b_egoverse_annotation = _make_experiment(
    dataloader_train_egoverse_annotation, "2b_egoverse_annotation"
)

predict2_video2world_training_2b_egoverse_augmented = _make_experiment(
    dataloader_train_egoverse_augmented, "2b_egoverse_augmented"
)

predict2_video2world_training_2b_egoverse_dense_only = _make_experiment(
    dataloader_train_egoverse_dense_only, "2b_egoverse_dense_only"
)

predict2_video2world_training_2b_egoverse_dense_human = _make_experiment(
    dataloader_train_egoverse_dense_human, "2b_egoverse_dense_human"
)

# Precomputed-latent variants: same model/optimizer/trainer config as their
# raw-video counterparts, just a different dataloader. Names deliberately
# shadow the raw variants so the job.name checkpoints land in separate dirs.
predict2_video2world_training_2b_egoverse_annotation_precomputed = _make_experiment(
    dataloader_train_egoverse_annotation_precomputed, "2b_egoverse_annotation_precomputed"
)

predict2_video2world_training_2b_egoverse_augmented_precomputed = _make_experiment(
    dataloader_train_egoverse_augmented_precomputed, "2b_egoverse_augmented_precomputed"
)

predict2_video2world_training_2b_egoverse_random_dense_precomputed = _make_experiment(
    dataloader_train_egoverse_random_dense_precomputed, "2b_egoverse_random_dense_precomputed"
)

# Stage-2 resume variants using raw NRT EgoVerse data.
predict2_video2world_training_2b_egoverse_annotation_stage2 = _make_experiment(
    dataloader_train_egoverse_annotation_stage2, "2b_egoverse_annotation_stage2"
)
predict2_video2world_training_2b_egoverse_dense_human_stage2 = _make_experiment(
    dataloader_train_egoverse_dense_human_stage2, "2b_egoverse_dense_human_stage2"
)


cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_egoverse_annotation,
    predict2_video2world_training_2b_egoverse_augmented,
    predict2_video2world_training_2b_egoverse_dense_only,
    predict2_video2world_training_2b_egoverse_dense_human,
    predict2_video2world_training_2b_egoverse_annotation_precomputed,
    predict2_video2world_training_2b_egoverse_augmented_precomputed,
    predict2_video2world_training_2b_egoverse_random_dense_precomputed,
    predict2_video2world_training_2b_egoverse_annotation_stage2,
    predict2_video2world_training_2b_egoverse_dense_human_stage2,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
