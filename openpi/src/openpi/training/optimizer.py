"""Optimizer / LR-schedule configuration dataclasses.

NOTE (DeMiAn release): the original openpi implements the actual schedules and
optimizers with optax (JAX). The PyTorch trainer (`scripts/train_pytorch.py`)
only reads the *fields* of these dataclasses (warmup_steps, peak_lr, decay_steps,
decay_lr, b1/b2/eps/weight_decay) and reconstructs the cosine schedule + AdamW in
PyTorch. The optax `.create()` methods and `create_optimizer()` have therefore been
removed so this module carries no JAX dependency. The dataclasses are kept so the
config registry (`openpi.training.config`) and the JAX-era type annotations still
resolve.
"""

import dataclasses
from typing import Protocol, runtime_checkable


@runtime_checkable
class LRScheduleConfig(Protocol):
    warmup_steps: int
    peak_lr: float


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000


@runtime_checkable
class OptimizerConfig(Protocol):
    weight_decay: float


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0
