# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma variant configuration registry.

NOTE (DeMiAn release): the original openpi `gemma.py` also contains the full JAX
(flax.linen) Gemma implementation. The PyTorch model
(`openpi.models_pytorch.gemma_pytorch`) only needs the `Config` dataclass and
`get_config(variant)` registry below (it builds its backbone from HuggingFace
PaliGemma/Gemma using these width/depth/head dims), so the JAX module body has
been removed to keep this file JAX-free. The `lora_configs` field is retained for
API compatibility but is unused by the PyTorch path (which full-fine-tunes).
"""

import dataclasses
from typing import Literal


@dataclasses.dataclass
class LoRAConfig:
    """Minimal LoRA config (rank/alpha) kept for `Config` compatibility.

    The PyTorch backbone does not read this; LoRA fine-tuning in this release is
    handled separately (HF PEFT for the instructor). Original JAX LoRA fields
    beyond rank/alpha/rslora are omitted.
    """

    rank: int
    alpha: float = 1.0
    rslora: bool = False


PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": LoRAConfig(rank=16, alpha=16.0), "ffn": LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": LoRAConfig(rank=32, alpha=32.0), "ffn": LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")
