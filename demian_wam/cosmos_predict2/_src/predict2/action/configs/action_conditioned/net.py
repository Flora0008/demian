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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
    ActionConditionedMinimalV1LVGDiT,
)
from cosmos_predict2._src.predict2.action.networks.joint_action_video_dit import JointActionVideoDiT

# from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

COSMOS_V1_7B_NET_MININET: LazyDict = L(ActionConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_2B_NET_MININET.model_channels = 2048
COSMOS_V1_2B_NET_MININET.num_heads = 16
COSMOS_V1_2B_NET_MININET.num_blocks = 28
COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_14B_NET_MININET.model_channels = 5120
COSMOS_V1_14B_NET_MININET.num_heads = 40
COSMOS_V1_14B_NET_MININET.num_blocks = 36
COSMOS_V1_14B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET.rope_t_extrapolation_ratio = 1.0

mini_net = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0


COSMOS_V1_7B_NET_MININET_ACTION_CHUNK: LazyDict = L(ActionChunkConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK = copy.deepcopy(COSMOS_V1_7B_NET_MININET_ACTION_CHUNK)
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.model_channels = 2048
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.num_heads = 16
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.num_blocks = 28
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET_ACTION_CHUNK = copy.deepcopy(COSMOS_V1_7B_NET_MININET_ACTION_CHUNK)
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.model_channels = 5120
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.num_heads = 40
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.num_blocks = 36
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.rope_t_extrapolation_ratio = 1.0


# Joint action-video WAM network (based on 2B backbone)
COSMOS_V1_2B_NET_MININET_WAM: LazyDict = L(JointActionVideoDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
    # WAM-specific parameters
    state_dim=16,
    action_dim_for_denoising=7,
    num_action_tokens_per_latent_frame=4,
    zero_init_action_decoder=True,
    use_flex_attention=False,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_2B_action_conditioned", node=COSMOS_V1_2B_NET_MININET
    )
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_7B_action_conditioned", node=COSMOS_V1_7B_NET_MININET
    )
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_14B_action_conditioned", node=COSMOS_V1_14B_NET_MININET
    )

    # action chunk conditioned
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_chunk_conditioned",
        node=COSMOS_V1_2B_NET_MININET_ACTION_CHUNK,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_7B_action_chunk_conditioned",
        node=COSMOS_V1_7B_NET_MININET_ACTION_CHUNK,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_14B_action_chunk_conditioned",
        node=COSMOS_V1_14B_NET_MININET_ACTION_CHUNK,
    )

    # Joint action-video WAM network (2B)
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_joint_action_video",
        node=COSMOS_V1_2B_NET_MININET_WAM,
    )

    # Base 2B DiT for action head model (no action tokens in transformer)
    # Uses ActionConditionedMinimalV1LVGDiT which adds +1 channel for conditioning mask
    # Configured for MEAN_POOLING text encoding (3584D) matching mid-training checkpoint
    _base_2b = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    _base_2b.use_crossattn_projection = True
    _base_2b.crossattn_proj_in_channels = 3584  # MEAN_POOLING: avg of 28 Qwen layers → 3584D
    _base_2b.crossattn_emb_channels = 1024
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_base",
        node=_base_2b,
    )
