# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ActionHeadVideoModel: Frozen DiT backbone with lightweight action regression head.

Instead of jointly denoising video + action tokens (WAM), this model:
1. Runs standard video flow matching on the DiT (frozen backbone, AdaLN-LoRA trainable)
2. Extracts hidden states from the last DiT block via forward hook
3. Predicts actions via a cross-attention ActionHead (deterministic regression, no flow matching)

This avoids the gradient starvation problem that plagued WAM (action decoder weight norm
barely changed over 30K steps because the full 2B backbone absorbed all gradient signal).
"""

from typing import Callable, Dict, Optional, Tuple

import attrs
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.text2world_model import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
    NUM_CONDITIONAL_FRAMES_KEY,
)
from cosmos_predict2._src.predict2.action.networks.action_head import ActionHead
from cosmos_predict2._src.predict2.action.networks.action_expert import ActionExpert
from cosmos_predict2._src.predict2.action.networks.flow_action_expert import (
    FlowActionExpert,
    sample_beta,
)
from cosmos_predict2._src.predict2.action.networks.joint_flow_action_expert import (
    JointFlowActionExpert,
)
from cosmos_predict2._src.predict2.action.networks.joint_action_expert import (
    JointActionExpert,
)
from cosmos_predict2._src.predict2.action.networks.vlm_action_expert import VlmActionExpert

log = logging.getLogger(__name__)


class ProprioConditioner(nn.Module):
    """Project low-dimensional robot state into one cross-attention token."""

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Gate starts at zero so old checkpoints begin with a near-no-op token.
        # The extra attention slot still changes the softmax denominator slightly.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        assert state.dim() == 2 and state.shape[-1] == self.state_dim, (
            f"state must be (B, {self.state_dim}); got {tuple(state.shape)}"
        )
        return torch.tanh(self.gate) * self.proj(state)


@attrs.define(slots=False)
class ActionHeadVideoModelConfig(Video2WorldModelRectifiedFlowConfig):
    """Configuration for action head video model."""
    action_dim: int = 7
    num_action_steps: int = 12
    action_loss_weight: float = 1.0
    action_head_lr: float = 5e-5
    action_head_warmup_steps: int = 0
    action_head_lr_start_factor: float = 1e-6
    action_head_num_heads: int = 8
    action_head_mlp_ratio: int = 4

    # Action-expert (multi-layer transformer) alternative to the single-layer
    # ActionHead. When True, set_up_model builds an ActionExpert that taps K
    # different DiT blocks via per-layer cross-attention plus a shared text
    # cross-attention. Mirrors the action-expert pattern used in pi0 / GR00T-N1.
    use_action_expert: bool = False
    # Number of action-expert layers (K). Each layer cross-attends to one DiT tap.
    action_expert_num_layers: int = 4
    # FFN expansion ratio inside each expert layer (transformer convention:
    # FFN hidden = mlp_ratio * hidden_dim). Note: opposite of action_head_mlp_ratio
    # which compresses; here we expand.
    action_expert_mlp_ratio: int = 2
    # Indices of DiT blocks to tap for the K expert layers (must have len = K).
    # Default for the 28-block 2B DiT: evenly spread, ending at the last block.
    action_expert_taps: tuple = (6, 13, 20, 27)

    # Offset into the multi-embodiment 384-D action vector where this embodiment's
    # action lives. RoboCasa=169, GR1=0, etc. Only used when the dataset returns
    # a 384-D vector (i.e. shape[-1] > action_dim).
    gt_action_offset: int = 169

    # Flow-matching action expert (pi0/pi05-style): same multi-tap cross-attention
    # structure as the deterministic ActionExpert, but takes (noisy_action, t) and
    # predicts velocity. Trained with rectified-flow MSE on velocity, Beta(1.5,1.0)
    # time sampling, plain MSE (no time weighting). Mutually exclusive with
    # use_action_expert.
    use_flow_action_expert: bool = False
    # Number of Euler steps for the action ODE at inference time. pi0 default=10.
    flow_inference_num_steps: int = 10

    # Joint-attention variant of FlowActionExpert (pi05-style joint blockwise
    # attention). Each layer runs ONE softmax over [video_kv; text_kv; action_q]
    # rather than three separate cross/self attentions, letting the model trade
    # off attention budget across modalities within a single normalization.
    # Same forward signature and per-layer DiT taps as FlowActionExpert.
    #
    # Default True: this is the architecture that escaped the
    # MSE-flow mode collapse the other variants hit (FlowActionExpert with
    # 3 separate attentions, ActionExpert deterministic, VlmActionExpert all
    # produced bit-identical action chunks across episodes; joint attention's
    # single softmax forces real attention competition across modalities).
    # YAMLs that want a different head (deterministic single-head ActionHead,
    # ActionExpert, FlowActionExpert, or VlmActionExpert) must explicitly set
    # `use_joint_flow_action_expert: false` to clear this default.
    use_joint_flow_action_expert: bool = True

    # Deterministic version of JointFlowActionExpert. Same per-layer joint
    # cross-attention (K, V over [video; text; action_self]; Q over action only)
    # but predicts the action chunk directly (single forward, MSE loss) instead
    # of velocity (flow ODE). Backup for when flow's small-t velocity field
    # produces gripping-phase failures.
    use_joint_action_expert: bool = False

    # When True, the action head's video cross-attention reads only the FIRST
    # latent frame's tokens (the one grounded in the current observation). The
    # remaining T-1 latent frames carry the DiT's predicted-future video, which
    # under open-loop drift can dilute the head's spatial grounding at the
    # contact phase. Default False preserves the current full-readout behavior.
    # Applies to both ActionExpert and FlowActionExpert paths.
    video_xattn_frame0_only: bool = False

    # VLM-conditioned flow-matching action expert (pi05-style). Replaces the
    # DiT-feature stream with a single cross-attention to Reason1's joint
    # vision-language tokens (image+text). Requires the text encoder to be
    # configured with `image_conditioning=True` so the conditioning frame is
    # fed alongside the caption. Mutually exclusive with the other expert flags.
    use_vlm_action_expert: bool = False

    # When the text encoder has LoRA enabled (use_lora=True on TextEncoderConfig),
    # this LR is used for the dedicated optimizer over Reason1's LoRA adapters.
    # Standard practice is much lower than DiT/head LRs (1e-5 here vs 1e-4 DiT,
    # 5e-5 head) — VLMs are sensitive to large updates.
    text_encoder_lora_lr: float = 1e-5

    # Proprioceptive state conditioning. When enabled, data_batch["state"] is
    # expected to be (B, 1, proprio_dim) or (B, proprio_dim). It is projected into
    # one cross-attention token and can be appended to the DiT context, the
    # action-head context, or both.
    use_proprio_conditioning: bool = False
    proprio_dim: int = 16
    proprio_hidden_dim: int = 256
    proprio_condition_dit: bool = True
    proprio_condition_action_head: bool = True


class ActionHeadVideoModel(ActionVideo2WorldModelRectifiedFlow):
    """
    Video flow matching model with frozen backbone + action regression head.

    The DiT backbone is frozen except for AdaLN-LoRA parameters (~190M trainable).
    A lightweight ActionHead (~10M params) reads the last block's hidden states
    and predicts actions via cross-attention + MLP.
    """

    def set_up_model(self):
        super().set_up_model()

        # The action head consumes DiT-projected text via a forward hook on
        # self.net.crossattn_proj. If projection is disabled, the hook never fires
        # and training will crash deep inside the action regression block. Catch
        # this at model build time with a clear message instead.
        assert getattr(self.net, "use_crossattn_projection", False), (
            "ActionHeadVideoModel requires self.net.use_crossattn_projection=True. "
            "The action head reads the DiT's projected text embeddings; without "
            "projection the head has no text input."
        )

        # Create action head.
        # D = self.net.model_channels (e.g. 2048 for 2B) — video & query dim.
        # T_dim = output dim of self.net.crossattn_proj (e.g. 1024 for 2B) —
        #   we read it from the actual module instead of a missing attribute,
        #   because MiniTrainDIT only stores crossattn_proj_in_channels, not _emb_channels.
        #   Fallback to D if no projection (use_crossattn_projection=False).
        D = self.net.model_channels
        proj = getattr(self.net, "crossattn_proj", None)
        if isinstance(proj, nn.Sequential):
            # crossattn_proj = Sequential(Linear(in, out), GELU); first child is the Linear.
            T_dim = proj[0].out_features
        else:
            T_dim = D

        self.proprio_conditioner = None
        if self.config.use_proprio_conditioning:
            self.proprio_conditioner = ProprioConditioner(
                state_dim=self.config.proprio_dim,
                hidden_dim=self.config.proprio_hidden_dim,
                output_dim=T_dim,
            ).cuda().to(torch.bfloat16)
        self._current_proprio_ctx = None
        self._logged_proprio_ctx = False

        head_flags = [
            self.config.use_action_expert,
            self.config.use_flow_action_expert,
            self.config.use_joint_flow_action_expert,
            self.config.use_vlm_action_expert,
            self.config.use_joint_action_expert,
        ]
        assert sum(head_flags) <= 1, (
            "use_action_expert, use_flow_action_expert, use_joint_flow_action_expert, "
            "use_vlm_action_expert, use_joint_action_expert are mutually exclusive; "
            "set at most one to True."
        )
        if self.config.use_vlm_action_expert:
            # Joint blockwise attention over [DiT video; Reason1 VL; action self].
            # Same K-layer + per-tap structure as JointFlowActionExpert; the
            # difference is the second K/V slot is Reason1's joint vision-language
            # context (image+text) rather than text-only DiT projected text.
            assert getattr(self.config, "text_encoder_config", None) is not None and (
                getattr(self.config.text_encoder_config, "image_conditioning", False)
            ), (
                "use_vlm_action_expert=True requires "
                "text_encoder_config.image_conditioning=True so Reason1 produces "
                "joint vision-language tokens."
            )
            taps = list(self.config.action_expert_taps)
            K = self.config.action_expert_num_layers
            assert len(taps) == K, (
                f"action_expert_taps must have len == action_expert_num_layers "
                f"({len(taps)} vs {K})"
            )
            num_dit_blocks = len(self.net.blocks)
            for t in taps:
                assert 0 <= t < num_dit_blocks or -num_dit_blocks <= t < 0, (
                    f"tap index {t} out of range for DiT with {num_dit_blocks} blocks"
                )
            self._action_expert_taps = taps
            self.action_head = VlmActionExpert(
                hidden_dim=D,
                vlm_dim=T_dim,
                num_actions=self.config.num_action_steps,
                action_dim=self.config.action_dim,
                num_heads=self.config.action_head_num_heads,
                mlp_ratio=self.config.action_expert_mlp_ratio,
                num_layers=K,
            ).cuda().to(torch.bfloat16)
        elif self.config.use_flow_action_expert or self.config.use_joint_flow_action_expert:
            # Flow-matching multi-layer action expert (pi0/pi05 style adapted to
            # cross-attend to a frozen video DiT).
            #
            # Two variants share this branch — both have identical forward
            # signatures, K-layer + per-tap structure, and flow loss recipe;
            # only the per-layer attention shape differs:
            #   FlowActionExpert      — 3 separate attentions per layer
            #                            (self / video-x / text-x).
            #   JointFlowActionExpert — 1 joint softmax per layer over
            #                            [video_kv; text_kv; action_q].
            taps = list(self.config.action_expert_taps)
            K = self.config.action_expert_num_layers
            assert len(taps) == K, (
                f"action_expert_taps must have len == action_expert_num_layers "
                f"({len(taps)} vs {K})"
            )
            num_dit_blocks = len(self.net.blocks)
            for t in taps:
                assert 0 <= t < num_dit_blocks or -num_dit_blocks <= t < 0, (
                    f"tap index {t} out of range for DiT with {num_dit_blocks} blocks"
                )
            self._action_expert_taps = taps
            _expert_cls = (
                JointFlowActionExpert if self.config.use_joint_flow_action_expert
                else FlowActionExpert
            )
            self.action_head = _expert_cls(
                hidden_dim=D,
                text_dim=T_dim,
                num_actions=self.config.num_action_steps,
                action_dim=self.config.action_dim,
                num_heads=self.config.action_head_num_heads,
                mlp_ratio=self.config.action_expert_mlp_ratio,
                num_layers=K,
            ).cuda().to(torch.bfloat16)
        elif self.config.use_action_expert or self.config.use_joint_action_expert:
            # Deterministic multi-layer action expert: K layers, predict action
            # chunk directly (no flow, MSE loss). Two variants share this branch:
            #   ActionExpert       — per layer 3 separate attentions
            #                        (self / video-x / text-x).
            #   JointActionExpert  — per layer 1 joint softmax over
            #                        [video_kv; text_kv; action_self]. Same
            #                        attention shape as JointFlowActionExpert
            #                        minus the time conditioning.
            taps = list(self.config.action_expert_taps)
            K = self.config.action_expert_num_layers
            assert len(taps) == K, (
                f"action_expert_taps must have len == action_expert_num_layers "
                f"({len(taps)} vs {K})"
            )
            num_dit_blocks = len(self.net.blocks)
            for t in taps:
                assert 0 <= t < num_dit_blocks or -num_dit_blocks <= t < 0, (
                    f"tap index {t} out of range for DiT with {num_dit_blocks} blocks"
                )
            self._action_expert_taps = taps
            _det_expert_cls = (
                JointActionExpert if self.config.use_joint_action_expert
                else ActionExpert
            )
            self.action_head = _det_expert_cls(
                hidden_dim=D,
                text_dim=T_dim,
                num_actions=self.config.num_action_steps,
                action_dim=self.config.action_dim,
                num_heads=self.config.action_head_num_heads,
                mlp_ratio=self.config.action_expert_mlp_ratio,
                num_layers=K,
            ).cuda().to(torch.bfloat16)
        else:
            self._action_expert_taps = None
            self.action_head = ActionHead(
                hidden_dim=D,
                text_dim=T_dim,
                num_actions=self.config.num_action_steps,
                action_dim=self.config.action_dim,
                num_heads=self.config.action_head_num_heads,
                mlp_ratio=self.config.action_head_mlp_ratio,
            ).cuda().to(torch.bfloat16)

        # Dedicated optimizer for action head/expert (lives outside FSDP boundary)
        aux_params = list(self.action_head.parameters())
        if self.proprio_conditioner is not None:
            aux_params.extend(self.proprio_conditioner.parameters())
        self._action_head_optim = torch.optim.AdamW(
            aux_params,
            lr=self.config.action_head_lr,
            weight_decay=1e-5,
        )

        # Reason1 LoRA optimizer is created lazily in on_before_zero_grad() —
        # `self.text_encoder` is initialized AFTER set_up_model() in the parent
        # `__init__` (text2world_model_rectified_flow.py:172-177), so we can't
        # construct an optimizer over its LoRA params here. Storing a sentinel
        # for now; the lazy path below makes this transparent.
        self._text_encoder_lora_optim = None
        self._text_encoder_lora_optim_built = False

        # Register forward hook(s) on DiT blocks to capture hidden states.
        # - Single-head mode: one hook on blocks[-1] → self._cached_hidden.
        # - Multi-tap mode (deterministic ActionExpert / FlowActionExpert /
        #   JointFlowActionExpert / VlmActionExpert): K hooks on blocks[taps[k]]
        #   → self._cached_hiddens[k]. VLM expert was previously DiT-blind
        #   (read only the projected VLM context); since the May 2026 joint-attn
        #   refactor it also reads per-layer DiT taps in a 3-source joint
        #   softmax — same hook setup as the other multi-tap experts.
        self._hook_handles = []
        if (
            self.config.use_action_expert
            or self.config.use_flow_action_expert
            or self.config.use_joint_flow_action_expert
            or self.config.use_vlm_action_expert
            or self.config.use_joint_action_expert
        ):
            self._cached_hidden = None  # unused in expert mode
            self._cached_hiddens = [None] * self.config.action_expert_num_layers
            for k, tap in enumerate(self._action_expert_taps):
                handle = self.net.blocks[tap].register_forward_hook(
                    self._make_hidden_hook_for_slot(k)
                )
                self._hook_handles.append(handle)
        else:
            self._cached_hiddens = None  # unused in single-head mode
            self._cached_hidden = None
            handle = self.net.blocks[-1].register_forward_hook(self._capture_hidden)
            self._hook_handles.append(handle)

        # Register forward hook on the DiT's text-embedding projection so the action
        # head can re-use the same projected text the DiT cross-attends to. At training
        # time there's no CFG (one denoise call per forward) → the cached value is the
        # conditional projected text. At inference the cache will hold the *last*
        # writer (uncond under CFG); generate_samples_with_latents_from_batch
        # re-projects explicitly to ensure conditional text is used (see below).
        self._cached_text_ctx = None
        self._text_hook_handle = self.net.crossattn_proj.register_forward_hook(
            self._capture_text_ctx
        )

        # NOTE: We do NOT freeze backbone params because FSDP requires all params
        # to participate in backward. The action head still gets clean gradient
        # signal because it's a separate module outside the FSDP boundary.
        # The backbone gets gradients from video_loss (through FSDP optimizer)
        # and small gradients from action_loss (through the hook). This is fine —
        # the backbone slowly adapts to the domain while the action head trains fast.

    def _capture_hidden(self, module, input, output):
        """Forward hook: cache hidden states from last DiT block.

        Used in single-head (ActionHead) mode only. Output shape: (B, T, H, W, D).
        """
        self._cached_hidden = output

    def _make_hidden_hook_for_slot(self, slot: int):
        """Build a forward hook that writes its DiT block's output to a specific slot.

        Used in action-expert mode where K different DiT blocks are tapped, each
        hook writing into its own index of self._cached_hiddens.
        """
        def _hook(module, input, output):
            self._cached_hiddens[slot] = output
        return _hook

    def _capture_text_ctx(self, module, input, output):
        """Forward hook: cache the DiT's projected text context (B, L_t, T_dim)."""
        self._cached_text_ctx = output

    def _build_proprio_context_from_batch(self, data_batch: Dict) -> Optional[torch.Tensor]:
        """Build one projected proprio token from data_batch['state']."""
        if self.proprio_conditioner is None:
            return None

        state = data_batch.get("state", None)
        assert state is not None, (
            "use_proprio_conditioning=True requires data_batch['state']; "
            f"available keys={list(data_batch.keys())}"
        )
        if state.dim() == 3:
            assert state.shape[1] >= 1, f"state must have at least one time step; got {tuple(state.shape)}"
            state = state[:, 0, :]
        elif state.dim() != 2:
            raise AssertionError(
                f"state must be (B, D) or (B, T, D); got {tuple(state.shape)}"
            )
        assert state.shape[-1] == self.config.proprio_dim, (
            f"state last dim {state.shape[-1]} != proprio_dim {self.config.proprio_dim}"
        )

        param = next(self.proprio_conditioner.parameters())
        state = state.to(device=param.device, dtype=param.dtype)
        proprio_ctx = self.proprio_conditioner(state).unsqueeze(1)

        if not self._logged_proprio_ctx:
            import os
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[Proprio] state={tuple(state.shape)} "
                    f"ctx={tuple(proprio_ctx.shape)} "
                    f"condition_dit={self.config.proprio_condition_dit} "
                    f"condition_action_head={self.config.proprio_condition_action_head} "
                    f"gate={torch.tanh(self.proprio_conditioner.gate).item():.6f}",
                    flush=True,
                )
            self._logged_proprio_ctx = True
        return proprio_ctx

    def _proprio_context_for_dit(self) -> Optional[torch.Tensor]:
        if not (
            self.config.use_proprio_conditioning
            and self.config.proprio_condition_dit
        ):
            return None
        assert self._current_proprio_ctx is not None, (
            "proprio_condition_dit=True but no current proprio context was built."
        )
        return self._current_proprio_ctx

    def _append_proprio_context_for_action_head(self, text_ctx: torch.Tensor) -> torch.Tensor:
        if not (
            self.config.use_proprio_conditioning
            and self.config.proprio_condition_action_head
        ):
            return text_ctx
        assert self._current_proprio_ctx is not None, (
            "proprio_condition_action_head=True but no current proprio context was built."
        )
        proprio_ctx = self._current_proprio_ctx.to(device=text_ctx.device, dtype=text_ctx.dtype)
        return torch.cat([text_ctx, proprio_ctx], dim=1)

    def _flatten_video_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Flatten DiT block hidden state from (B, T, H, W, D) to (B, T*H*W, D).

        When `video_xattn_frame0_only=True` is set on the config, slice T to
        keep only the first latent frame before flattening, yielding (B, H*W, D).
        That removes the model-predicted future-frame tokens from the action
        head's cross-attention KV, leaving only tokens grounded in the current
        observation (frame 0). See ActionHeadVideoModelConfig docstring.
        """
        f0_only = getattr(self.config, "video_xattn_frame0_only", False)
        if not getattr(self, "_logged_video_slice", False):
            import logging as _logging
            _Bin, _Tin, _Hin, _Win, _Din = hidden.shape
            _Tout = 1 if f0_only else _Tin
            _logging.getLogger(__name__).info(
                f"[ActionHead] _flatten_video_hidden first-call: "
                f"video_xattn_frame0_only={f0_only}, "
                f"input T={_Tin}, kept T={_Tout}, HxW={_Hin}x{_Win}, D={_Din}"
            )
            self._logged_video_slice = True
        if f0_only:
            hidden = hidden[:, :1, :, :, :]
        B, T, H, W, D = hidden.shape
        return hidden.reshape(B, T * H * W, D)

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: "Text2WorldCondition",
    ):
        """Override denoise to strip 'action' from condition dict.

        The ActionConditionedConditioner adds 'action' to the condition, but our
        base DiT (ActionConditionedMinimalV1LVGDiT) has an action_embedder with
        wrong dimensions for raw 7D actions. Since we predict actions via the
        separate ActionHead (not through the DiT), we remove 'action' here.
        """
        # The base DiT (ActionConditionedMinimalV1LVGDiT) requires 'action' in its
        # forward for conditioning. Pass a zero action of the expected shape to make
        # the embedder produce near-zero contributions. The network expects
        # action_dim (default 80) * num_action_per_chunk (12) = 960 flattened input,
        # but data batch has 384D multi-embodiment actions. We create a dummy.
        cond_dict = condition.to_dict()
        B = xt_B_C_T_H_W.shape[0]
        # Network expects (B, num_action_per_chunk, action_dim) = (B, 12, 80)
        net_action_dim = getattr(self.net, '_action_dim', 80)
        net_num_chunks = getattr(self.net, '_num_action_per_chunk', 12)
        cond_dict["action"] = torch.zeros(
            B, net_num_chunks, net_action_dim,
            device=xt_B_C_T_H_W.device, dtype=xt_B_C_T_H_W.dtype,
        )

        if condition.is_video:
            condition_state_in = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in = condition_state_in * 0
            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, C, 1, 1, 1
            ).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = (
                condition_state_in * condition_video_mask
                + xt_B_C_T_H_W * (1 - condition_video_mask)
            )

        net_output = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            proprio_context_emb=self._proprio_context_for_dit(),
            **cond_dict,
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            _, C, _, _, _ = net_output.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, C, 1, 1, 1
            ).type_as(net_output)
            net_output = net_output * (1 - condition_video_mask)

        return net_output

    @torch.no_grad()
    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """Override of parent's get_velocity_fn_from_batch with CFG order swapped:
        UNCOND first, COND last (parent
        action_conditioned_video2world_rectified_flow_model.py:222-294 runs
        cond first, uncond last).

        Why: the action head reads from forward-hook caches (_cached_hiddens,
        _cached_text_ctx) populated during the DiT forward. The hook fires on
        every denoise call, so the LAST denoise per step wins. With cond-last,
        the cache ends every step holding COND hiddens — matching the
        distribution the action head saw during training (training is no-CFG
        with 20% text dropout). With cond-first (parent default), the cache
        held UNCOND hiddens, which the head never trained on. Math of CFG
        is order-independent (`cond + guidance * (cond - uncond)` is
        symmetric in evaluation order), so this swap is compute-free.

        Mirrors parent's structure verbatim except the cond/uncond order
        inside the inner velocity_fn closure.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # set_video_condition populates gt_frames on both branches — required by
        # our overridden denoise() which reads condition.gt_frames at line ~360.
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        self._current_proprio_ctx = self._build_proprio_context_from_batch(data_batch)

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            # UNCOND first, COND last → forward-hook caches end the call holding
            # COND hiddens, which is what the action head was trained on.
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            # Same CFG formula as the parent
            # (action_conditioned_video2world_rectified_flow_model.py:291).
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor]:
        """
        Forward pass: video flow matching + action regression.

        1. Compute text embeddings (online or precomputed)
        2. Run standard flow matching on video (triggers hook → captures hidden states)
        3. Action head reads hidden states → predicts actions
        4. Return video_loss + action_loss_weight * action_MSE
        """
        # --- Text embeddings ---
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            # When the text encoder is configured for image conditioning, feed the
            # current observation frame to Reason1 alongside the caption so it
            # produces joint vision-language tokens. Standard convention:
            # data_batch["video"] is (B, C, T, H, W) in [-1, 1] (the encoder's
            # joint method clamp+rescales explicitly).
            #
            # Gating on text_encoder_config.image_conditioning (not on a specific
            # action head flag) lets the encoder mode be chosen orthogonally to
            # the action head architecture. VlmActionExpert YAMLs set both flags;
            # JointFlowActionExpert YAMLs that opt into image-conditioned text
            # only need to flip image_conditioning=True.
            if getattr(self.config.text_encoder_config, "image_conditioning", False):
                if "reason1_conditioning_frames" in data_batch:
                    data_batch["conditioning_frames"] = data_batch["reason1_conditioning_frames"].contiguous()
                else:
                    video_tensor = data_batch[self.input_data_key]
                    assert video_tensor.dim() == 5, (
                        f"expected (B, C, T, H, W) video; got {tuple(video_tensor.shape)}"
                    )
                    data_batch["conditioning_frame"] = video_tensor[:, :, 0].contiguous()
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        self._current_proprio_ctx = self._build_proprio_context_from_batch(data_batch)

        # --- Video flow matching ---
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        # Denoise (triggers forward hook → self._cached_hidden is populated)
        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        # Video loss (standard rectified flow)
        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
            dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        video_loss = torch.mean(time_weights_B * per_instance_loss)

        # --- Action regression from hidden states ---
        action_loss = torch.tensor(0.0, device=video_loss.device)
        # Branch: multi-tap experts (deterministic ActionExpert / JointActionExpert /
        #         FlowActionExpert / JointFlowActionExpert / VlmActionExpert) all
        #         need K DiT block taps populated; single-head ActionHead only
        #         needs the last block.
        if (
            self.config.use_action_expert
            or self.config.use_flow_action_expert
            or self.config.use_joint_flow_action_expert
            or self.config.use_vlm_action_expert
            or self.config.use_joint_action_expert
        ):
            head_inputs_ready = (
                self._cached_hiddens is not None
                and all(h is not None for h in self._cached_hiddens)
            )
        else:
            head_inputs_ready = self._cached_hidden is not None

        if head_inputs_ready and "action" in data_batch:
            # Text context cached by the crossattn_proj forward hook above.
            # Training has no CFG → the cache is the conditional projected text.
            assert self._cached_text_ctx is not None, (
                "_cached_text_ctx is None — DiT forward did not fire crossattn_proj hook. "
                "Check that data_batch['t5_text_embeddings'] was set and use_crossattn_projection=True."
            )
            text_ctx = self._cached_text_ctx
            text_ctx = self._append_proprio_context_for_action_head(text_ctx)

            # Ground truth actions: 384D multi-embodiment vector → embodiment slice.
            full_action = data_batch["action"]  # (B, num_steps, 384 or action_dim)
            action_dim = self.config.action_dim
            if full_action.shape[-1] > action_dim:
                offset = self.config.gt_action_offset
                gt_actions = full_action[:, :self.config.num_action_steps,
                                         offset:offset + action_dim]
            else:
                gt_actions = full_action[:, :self.config.num_action_steps, :action_dim]

            if (
                self.config.use_flow_action_expert
                or self.config.use_joint_flow_action_expert
                or self.config.use_vlm_action_expert
            ):
                # Flow-matching loss: pi0-style Beta(1.5, 1.0) time, MSE on velocity.
                # All three flow-expert variants share this loss recipe and call
                # signature: (video_hiddens, text_or_vlm_ctx, a_t, t) → v.
                #   FlowActionExpert      — text_ctx is text-only DiT projection
                #   JointFlowActionExpert — text_ctx is text-only DiT projection
                #   VlmActionExpert       — text_ctx is the joint Reason1 VL ctx
                #                            (when image_conditioning=True upstream)
                video_hiddens_flat = [
                    self._flatten_video_hidden(h) for h in self._cached_hiddens
                ]
                B = gt_actions.shape[0]
                device = gt_actions.device
                # Sample t ~ Beta(1.5, 1.0), clip to [0.001, 1.0] (pi0 convention).
                t = sample_beta(1.5, 1.0, B, device) * 0.999 + 0.001
                # Cast a0 to head dtype (bfloat16) for interpolation in matching dtype.
                head_dtype = video_hiddens_flat[0].dtype
                a0 = gt_actions.to(dtype=head_dtype, device=device)
                eps = torch.randn_like(a0)
                # x_t = t*noise + (1-t)*a0; velocity target = noise - a0.
                t_exp = t.to(dtype=head_dtype).view(B, 1, 1)
                a_t = t_exp * eps + (1.0 - t_exp) * a0
                v_target = eps - a0
                v_pred = self.action_head(video_hiddens_flat, text_ctx, a_t, t)
                action_loss = F.mse_loss(v_pred.float(), v_target.float())
            else:
                if self.config.use_action_expert or self.config.use_joint_action_expert:
                    # Both deterministic multi-tap experts use the same call
                    # signature (video_hiddens_flat, text_ctx) -> actions.
                    video_hiddens_flat = [
                        self._flatten_video_hidden(h) for h in self._cached_hiddens
                    ]
                    pred_actions = self.action_head(video_hiddens_flat, text_ctx)  # (B, 12, 7)
                else:
                    hidden_flat = self._flatten_video_hidden(self._cached_hidden)
                    pred_actions = self.action_head(hidden_flat, text_ctx)  # (B, 12, 7)
                gt_actions = gt_actions.to(dtype=pred_actions.dtype, device=pred_actions.device)
                action_loss = F.mse_loss(pred_actions, gt_actions)

        # Total loss
        loss = video_loss + self.config.action_loss_weight * action_loss

        # Clear hook cache(s)
        self._cached_hidden = None
        if self._cached_hiddens is not None:
            for k in range(len(self._cached_hiddens)):
                self._cached_hiddens[k] = None
        self._current_proprio_ctx = None

        # Logging (every 100 steps)
        if not hasattr(self, '_fwd_counter'):
            self._fwd_counter = 0
        self._fwd_counter += 1
        if self._fwd_counter % 100 == 1:
            import os
            if int(os.environ.get("RANK", "0")) == 0:
                ah_w = sum(
                    p.data.norm().item() for p in self.action_head.parameters()
                ) / max(1, sum(1 for _ in self.action_head.parameters()))
                print(
                    f"[ActionHead] total={loss.item():.4f} "
                    f"video={video_loss.item():.4f} "
                    f"action={action_loss.item():.4f} "
                    f"(w_action={self.config.action_loss_weight}) "
                    f"head_avg_w={ah_w:.4f}",
                    flush=True,
                )

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": loss,
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
        }
        return output_batch, loss

    def _maybe_build_text_encoder_lora_optim(self):
        """Lazily build the Reason1 LoRA optimizer on first use.

        Deferred from set_up_model() because self.text_encoder is created AFTER
        set_up_model() in the parent class's __init__ (see
        text2world_model_rectified_flow.py:172-177). Idempotent.
        """
        if self._text_encoder_lora_optim_built:
            return
        self._text_encoder_lora_optim_built = True
        if not (
            self.config.text_encoder_config is not None
            and getattr(self.config.text_encoder_config, "use_lora", False)
        ):
            return
        if getattr(self, "text_encoder", None) is None:
            log.warning(
                "[Reason1 LoRA] use_lora=True but self.text_encoder is None "
                "(maybe compute_online=False?). Skipping LoRA optimizer."
            )
            return
        lora_params = list(self.text_encoder.lora_parameters())
        if len(lora_params) == 0:
            log.warning(
                "[Reason1 LoRA] use_lora=True but no LoRA params were collected — "
                "no Reason1 optimizer will be created."
            )
            return
        self._text_encoder_lora_optim = torch.optim.AdamW(
            lora_params,
            lr=self.config.text_encoder_lora_lr,
            weight_decay=0.0,
        )
        n = sum(p.numel() for p in lora_params)
        log.info(
            f"[Reason1 LoRA] dedicated optimizer over {n:,} params at "
            f"lr={self.config.text_encoder_lora_lr}"
        )
        print(f"[Reason1 LoRA] optimizer built ({n:,} params, lr={self.config.text_encoder_lora_lr})", flush=True)

    def on_before_zero_grad(self, optimizer, scheduler, iteration):
        """Step dedicated optimizers for action head + Reason1 LoRA (non-FSDP params)."""
        self._maybe_build_text_encoder_lora_optim()
        if self.config.action_head_warmup_steps > 0:
            warmup_steps = self.config.action_head_warmup_steps
            if iteration < warmup_steps:
                factor = self.config.action_head_lr_start_factor + (
                    1.0 - self.config.action_head_lr_start_factor
                ) * iteration / warmup_steps
            else:
                factor = 1.0
            for group in self._action_head_optim.param_groups:
                group["lr"] = self.config.action_head_lr * factor
        torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), max_norm=1.0)
        self._action_head_optim.step()
        self._action_head_optim.zero_grad(set_to_none=True)

        # Step Reason1 LoRA optimizer (when enabled).
        if self._text_encoder_lora_optim is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.text_encoder.lora_parameters()),
                max_norm=1.0,
            )
            self._text_encoder_lora_optim.step()
            self._text_encoder_lora_optim.zero_grad(set_to_none=True)

    def save_auxiliary_state(self, ckpt_dir: str, iteration: int):
        """Save action head model weights + optimizer state alongside DCP checkpoint.

        The action_head lives outside the FSDP boundary, so its weights are NOT
        included in the DCP model checkpoint. We save them separately here.
        """
        import os
        if int(os.environ.get("RANK", "0")) != 0:
            return
        # Ensure LoRA optimizer is built so we can save its state when LoRA is enabled.
        self._maybe_build_text_encoder_lora_optim()
        aux = {
            'action_head_state_dict': self.action_head.state_dict(),
            'action_head_optim': self._action_head_optim.state_dict(),
        }
        if self.proprio_conditioner is not None:
            aux["proprio_conditioner_state_dict"] = self.proprio_conditioner.state_dict()
        # Reason1 LoRA: persist only the trainable adapter params + optim state
        # (NOT the full Reason1 7B weights, which are restored from HF on load).
        if self._text_encoder_lora_optim is not None:
            lora_state = {
                k: v for k, v in self.text_encoder.model.state_dict().items()
                if "lora_" in k
            }
            aux["text_encoder_lora_state_dict"] = lora_state
            aux["text_encoder_lora_optim"] = self._text_encoder_lora_optim.state_dict()
        save_path = os.path.join(ckpt_dir, f"iter_{iteration:09}", "aux_optim.pt")
        torch.save(aux, save_path)
        print(f"[ActionHead] Saved action_head weights + optimizer to {save_path}", flush=True)

    def load_auxiliary_state(self, ckpt_dir: str, iteration: int):
        """Restore action head model weights + optimizer state from DCP checkpoint dir."""
        import os
        # Build the LoRA optim before loading so we can restore its state.
        self._maybe_build_text_encoder_lora_optim()
        load_path = os.path.join(ckpt_dir, f"iter_{iteration:09}", "aux_optim.pt")
        print(f"[ActionHead] load_auxiliary_state: ckpt_dir={ckpt_dir}, iter={iteration}, "
              f"path={load_path}, exists={os.path.exists(load_path)}", flush=True)
        if not os.path.exists(load_path):
            print(f"[ActionHead] WARNING: {load_path} not found — action_head weights are RANDOM", flush=True)
            return
        aux = torch.load(load_path, map_location="cpu", weights_only=True)
        print(f"[ActionHead] aux_optim.pt keys: {list(aux.keys())}", flush=True)
        if 'action_head_state_dict' in aux:
            # strict=False so a checkpoint saved under the old (text-less) architecture
            # can still be loaded: missing keys (e.g. text_xattn.*, fuse_proj.*) stay
            # at random init, which is fine — training will fit them.
            result = self.action_head.load_state_dict(aux['action_head_state_dict'], strict=False)
            print(
                f"[ActionHead] Restored {len(aux['action_head_state_dict'])} tensors "
                f"(missing={len(getattr(result, 'missing_keys', []))}, "
                f"unexpected={len(getattr(result, 'unexpected_keys', []))})",
                flush=True,
            )
            if getattr(result, 'missing_keys', []):
                print(f"[ActionHead] missing keys: {result.missing_keys}", flush=True)
            if getattr(result, 'unexpected_keys', []):
                print(f"[ActionHead] unexpected keys: {result.unexpected_keys}", flush=True)
        else:
            print(f"[ActionHead] WARNING: no action_head_state_dict in checkpoint!", flush=True)
        if self.proprio_conditioner is not None:
            if "proprio_conditioner_state_dict" in aux:
                result = self.proprio_conditioner.load_state_dict(
                    aux["proprio_conditioner_state_dict"], strict=False
                )
                print(
                    f"[Proprio] Restored conditioner "
                    f"(missing={len(getattr(result, 'missing_keys', []))}, "
                    f"unexpected={len(getattr(result, 'unexpected_keys', []))})",
                    flush=True,
                )
            else:
                print(
                    "[Proprio] No proprio_conditioner_state_dict in checkpoint; "
                    "using fresh projector with zero gate.",
                    flush=True,
                )
        if 'action_head_optim' in aux:
            try:
                self._action_head_optim.load_state_dict(aux['action_head_optim'])
            except (KeyError, ValueError) as e:
                print(
                    f"[ActionHead] WARNING: action_head_optim restore failed ({e}); "
                    f"keeping fresh optimizer state. This is expected when the head "
                    f"architecture has changed since the checkpoint was saved.",
                    flush=True,
                )
            print(f"[ActionHead] Restored action_head_optim", flush=True)

        # Reason1 LoRA: restore adapter weights + optim state (only if LoRA enabled).
        if self._text_encoder_lora_optim is not None:
            if 'text_encoder_lora_state_dict' in aux:
                # strict=False so a checkpoint saved with different LoRA target_modules
                # or rank still loads cleanly (mismatched keys stay at zero init).
                result = self.text_encoder.model.load_state_dict(
                    aux['text_encoder_lora_state_dict'], strict=False
                )
                # `result.missing_keys` will include ALL non-LoRA keys (since we only
                # saved LoRA params); only `unexpected_keys` is diagnostic here.
                if getattr(result, 'unexpected_keys', []):
                    print(
                        f"[Reason1 LoRA] unexpected keys on restore: {result.unexpected_keys}",
                        flush=True,
                    )
                print(
                    f"[Reason1 LoRA] Restored {len(aux['text_encoder_lora_state_dict'])} adapter tensors",
                    flush=True,
                )
            else:
                print(
                    "[Reason1 LoRA] WARNING: no text_encoder_lora_state_dict in checkpoint — "
                    "adapters at zero init.",
                    flush=True,
                )
            if 'text_encoder_lora_optim' in aux:
                try:
                    self._text_encoder_lora_optim.load_state_dict(aux['text_encoder_lora_optim'])
                    print("[Reason1 LoRA] Restored optimizer state", flush=True)
                except (KeyError, ValueError) as e:
                    print(
                        f"[Reason1 LoRA] WARNING: LoRA optim restore failed ({e}); "
                        "keeping fresh optimizer state.",
                        flush=True,
                    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_with_latents_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        query_steps=[0, 9, 18, 27, 34],
        **kwargs,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor, None]:
        """
        Generate video samples via ODE solver, then predict actions from final hidden states.

        Returns same 4-tuple as WAM model for policy server compatibility:
            (video_latents, latent_dict, action_pred, None)
        """
        # VLM-expert mode: re-encode joint vision-language tokens at inference
        # so the head sees the SAME distribution it trained on. The policy
        # server only sends raw text; we splice the conditioning frame in
        # here and overwrite t5_text_embeddings before the DiT runs.
        if (
            getattr(self.config, "use_vlm_action_expert", False)
            and self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
        ):
            video_tensor = data_batch.get(self.input_data_key, None)
            assert video_tensor is not None and video_tensor.dim() == 5, (
                f"VLM mode at inference requires data_batch['{self.input_data_key}'] "
                f"of shape (B, C, T, H, W); got {None if video_tensor is None else tuple(video_tensor.shape)}"
            )
            if "reason1_conditioning_frames" in data_batch:
                data_batch["conditioning_frames"] = data_batch["reason1_conditioning_frames"].contiguous()
            else:
                data_batch["conditioning_frame"] = video_tensor[:, :, 0].contiguous()
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        # Video noise
        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape), torch.bfloat16,
            self.tensor_kwargs["device"], seed,
        )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps, device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        timesteps = self.sample_scheduler.timesteps

        # Get velocity function (standard video-only CFG)
        velocity_fn = self.get_velocity_fn_from_batch(
            data_batch, guidance, is_negative_prompt=is_negative_prompt,
        )

        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(
                tensor=noise, seq_dim=2,
                process_group=self.get_context_parallel_group(),
            )

        latents = noise
        latent_to_save = {}
        timesteps_iter = timesteps if INTERNAL else tqdm.tqdm(
            timesteps, desc="Generating samples", total=len(timesteps)
        )

        # ODE solver (standard video generation)
        for num_step, t in enumerate(timesteps_iter):
            if num_step in query_steps:
                latent_to_save[num_step] = latents

            timestep = torch.stack([t]).to(**self.tensor_kwargs)
            velocity_pred = velocity_fn(noise, latents, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred, t, latents,
                return_dict=False, generator=seed_g,
            )[0]
            latents = temp_x0

        latent_to_save[num_step] = latents

        # Bug 2 fix lives in the overridden get_velocity_fn_from_batch below
        # (CFG runs uncond → cond, so cache ends each step holding COND hiddens
        # which is what the action head was trained on).

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(
                latents, seq_dim=2, cp_group=self.get_context_parallel_group()
            )

        # --- Action prediction from final hidden states ---
        action_pred = None

        # Branch: multi-tap experts (deterministic ActionExpert / JointActionExpert /
        #         FlowActionExpert / JointFlowActionExpert / VlmActionExpert) all
        #         need K DiT block taps populated; single-head ActionHead only needs
        #         the last block.
        if (
            self.config.use_action_expert
            or self.config.use_flow_action_expert
            or self.config.use_joint_flow_action_expert
            or self.config.use_vlm_action_expert
            or self.config.use_joint_action_expert
        ):
            head_inputs_ready = (
                self._cached_hiddens is not None
                and all(h is not None for h in self._cached_hiddens)
            )
        else:
            head_inputs_ready = self._cached_hidden is not None

        if head_inputs_ready:
            # CFG runs cond + uncond denoise per step → the crossattn_proj hook ends
            # the loop with the *unconditional* projected text. Re-project explicitly
            # to feed the action head conditional text. Cheap: one Linear over the
            # already-computed Qwen embeddings, no re-encoding.
            t5_emb = data_batch["t5_text_embeddings"]
            if getattr(self.net, "use_crossattn_projection", False):
                text_ctx = self.net.crossattn_proj(t5_emb)
            else:
                text_ctx = t5_emb
            text_ctx = self._append_proprio_context_for_action_head(text_ctx)

            if (
                self.config.use_flow_action_expert
                or self.config.use_joint_flow_action_expert
                or self.config.use_vlm_action_expert
            ):
                # Action ODE: pi0-style Euler loop on actions only, with cached
                # video hiddens held fixed across steps. All three flow-expert
                # variants share this loop and the same forward signature
                # (video_hiddens, text_or_vlm_ctx, a, t) → v. For the VLM expert,
                # text_ctx is the joint Reason1 vision-language context (because
                # the encoder ran with image_conditioning=True upstream).
                video_hiddens_flat = [
                    self._flatten_video_hidden(h) for h in self._cached_hiddens
                ]
                head_dtype = video_hiddens_flat[0].dtype
                text_ctx = text_ctx.to(dtype=head_dtype)

                B = video_hiddens_flat[0].shape[0]
                T_a = self.config.num_action_steps
                A = self.config.action_dim
                # Use the same arch_invariant rng as the video noise so eval is
                # reproducible across runs at fixed seed.
                a = misc.arch_invariant_rand(
                    (B, T_a, A), torch.bfloat16,
                    self.tensor_kwargs["device"], seed,
                ).to(dtype=head_dtype)
                num_steps_a = max(1, int(self.config.flow_inference_num_steps))
                dt = -1.0 / num_steps_a
                t_val = 1.0
                # pi0_pytorch.py:647-665 — loop while time >= -dt/2.
                for _ in range(num_steps_a):
                    t_batch = torch.full(
                        (B,), t_val, dtype=torch.float32,
                        device=self.tensor_kwargs["device"],
                    )
                    v = self.action_head(video_hiddens_flat, text_ctx, a, t_batch)
                    a = a + dt * v.to(dtype=head_dtype)
                    t_val += dt
                action_pred = a
                # Clear caches
                for k in range(len(self._cached_hiddens)):
                    self._cached_hiddens[k] = None
            elif self.config.use_action_expert or self.config.use_joint_action_expert:
                # Both deterministic multi-tap experts use the same call signature
                # (video_hiddens_flat, text_ctx) -> actions; single forward pass.
                video_hiddens_flat = [
                    self._flatten_video_hidden(h) for h in self._cached_hiddens
                ]
                text_ctx = text_ctx.to(dtype=video_hiddens_flat[0].dtype)
                action_pred = self.action_head(video_hiddens_flat, text_ctx)
                # Clear caches
                for k in range(len(self._cached_hiddens)):
                    self._cached_hiddens[k] = None
            else:
                hidden_flat = self._flatten_video_hidden(self._cached_hidden)
                text_ctx = text_ctx.to(dtype=hidden_flat.dtype)
                action_pred = self.action_head(hidden_flat, text_ctx)  # (B, 12, 7)
                self._cached_hidden = None
            self._cached_text_ctx = None

        self._current_proprio_ctx = None
        return latents, latent_to_save, action_pred, None
