# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VlmActionExpert: joint blockwise attention over [DiT video; Reason1 VL; action self].

The earlier VlmActionExpert read ONLY Reason1's joint vision-language tokens
and dropped the DiT-feature stream entirely (motivation: "DiT is a video
generator, not a feature provider"). The 11662272 joint_flow run validated
in May 2026 that joint blockwise attention over [DiT video; text; action]
produces purposeful, scene-grounded actions — so refusing to attend to the
DiT was leaving signal on the table. This module brings VLM-conditioned
training in line with that finding by widening the joint softmax to all
three sources.

Architecture (mirrors JointFlowActionExpert with one extra K/V slot):

    queries (12 learnable + linearly-projected noisy actions) ──┐
                                                                 ├─► [_JointVlmExpertLayer_0]  ◄── DiT tap[0], VLM ctx
                                                                 ├─► [_JointVlmExpertLayer_1]  ◄── DiT tap[1], VLM ctx
                                                                 ├─► ...
                                                                 ├─► [_JointVlmExpertLayer_{K-1}]
                                                                 └─► LayerNorm → Linear → velocity (12, action_dim)

Each layer runs ONE softmax over `concat([K_video; K_vlm; K_action_self], seq)`
(and similarly concat'd V). Action queries get AdaLN modulation carrying the
flow timestep; video and VLM K/V are LN'd but not modulated. Two AdaLN sites
per layer: joint-attn input (modulates action q only) and FFN input.

Forward signature matches JointFlowActionExpert so the model dispatcher can
route flow_expert / joint_flow_expert / vlm_expert through the same call site.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_predict2._src.predict2.action.networks.flow_action_expert import (
    create_sinusoidal_pos_embedding,
)


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _JointVlmExpertLayer(nn.Module):
    """One joint-attention layer with three K/V sources.

    Out-proj is zero-init so the residual is identity at start — the expert
    behaves like a pass-through until time conditioning + projections are
    learned. No mask: action queries see all video tokens, all VLM tokens,
    and all action chunk positions (full bidirectional within the chunk).
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Action side: norm + Q,K,V projections (K,V used for action self-attn slot).
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_action = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_action = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Video side: norm + K,V projections (per-layer, fed by DiT block taps).
        self.norm_video = nn.LayerNorm(hidden_dim)
        self.k_video = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_video = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # VLM side: norm + K,V projections (per-layer K,V; VLM ctx itself shared
        # across layers via the top-level vlm_in_proj).
        self.norm_vlm = nn.LayerNorm(hidden_dim)
        self.k_vlm = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_vlm = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Joint output projection — zero-init for stable start.
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # FFN.
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden = mlp_ratio * hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → (B, H, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, head_dim) → (B, T, D)
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        q: torch.Tensor,
        video_kv: torch.Tensor,
        vlm_kv: torch.Tensor,
        scale_shifts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s_at, b_at, s_ff, b_ff = scale_shifts

        # ---- Joint attention: 1 softmax over [video_kv; vlm_kv; action_self] ----
        q_n = _modulate(self.norm_q(q), s_at, b_at)
        v_n = self.norm_video(video_kv)
        m_n = self.norm_vlm(vlm_kv)

        Q = self._split_heads(self.q_proj(q_n))
        Kv = self._split_heads(self.k_video(v_n))
        Vv = self._split_heads(self.v_video(v_n))
        Km = self._split_heads(self.k_vlm(m_n))
        Vm = self._split_heads(self.v_vlm(m_n))
        Kq = self._split_heads(self.k_action(q_n))
        Vq = self._split_heads(self.v_action(q_n))

        # Concat along the SEQUENCE dim (dim=2 after split_heads)
        K = torch.cat([Kv, Km, Kq], dim=2)
        V = torch.cat([Vv, Vm, Vq], dim=2)

        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)
        q = q + self.out_proj(self._merge_heads(attn))

        # ---- FFN ----
        h = _modulate(self.norm_ffn(q), s_ff, b_ff)
        q = q + self.ffn(h)
        return q


class VlmActionExpert(nn.Module):
    """K-layer joint-attention expert reading DiT video + Reason1 VL + action self.

    Forward signature matches JointFlowActionExpert / FlowActionExpert:
        forward(video_hiddens: list, vlm_ctx, noisy_action, timestep) → velocity

    so the model dispatcher can route flow_expert / joint_flow_expert /
    vlm_expert through the same training and inference code paths.

    Args:
        hidden_dim: working dim (e.g. 2048 for cosmos 2B). Action queries and
            both K/V streams (after projection) live in this dim.
        vlm_dim: input dim of the projected VLM tokens (output of DiT's
            crossattn_proj when image_conditioning=True; 1024 for cosmos 2B).
            Different from hidden_dim; we project up once at the front.
        num_actions: number of action steps to predict (e.g. 12).
        action_dim: action dimensionality (e.g. 7 for RoboCasa).
        num_heads: attention heads.
        mlp_ratio: FFN expansion factor.
        num_layers: K — must match the number of DiT taps the caller passes.
        time_min_period / time_max_period: pi0 defaults (4e-3, 4.0).
    """

    NUM_MODS_PER_LAYER = 2  # joint-attn, ffn

    def __init__(
        self,
        hidden_dim: int = 2048,
        vlm_dim: int = 1024,
        num_actions: int = 12,
        action_dim: int = 7,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_layers: int = 4,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vlm_dim = vlm_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period

        # Action chunk in-projection + learned per-step positional embedding.
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos_emb = nn.Parameter(torch.randn(1, num_actions, hidden_dim) * 0.02)

        # One-time projection of the DiT-projected VLM tokens (vlm_dim) into
        # hidden_dim. Per-layer K,V projections live inside _JointVlmExpertLayer.
        if vlm_dim != hidden_dim:
            self.vlm_in_proj = nn.Linear(vlm_dim, hidden_dim)
        else:
            self.vlm_in_proj = nn.Identity()

        # Time MLP. 2 modulation sites per layer × 2 (scale,shift) × K × D.
        out_dim = 2 * self.NUM_MODS_PER_LAYER * num_layers * hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.layers = nn.ModuleList([
            _JointVlmExpertLayer(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

    def forward(
        self,
        video_hiddens: list[torch.Tensor],
        vlm_ctx: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_hiddens: list of K tensors, each (B, L_v, hidden_dim) — one per
                layer (caller flattens (B, T, H, W, D) → (B, T*H*W, D)).
            vlm_ctx: (B, L_vlm, vlm_dim) — DiT-projected Reason1 joint VL tokens
                (output of `crossattn_proj` applied to Reason1's joint hidden
                states; requires text_encoder.image_conditioning=True upstream).
            noisy_action: (B, num_actions, action_dim) — `a_t`.
            timestep: (B,) continuous time in [0, 1].

        Returns:
            velocity: (B, num_actions, action_dim).
        """
        assert len(video_hiddens) == self.num_layers, (
            f"VlmActionExpert has {self.num_layers} layers but caller supplied "
            f"{len(video_hiddens)} video hiddens"
        )
        assert vlm_ctx.dim() == 3 and vlm_ctx.shape[-1] == self.vlm_dim, (
            f"vlm_ctx must be (B, L, {self.vlm_dim}); got {tuple(vlm_ctx.shape)}"
        )
        assert noisy_action.dim() == 3 and noisy_action.shape[-2:] == (
            self.num_actions, self.action_dim,
        ), (
            f"noisy_action must be (B, {self.num_actions}, {self.action_dim}); "
            f"got {tuple(noisy_action.shape)}"
        )
        assert timestep.dim() == 1 and timestep.shape[0] == noisy_action.shape[0], (
            f"timestep must be (B,) matching noisy_action batch; "
            f"got {tuple(timestep.shape)} vs B={noisy_action.shape[0]}"
        )
        for k, h in enumerate(video_hiddens):
            assert h.dim() == 3 and h.shape[-1] == self.hidden_dim, (
                f"video_hiddens[{k}] must be (B, L_v, {self.hidden_dim}); "
                f"got {tuple(h.shape)}"
            )

        B = noisy_action.shape[0]
        dtype = noisy_action.dtype

        q = self.action_in_proj(noisy_action) + self.action_pos_emb.to(dtype=dtype)
        vlm_kv = self.vlm_in_proj(vlm_ctx).to(dtype=dtype)

        time_emb = create_sinusoidal_pos_embedding(
            timestep.to(torch.float32),
            self.hidden_dim,
            min_period=self.time_min_period,
            max_period=self.time_max_period,
            device=timestep.device,
        ).to(dtype=dtype)
        all_mods = self.time_mlp(time_emb)  # (B, 2 * NUM_MODS_PER_LAYER * K * D)
        # Reshape → (K, NUM_MODS*2, B, D); per-layer slice = (s_at, b_at, s_ff, b_ff).
        mods = all_mods.view(B, self.num_layers, self.NUM_MODS_PER_LAYER * 2, self.hidden_dim)
        mods = mods.permute(1, 2, 0, 3).contiguous()  # (K, 4, B, D)

        for k, layer in enumerate(self.layers):
            scale_shifts = tuple(mods[k][i] for i in range(self.NUM_MODS_PER_LAYER * 2))
            q = layer(q, video_hiddens[k].to(dtype=dtype), vlm_kv, scale_shifts)

        return self.action_proj(self.final_norm(q))
