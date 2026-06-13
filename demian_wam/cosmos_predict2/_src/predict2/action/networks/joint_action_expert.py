# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JointActionExpert: deterministic action expert with joint cross-attention.

Same per-layer joint softmax structure as JointFlowActionExpert
(K, V = concat([video_kv; text_kv; action_self_kv]); Q = action_only) BUT:
  - No flow matching: predicts the action chunk DIRECTLY, MSE against
    ground-truth actions.
  - No noisy action input: Q seed is just a learnable per-step positional
    embedding (12 × D).
  - No time conditioning: no sinusoidal time embedding, no time_mlp, no
    AdaLN scale/shift modulation.
  - Inference: single forward pass, no Euler ODE.

This is the "backup plan" when flow matching's small-t velocity field
produces gripping-phase failures (observed: K=4 spread joint flow at
iter_60K had 0% SR despite good approach trajectories, with inference
knobs ruled out as the cause). The architecture that proved useful in
joint_flow (multi-tap joint cross-attention) is preserved; only the
loss formulation is dropped.

Forward signature matches the deterministic ActionExpert so the model
dispatcher can route both through the same training and inference paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _JointActionExpertLayer(nn.Module):
    """One deterministic joint-attention layer with three K/V sources.

    Out-proj is zero-init so the residual is identity at start — the expert
    is a pass-through until projections are learned. No mask: action queries
    see all video tokens, all text/VLM tokens, and all action chunk positions
    (full bidirectional within the chunk).
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

        # Text/VLM side: norm + K,V projections (per-layer; shared ctx via top-level
        # text_in_proj). Naming follows JointFlowActionExpert (k_text/v_text) for
        # backward consistency — the "text" slot carries whatever the upstream
        # text encoder produces (text-only Reason1 OR joint VL Reason1).
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.k_text = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_text = nn.Linear(hidden_dim, hidden_dim, bias=False)

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
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        q: torch.Tensor,
        video_kv: torch.Tensor,
        text_kv: torch.Tensor,
    ) -> torch.Tensor:
        # ---- Joint attention: 1 softmax over [video_kv; text_kv; action_self] ----
        q_n = self.norm_q(q)
        v_n = self.norm_video(video_kv)
        t_n = self.norm_text(text_kv)

        Q = self._split_heads(self.q_proj(q_n))
        Kv = self._split_heads(self.k_video(v_n))
        Vv = self._split_heads(self.v_video(v_n))
        Kt = self._split_heads(self.k_text(t_n))
        Vt = self._split_heads(self.v_text(t_n))
        Kq = self._split_heads(self.k_action(q_n))
        Vq = self._split_heads(self.v_action(q_n))

        K = torch.cat([Kv, Kt, Kq], dim=2)
        V = torch.cat([Vv, Vt, Vq], dim=2)

        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)
        q = q + self.out_proj(self._merge_heads(attn))

        # ---- FFN ----
        h = self.norm_ffn(q)
        q = q + self.ffn(h)
        return q


class JointActionExpert(nn.Module):
    """K-layer deterministic action expert with per-layer joint cross-attention.

    Args:
        hidden_dim: working dim (e.g. 2048 for cosmos 2B).
        text_dim: input dim of DiT-projected text/VLM (e.g. 1024 for cosmos 2B).
        num_actions: number of action steps (e.g. 12 for RoboCasa).
        action_dim: action dimensionality (e.g. 7 for RoboCasa EE control).
        num_heads: attention heads.
        mlp_ratio: FFN expansion factor.
        num_layers: K — must match the number of DiT taps the caller passes.

    Forward signature is `(video_hiddens, text_ctx) → (B, num_actions, action_dim)`,
    matching the existing deterministic ActionExpert so the model dispatcher can
    route both through the same training-loss / inference call site.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        text_dim: int = 1024,
        num_actions: int = 12,
        action_dim: int = 7,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_layers: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_layers = num_layers

        # Action queries: learnable per-step positional embedding only (no
        # noisy-action input, no time conditioning).
        self.action_pos_emb = nn.Parameter(torch.randn(1, num_actions, hidden_dim) * 0.02)

        # One-time projection of the DiT-projected text/VLM ctx (text_dim) into
        # hidden_dim. Per-layer K,V projections live inside _JointActionExpertLayer.
        if text_dim != hidden_dim:
            self.text_in_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_in_proj = nn.Identity()

        self.layers = nn.ModuleList([
            _JointActionExpertLayer(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Output head: LayerNorm → Linear, zero-init for stable training start.
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

    def forward(
        self,
        video_hiddens: list[torch.Tensor],
        text_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_hiddens: list of K tensors, each (B, L_v, hidden_dim) — one per
                layer (caller flattens (B, T, H, W, D) → (B, T*H*W, D)).
            text_ctx: (B, L_t, text_dim) — DiT-projected text or joint VL ctx.

        Returns:
            actions: (B, num_actions, action_dim).
        """
        assert len(video_hiddens) == self.num_layers, (
            f"JointActionExpert has {self.num_layers} layers but caller supplied "
            f"{len(video_hiddens)} video hiddens"
        )
        assert text_ctx.dim() == 3 and text_ctx.shape[-1] == self.text_dim, (
            f"text_ctx must be (B, L_t, {self.text_dim}); got {tuple(text_ctx.shape)}"
        )
        for k, h in enumerate(video_hiddens):
            assert h.dim() == 3 and h.shape[-1] == self.hidden_dim, (
                f"video_hiddens[{k}] must be (B, L_v, {self.hidden_dim}); "
                f"got {tuple(h.shape)}"
            )

        B = video_hiddens[0].shape[0]
        dtype = video_hiddens[0].dtype

        # Q seed: learnable per-step embedding broadcast to batch.
        q = self.action_pos_emb.expand(B, -1, -1).to(dtype=dtype)

        # Text/VLM K/V context (shared across layers; per-layer K,V projections live
        # inside _JointActionExpertLayer).
        text_kv = self.text_in_proj(text_ctx).to(dtype=dtype)

        for k, layer in enumerate(self.layers):
            q = layer(q, video_hiddens[k].to(dtype=dtype), text_kv)

        return self.action_proj(self.final_norm(q))
