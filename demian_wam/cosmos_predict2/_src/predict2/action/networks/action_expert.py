# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-layer ActionExpert: a small transformer that reads from K different DiT
blocks (per-layer cross-attention) and the DiT's projected text context.

Architecture (mirrors π0 / GR00T-N1's action-expert pattern adapted to a
video-only DiT residual stream):

    queries (12 learnable) ──┐
                             ├─► [ ExpertLayer_0 ] ◄── DiT.blocks[L_0].out
                             │                    ◄── crossattn_proj_out (text)
                             ├─► [ ExpertLayer_1 ] ◄── DiT.blocks[L_1].out
                             │                    ◄── crossattn_proj_out
                             │     ...
                             ├─► [ ExpertLayer_{K-1} ] ◄── DiT.blocks[L_{K-1}].out
                             │                        ◄── crossattn_proj_out
                             └─► LayerNorm → Linear → (12, action_dim)

Each ExpertLayer is a transformer-decoder block:
    q = q + SelfAttn(LN(q))                        # action queries reason
    q = q + DiTCrossAttn(LN(q), video_kv, video_kv)# read DiT block at tap k
    q = q + TextCrossAttn(LN(q), text_kv, text_kv) # read shared projected text
    q = q + FFN(LN(q))                             # mlp_ratio expansion

Notes:
- Pre-LayerNorm everywhere (standard for stability with deep stacks).
- `text_in_proj` brings DiT's text-context (text_dim, e.g. 1024 for 2B) up to
  hidden_dim (e.g. 2048) once at the front; the projected text is then shared
  across all K expert layers (same KV, different queries each layer).
- Final `action_proj` is zero-initialized so the expert starts at "predict
  zero" and improves from there — critical for stability when bolting a new
  head onto a pre-trained DiT.
- `mlp_ratio` here is *expansion* (transformer convention): FFN hidden =
  mlp_ratio * hidden_dim. This is the opposite convention from the older
  ActionHead's `mlp_ratio`, which compressed.
"""

import torch
import torch.nn as nn


class _ExpertLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        # Self-attention over action queries
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        # Cross-attention to one DiT block's hidden states (the per-layer tap)
        self.norm_video_q = nn.LayerNorm(hidden_dim)
        self.video_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        # Cross-attention to the (shared) projected text context
        self.norm_text_q = nn.LayerNorm(hidden_dim)
        self.text_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        # Feed-forward block (transformer convention: hidden → expansion → hidden)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden = mlp_ratio * hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        video_kv: torch.Tensor,
        text_kv: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attn over the 12 action queries — lets actions in the chunk reason
        # about each other (rough autoregressive structure within the chunk).
        sa_out, _ = self.self_attn(
            self.norm_self(q), self.norm_self(q), self.norm_self(q),
        )
        q = q + sa_out

        # Read DiT video features at this layer's tap.
        v_q = self.norm_video_q(q)
        v_out, _ = self.video_xattn(v_q, video_kv, video_kv)
        q = q + v_out

        # Read text features (same KV every expert layer; queries differ).
        t_q = self.norm_text_q(q)
        t_out, _ = self.text_xattn(t_q, text_kv, text_kv)
        q = q + t_out

        # Position-wise FFN.
        q = q + self.ffn(self.norm_ffn(q))
        return q


class ActionExpert(nn.Module):
    """K-layer transformer action expert with per-layer DiT taps + shared text.

    Args:
        hidden_dim: DiT model channel dimension (e.g. 2048 for 2B). Also the
            queries' working dim.
        text_dim: DiT text-context channel dim (e.g. 1024 for 2B; output of
            `crossattn_proj`). Different from hidden_dim — we add an
            in-projection so text K/V live in hidden_dim space.
        num_actions: Number of action steps to predict (e.g. 12).
        action_dim: Action dimensionality (e.g. 7 for RoboCasa).
        num_heads: Number of attention heads (used by all three attns).
        mlp_ratio: FFN expansion factor (transformer convention; e.g. 2 →
            FFN hidden = 4096 for hidden_dim=2048). Note: this is the OPPOSITE
            of `ActionHead.mlp_ratio`, which compresses.
        num_layers: K — number of expert layers, must match the number of DiT
            taps wired by the caller.
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

        self.action_queries = nn.Parameter(
            torch.randn(1, num_actions, hidden_dim) * 0.02
        )

        # One-time projection of DiT-projected text from text_dim to hidden_dim.
        # All K expert layers share this projected text as KV.
        if text_dim != hidden_dim:
            self.text_in_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_in_proj = nn.Identity()

        # K transformer-decoder-style expert layers.
        self.layers = nn.ModuleList([
            _ExpertLayer(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, action_dim)

        # Zero-init output projection for stable training start
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

    def forward(
        self,
        video_hiddens: list[torch.Tensor],
        text_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_hiddens: list of K tensors, each (B, L_v, hidden_dim), one
                per expert layer; ordered from earliest DiT tap to latest.
                Caller is responsible for flattening (B, T, H, W, D) → (B, L_v, D).
            text_ctx: (B, L_t, text_dim) — DiT's `crossattn_proj` output, which
                is `crossattn_proj(t5_text_embeddings)`.

        Returns:
            actions: (B, num_actions, action_dim).
        """
        assert len(video_hiddens) == self.num_layers, (
            f"action expert has {self.num_layers} layers but caller supplied "
            f"{len(video_hiddens)} video hiddens; one tap per layer required"
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
        q = self.action_queries.expand(B, -1, -1)        # (B, num_actions, D)
        text_kv = self.text_in_proj(text_ctx)            # (B, L_t, hidden_dim)

        for k, layer in enumerate(self.layers):
            q = layer(q, video_hiddens[k], text_kv)

        return self.action_proj(self.final_norm(q))      # (B, num_actions, action_dim)
