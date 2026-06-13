# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Action prediction head that reads DiT hidden states + DiT-projected text context
via two parallel cross-attentions whose outputs are concatenated and fused.

Why two parallel cross-attentions instead of one:
- Text and video signals are complementary, not competing — text carries task
  identity (stable per-episode); video carries pose/scene (changes per step).
- A single cross-attn over concatenated KV forces them to share one softmax
  budget, and video has ~2x more tokens, biasing attention away from text early.
- Parallel attns give each modality its own softmax; the fuse layer learns the
  per-channel mix.

Mandatory `text_ctx`: removing the optional guard turns silent text-loss into a
loud TypeError. This head will not function without text input.
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Parallel video+text cross-attention action head on DiT features.

    Architecture:
        action_queries
            ├─ video_xattn → q_v ─┐
            ├─ text_xattn  → q_t ─┴─ concat → fuse_proj → +residual → MLP → actions

    Args:
        hidden_dim: DiT model channel dimension (2048 for 2B model). The video
            features (DiT block-final hidden) and the queries operate at this dim.
        text_dim: DiT text-context channel dimension (the output dim of
            DiT.crossattn_proj). For Cosmos 2B this is 1024 — different from
            hidden_dim. We add an internal Linear(text_dim → hidden_dim) so
            the text K/V can attend in hidden_dim space alongside video.
        num_actions: Number of action steps to predict (12 = 3 latent frames * 4 per frame).
        action_dim: Action dimensionality (7 for RoboCasa: ee_pos*3, ee_rot*3, gripper).
        num_heads: Number of attention heads for both cross-attns.
        mlp_ratio: MLP hidden dim = hidden_dim // mlp_ratio.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        text_dim: int = 1024,
        num_actions: int = 12,
        action_dim: int = 7,
        num_heads: int = 8,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim

        # Learnable action query tokens
        self.action_queries = nn.Parameter(torch.randn(1, num_actions, hidden_dim) * 0.02)

        # Pre-norm for queries (shared input to both cross-attns)
        self.q_norm = nn.LayerNorm(hidden_dim)

        # DiT's crossattn_proj output is text_dim (1024 for 2B); the action head's
        # cross-attentions operate at hidden_dim (2048). Project text up so both
        # streams live in the same space when they hit MultiheadAttention.
        if text_dim != hidden_dim:
            self.text_in_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_in_proj = nn.Identity()

        # Two parallel cross-attentions
        self.video_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.text_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )

        # Fuse the two attended streams (concat along feature dim, project back to D)
        self.fuse_norm = nn.LayerNorm(hidden_dim * 2)
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output projection
        mlp_hidden = hidden_dim // mlp_ratio
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, action_dim),
        )

        # Zero-init output layer for stable training start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor, text_ctx: torch.Tensor) -> torch.Tensor:
        """Predict actions from DiT hidden states + DiT-projected text.

        Args:
            hidden_states: (B, L_v, hidden_dim) flattened DiT features from last block.
                L_v = T * H * W (e.g., 4*16*16 = 1024 for 256x256 input).
            text_ctx: (B, L_t, text_dim) DiT-projected text context (output of
                self.net.crossattn_proj on data_batch["t5_text_embeddings"]).
                For Cosmos 2B text_dim=1024; we project up to hidden_dim before attn.
                Mandatory — no default; missing arg raises TypeError.

        Returns:
            actions: (B, num_actions, action_dim) predicted actions.
        """
        assert text_ctx.dim() == 3 and text_ctx.shape[-1] == self.text_dim, (
            f"text_ctx must be (B, L_t, {self.text_dim}); got {tuple(text_ctx.shape)}"
        )
        assert hidden_states.shape[-1] == self.hidden_dim, (
            f"hidden_states last dim must be {self.hidden_dim}; got {hidden_states.shape[-1]}"
        )
        B = hidden_states.shape[0]
        q = self.action_queries.expand(B, -1, -1)  # (B, num_actions, D)
        qn = self.q_norm(q)

        # Project text into hidden_dim so it lives in the same space as video features.
        text_ctx = self.text_in_proj(text_ctx)  # (B, L_t, hidden_dim)

        # Two parallel cross-attentions: each modality gets its own softmax.
        q_v, _ = self.video_xattn(qn, hidden_states, hidden_states)
        q_t, _ = self.text_xattn(qn, text_ctx, text_ctx)

        # Concat outputs along feature dim → fuse → residual back to q.
        fused = self.fuse_proj(self.fuse_norm(torch.cat([q_v, q_t], dim=-1)))
        q = q + fused

        # Project to action space
        actions = self.mlp(self.norm(q))
        return actions
