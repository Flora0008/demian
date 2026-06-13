# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FlowActionExpert: a flow-matching variant of the multi-tap ActionExpert.

Same outer shape as `action_expert.ActionExpert` (K transformer-decoder layers,
per-layer DiT block taps + shared text cross-attn) but:

  - Action *queries* are content-based: a Linear projection of the noisy action
    chunk `a_t` (B, T_a, action_dim) plus a learned per-step positional emb.
  - A continuous timestep `t in [0, 1]` is encoded with the same sinusoidal
    embedding pi0 uses, fed through a 2-layer MLP, and split into per-layer
    AdaLN (scale, shift) pairs for self-attn / video-xattn / text-xattn / FFN
    sub-blocks. (Functional analogue of pi05's adaRMS conditioning.)
  - Output is *velocity* of shape (B, T_a, action_dim), zero-init for stable
    training start.

Trained with rectified-flow loss `MSE(v_pred, eps - a_0)` where
`a_t = sigma * eps + (1 - sigma) * a_0` (cosmos_wam convention).

`create_sinusoidal_pos_embedding` and `sample_beta` are direct copies of the
helpers in openpi `src/openpi/models_pytorch/pi0_pytorch.py:25-49`. They are
self-contained and copied here to avoid taking openpi as a runtime dep.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from openpi pi0_pytorch.py:25-49 — see module docstring).
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device=None,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions, pi0 convention."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be shape (batch_size,)")

    device = device if device is not None else time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device) -> torch.Tensor:
    """Beta(alpha, beta) sampling, pi0's high-noise-skewed default is (1.5, 1.0)."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_t, beta_t).sample((bsize,))


# ---------------------------------------------------------------------------
# Flow-matching expert layer with AdaLN time conditioning.
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # x: (B, L, D); scale/shift: (B, D) → broadcast over L.
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _FlowExpertLayer(nn.Module):
    """Mirrors action_expert._ExpertLayer, adds AdaLN modulation at three points.

    Layer ordering (pre-LN + AdaLN):
        q ← q + SelfAttn( mod(LN(q), scale_sa, shift_sa) )
        q ← q + VideoXAttn( mod(LN(q), scale_xv, shift_xv), video_kv )
        q ← q + TextXAttn ( mod(LN(q), scale_xt, shift_xt), text_kv  )
        q ← q + FFN(       mod(LN(q), scale_ff, shift_ff) )

    The four (scale, shift) pairs come from a per-layer slice of the time MLP's
    output, materialized once outside this module.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_video_q = nn.LayerNorm(hidden_dim)
        self.video_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_text_q = nn.LayerNorm(hidden_dim)
        self.text_xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
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
        scale_shifts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s_sa, b_sa, s_xv, b_xv, s_xt, b_xt, s_ff, b_ff = scale_shifts

        h = _modulate(self.norm_self(q), s_sa, b_sa)
        sa_out, _ = self.self_attn(h, h, h)
        q = q + sa_out

        h = _modulate(self.norm_video_q(q), s_xv, b_xv)
        v_out, _ = self.video_xattn(h, video_kv, video_kv)
        q = q + v_out

        h = _modulate(self.norm_text_q(q), s_xt, b_xt)
        t_out, _ = self.text_xattn(h, text_kv, text_kv)
        q = q + t_out

        h = _modulate(self.norm_ffn(q), s_ff, b_ff)
        q = q + self.ffn(h)
        return q


# ---------------------------------------------------------------------------
# FlowActionExpert.
# ---------------------------------------------------------------------------

class FlowActionExpert(nn.Module):
    """Flow-matching action expert reading per-layer DiT taps + projected text.

    Args:
        hidden_dim: working dim (e.g. 2048 for cosmos 2B).
        text_dim: input dim of DiT-projected text (e.g. 1024 for cosmos 2B);
            we project up to hidden_dim once at the front.
        num_actions: number of action steps to predict (e.g. 12).
        action_dim: action dimensionality (e.g. 7 for RoboCasa).
        num_heads: attention heads (used by all three attns).
        mlp_ratio: FFN expansion factor (transformer convention).
        num_layers: K — must match the number of DiT taps the caller passes.
        time_min_period / time_max_period: pi0's defaults (4e-3, 4.0).
    """

    NUM_MODS_PER_LAYER = 4  # self-attn, video-xattn, text-xattn, ffn

    def __init__(
        self,
        hidden_dim: int = 2048,
        text_dim: int = 1024,
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
        self.text_dim = text_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period

        # Action chunk in-projection + learned per-step positional embedding.
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos_emb = nn.Parameter(torch.randn(1, num_actions, hidden_dim) * 0.02)

        # One-time text in-projection (text K/V live in hidden_dim space, shared
        # across all K expert layers).
        if text_dim != hidden_dim:
            self.text_in_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_in_proj = nn.Identity()

        # Time MLP: sinusoidal(hidden_dim) → Linear → SiLU → Linear → SiLU →
        #   Linear → 2 * NUM_MODS_PER_LAYER * num_layers * hidden_dim
        # The factor of 2 is for (scale, shift) per modulation site.
        out_dim = 2 * self.NUM_MODS_PER_LAYER * num_layers * hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero-init the final projection so all (scale, shift) start at 0 →
        # AdaLN modulation is the identity at init, so the expert behaves like
        # a vanilla transformer until time conditioning is learned.
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        # K expert layers.
        self.layers = nn.ModuleList([
            _FlowExpertLayer(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Output head: LayerNorm → Linear, zero-init for "predict zero velocity"
        # at start (stable when bolted onto a pretrained DiT).
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_proj.weight)
        nn.init.zeros_(self.action_proj.bias)

    def forward(
        self,
        video_hiddens: list[torch.Tensor],
        text_ctx: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_hiddens: list of K tensors, each (B, L_v, hidden_dim), one per
                expert layer (caller flattens (B, T, H, W, D) → (B, L_v, D)).
            text_ctx: (B, L_t, text_dim) — DiT's `crossattn_proj` output.
            noisy_action: (B, num_actions, action_dim) — `a_t` in [0, 1] time.
            timestep: (B,) continuous time in [0, 1].

        Returns:
            velocity: (B, num_actions, action_dim).
        """
        assert len(video_hiddens) == self.num_layers, (
            f"FlowActionExpert has {self.num_layers} layers but caller supplied "
            f"{len(video_hiddens)} video hiddens"
        )
        assert text_ctx.dim() == 3 and text_ctx.shape[-1] == self.text_dim, (
            f"text_ctx must be (B, L_t, {self.text_dim}); got {tuple(text_ctx.shape)}"
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

        # Queries: project noisy actions, add per-step positional embedding.
        q = self.action_in_proj(noisy_action) + self.action_pos_emb.to(dtype=dtype)

        # Text K/V (shared across layers).
        text_kv = self.text_in_proj(text_ctx).to(dtype=dtype)

        # Time embedding → all per-layer scale/shift pairs.
        time_emb = create_sinusoidal_pos_embedding(
            timestep.to(torch.float32),
            self.hidden_dim,
            min_period=self.time_min_period,
            max_period=self.time_max_period,
            device=timestep.device,
        ).to(dtype=dtype)
        all_mods = self.time_mlp(time_emb)  # (B, 2 * 4 * K * D)
        # Reshape → (K, 8, B, D); chunk[k] = (s_sa, b_sa, s_xv, b_xv, s_xt, b_xt, s_ff, b_ff).
        mods = all_mods.view(B, self.num_layers, self.NUM_MODS_PER_LAYER * 2, self.hidden_dim)
        mods = mods.permute(1, 2, 0, 3).contiguous()  # (K, 8, B, D)

        for k, layer in enumerate(self.layers):
            scale_shifts = tuple(mods[k][i] for i in range(self.NUM_MODS_PER_LAYER * 2))
            q = layer(q, video_hiddens[k].to(dtype=dtype), text_kv, scale_shifts)

        return self.action_proj(self.final_norm(q))  # (B, num_actions, action_dim)
