# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JointFlowActionExpert: pi05-style joint blockwise attention.

Same interface as FlowActionExpert (K decoder layers + per-layer DiT taps +
shared text K/V) but each layer runs ONE softmax over the joint sequence
[video_kv ; text_kv ; action_q] rather than three separate attentions
(self / video-xattn / text-xattn).

This lets the model trade off "look at video vs look at text vs attend to
other action positions" within a single normalized softmax — matching the
joint blockwise attention used in pi0/pi05 (openpi gemma_pytorch.py:158-238
where prefix and suffix Q/K/V are concatenated for one attention call).

Differences vs FlowActionExpert (per-layer):
    FlowActionExpert: q ← q + SelfAttn(q)
                      q ← q + VideoXAttn(q, video_kv)
                      q ← q + TextXAttn(q, text_kv)
                      q ← q + FFN(q)        # 4 modulation sites
    JointFlowActionExpert:
                      q ← q + JointAttn(q, [video_kv; text_kv; q])
                      q ← q + FFN(q)        # 2 modulation sites

Forward signature matches FlowActionExpert so the model dispatcher can route
to either with a single config flag (`use_joint_flow_action_expert`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Helpers — same as flow_action_expert (sinusoidal time embedding + Beta sampler).
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device=None,
) -> torch.Tensor:
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
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_t, beta_t).sample((bsize,))


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _JointFlowExpertLayer(nn.Module):
    """One joint-attention layer.

    Action queries get AdaLN modulation (carries flow time t); video and text
    K/V are LN'd but NOT modulated (they don't depend on t). Two modulation
    sites per layer: joint-attn input (modulates action q only) and FFN input.

    Out-proj is zero-init so the residual is identity at start — the expert
    behaves like a pass-through until time conditioning + projections are
    learned, matching FlowActionExpert's stable-start strategy.

    No mask is used: action queries see all video tokens, all text tokens,
    and all action chunk positions (full bidirectional within the chunk —
    the chunk is predicted in parallel, not autoregressively).
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Action side: norm + Q,K,V projections (K,V used for action self-attention slot).
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_action = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_action = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Video side: norm + K,V projections (per-layer, fed by DiT block taps).
        self.norm_video = nn.LayerNorm(hidden_dim)
        self.k_video = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_video = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Text side: norm + K,V projections (per-layer K,V; text ctx itself shared across layers).
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
        text_kv: torch.Tensor,
        scale_shifts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        s_at, b_at, s_ff, b_ff = scale_shifts

        # ---- Joint attention: 1 softmax over [video_kv; text_kv; action_self] ----
        q_n = _modulate(self.norm_q(q), s_at, b_at)
        v_n = self.norm_video(video_kv)
        t_n = self.norm_text(text_kv)

        Q = self._split_heads(self.q_proj(q_n))
        Kv = self._split_heads(self.k_video(v_n))
        Vv = self._split_heads(self.v_video(v_n))
        Kt = self._split_heads(self.k_text(t_n))
        Vt = self._split_heads(self.v_text(t_n))
        Kq = self._split_heads(self.k_action(q_n))
        Vq = self._split_heads(self.v_action(q_n))

        # Concat along the SEQUENCE dim (dim=2 after split_heads)
        K = torch.cat([Kv, Kt, Kq], dim=2)
        V = torch.cat([Vv, Vt, Vq], dim=2)

        # Single softmax. No mask — all action positions see everything.
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)
        q = q + self.out_proj(self._merge_heads(attn))

        # ---- FFN ----
        h = _modulate(self.norm_ffn(q), s_ff, b_ff)
        q = q + self.ffn(h)
        return q


class JointFlowActionExpert(nn.Module):
    """Joint-attention variant of FlowActionExpert.

    Same constructor arguments and forward signature as FlowActionExpert so it
    is a drop-in replacement at the model construction site.

    Args:
        hidden_dim: working dim (e.g. 2048 for cosmos 2B).
        text_dim: input dim of DiT-projected text (e.g. 1024 for cosmos 2B).
        num_actions: number of action steps (e.g. 12).
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

        # Text in-projection (shared once across layers; per-layer K,V projections
        # are inside _JointFlowExpertLayer).
        if text_dim != hidden_dim:
            self.text_in_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_in_proj = nn.Identity()

        # Time MLP. Smaller output than FlowActionExpert because we have 2 modulation
        # sites per layer instead of 4 — out_dim halves (2 * 2 * K * D vs 2 * 4 * K * D).
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
            _JointFlowExpertLayer(hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

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
            video_hiddens: list of K tensors, each (B, L_v, hidden_dim) — one per layer.
            text_ctx: (B, L_t, text_dim) — DiT's `crossattn_proj` output.
            noisy_action: (B, num_actions, action_dim) — `a_t` in [0, 1] flow time.
            timestep: (B,) continuous time.

        Returns:
            velocity: (B, num_actions, action_dim).
        """
        assert len(video_hiddens) == self.num_layers, (
            f"JointFlowActionExpert has {self.num_layers} layers but caller supplied "
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

        q = self.action_in_proj(noisy_action) + self.action_pos_emb.to(dtype=dtype)
        text_kv = self.text_in_proj(text_ctx).to(dtype=dtype)

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
            q = layer(q, video_hiddens[k].to(dtype=dtype), text_kv, scale_shifts)

        return self.action_proj(self.final_norm(q))
