# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JointActionVideoDiT: Joint video + action DiT for World Action Model (WAM).

Extends MiniTrainDIT (Cosmos Predict2.5-2B backbone) to jointly denoise video
and action tokens using DreamZero-style blockwise causal attention.

Architecture (DDP mode — no FSDP, direct submodule access):
  For each transformer block:
    1. AdaLN modulation (video: per-frame from block; action: from predicted frames)
    2. Self-attention with blockwise causal masking (video + action in same attention)
    3. Cross-attention (video only — text conditioning)
    4. MLP (video + action through shared MLP)

Token sequence per latent frame block:
  [cond_video(H*W)] [frame_1_video(H*W) | action_1(N_a)] [frame_2 | action_2] ...

Blockwise causal attention:
  - Cond frame (frame 0): self-attends only
  - Video at frame i (i>0): attends to cond + all video <= i + action at frame i
  - Action at frame i: attends to cond + all video <= i + own action tokens
"""

from typing import Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, VideoSize


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.activation(self.fc1(x)))))


class JointActionVideoDiT(MiniTrainDIT):
    """
    Joint video + action DiT with blockwise causal attention (DDP mode).

    New parameters (beyond MiniTrainDIT):
        state_dim: Proprioceptive state dimension for conditioning (default: 16)
        action_dim_for_denoising: Per-step action dimension (default: 7)
        num_action_tokens_per_latent_frame: Action tokens per latent frame (default: 4)
        timestep_scale: Scale factor for timesteps (default: 1.0)
    """

    def __init__(self, *args, timestep_scale: float = 1.0, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # condition mask channel

        state_dim = kwargs.pop("state_dim", 16)
        action_dim_for_denoising = kwargs.pop("action_dim_for_denoising", 7)
        num_action_tokens_per_latent_frame = kwargs.pop("num_action_tokens_per_latent_frame", 4)
        zero_init_action_decoder = kwargs.pop("zero_init_action_decoder", True)
        use_flex_attention = kwargs.pop("use_flex_attention", False)

        # Skip action module reinit (use when loading a trained WAM checkpoint
        # without training state, e.g. adding LM head to a baseline run)
        skip_action_reinit = kwargs.pop("skip_action_reinit", False)

        # Language diffusion tokens (disabled by default for backward compat)
        num_language_tokens = kwargs.pop("num_language_tokens", 0)
        language_embed_dim = kwargs.pop("language_embed_dim", 3584)
        language_vocab_size = kwargs.pop("language_vocab_size", 152064)

        # Remove kwargs the parent doesn't accept
        for k in ("action_dim", "temporal_compression_ratio", "num_action_per_chunk",
                   "hidden_dim_in_action_embedder"):
            kwargs.pop(k, None)

        self.timestep_scale = timestep_scale
        log.info(f"JointActionVideoDiT: timestep_scale={timestep_scale}, "
                 f"state_dim={state_dim}, action_dim={action_dim_for_denoising}, "
                 f"tokens_per_frame={num_action_tokens_per_latent_frame}")

        super().__init__(*args, **kwargs)

        self.state_dim = state_dim
        self.action_dim_for_denoising = action_dim_for_denoising
        self.num_action_tokens_per_latent_frame = num_action_tokens_per_latent_frame
        D = self.model_channels

        # State conditioning → timestep modulation
        self.state_embedder_D = Mlp(state_dim, D * 4, D,
                                    act_layer=lambda: nn.GELU(approximate="tanh"))
        self.state_embedder_3D = Mlp(state_dim, D * 4, D * 3,
                                     act_layer=lambda: nn.GELU(approximate="tanh"))

        # Action token encoder/decoder
        self.action_token_encoder = nn.Sequential(
            nn.Linear(action_dim_for_denoising, D), nn.SiLU(), nn.Linear(D, D))
        self.action_token_decoder = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, action_dim_for_denoising))
        if zero_init_action_decoder:
            # Small random init (like DreamZero's 0.02*randn) instead of exact zeros.
            # Zero init traps the decoder at v_pred=0 (random baseline loss ~2.0).
            nn.init.normal_(self.action_token_decoder[-1].weight, std=0.02)
            nn.init.zeros_(self.action_token_decoder[-1].bias)

        self._action_reinit_checked = False  # Check once on first training forward
        self._skip_action_reinit = skip_action_reinit

        # Action LayerNorms (shared across blocks)
        self.action_ln_sa = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.action_ln_mlp = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)

        # Language diffusion tokens — only LayerNorms live here (no learnable params, FSDP-safe).
        # The embedding/projection modules live on the model (JointActionVideoModel) to avoid
        # FSDP DTensor wrapping conflicts. Language tokens enter forward() already in D_model space.
        self.num_language_tokens = num_language_tokens
        if num_language_tokens > 0:
            self.lang_ln_sa = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
            self.lang_ln_mlp = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
            log.info(f"Language diffusion: {num_language_tokens} tokens (encoder/decoder on model)")

        # Flex attention (fused blockwise causal kernel, opt-in)
        self.use_flex_attention = use_flex_attention
        self._flex_block_mask = None
        self._flex_mask_key = None
        self._compiled_flex_attention = None
        if use_flex_attention:
            log.info("JointActionVideoDiT: flex_attention enabled for blockwise causal attention")

    def fully_shard(self, mesh):
        """Override: do NOT shard individual blocks.

        On GB200 (189GB), keeping all 2B params (~4GB bf16) unsharded is fine.
        This lets _joint_block_forward access block submodules (adaln, self_attn,
        cross_attn, mlp) directly without triggering DTensor mixed-type errors.

        Only the top-level fully_shard (called by the parent model's build_net)
        wraps the whole network, which is sufficient for DCP checkpoint loading.

        NOTE: This causes ~2.8x slowdown vs per-block sharding (FSDP can't overlap
        comm/compute). TODO: fix DTensor compatibility in _joint_block_forward to
        enable per-block sharding.
        """
        from torch.distributed._composable.fsdp import fully_shard as _fully_shard

        # Shard non-block modules normally
        _fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False)
        _fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=False)
        if self.extra_per_block_abs_pos_emb:
            _fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=False)
        if self.extra_image_context_dim is not None:
            _fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False)

        # WAM modules
        _fully_shard(self.state_embedder_D, mesh=mesh, reshard_after_forward=False)
        _fully_shard(self.state_embedder_3D, mesh=mesh, reshard_after_forward=False)
        _fully_shard(self.action_token_encoder, mesh=mesh, reshard_after_forward=False)
        _fully_shard(self.action_token_decoder, mesh=mesh, reshard_after_forward=False)

        # Language diffusion: no parameterized modules in the network to shard.
        # Embedding/projection live on the model (outside FSDP).
        # Only lang_ln_sa/lang_ln_mlp are here (elementwise_affine=False, no params).

        # IMPORTANT: blocks are NOT individually sharded.
        # They will be sharded as part of the top-level fully_shard() call,
        # which keeps all their parameters available during forward().

    def reinit_action_modules(self):
        """Re-initialize action encoder/decoder after checkpoint loading.

        The pretrain checkpoint (DreamDojo) contains action modules saved with
        degenerate weights (zeros). This overwrites our desired init when
        load_state_dict(strict=False) is called. Re-init restores proper
        random initialization for fresh action learning.
        """
        D = self.model_channels
        for layer in self.action_token_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.action_token_decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # Small random init for decoder output (like DreamZero's 0.02*randn)
        nn.init.normal_(self.action_token_decoder[-1].weight, std=0.02)
        nn.init.zeros_(self.action_token_decoder[-1].bias)
        # State embedders also may be overwritten
        for layer in [self.state_embedder_D, self.state_embedder_3D]:
            for m in layer.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
        log.info("[WAM] Re-initialized action encoder/decoder/state_embedder after checkpoint load")
        self._action_modules_need_reinit = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        noisy_action_for_denoising: Optional[torch.Tensor] = None,
        noisy_language_tokens: Optional[torch.Tensor] = None,
        timesteps_action: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,  # ignored (conditioner compat)
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        del kwargs

        # Re-init action modules only on fresh start (iteration 0), not on resume.
        # The pretrain checkpoint (DreamDojo) saves degenerate action weights that
        # overwrite our init. The parent model propagates _start_iteration from the trainer.
        if not getattr(self, '_action_reinit_checked', True) and self.training:
            self._action_reinit_checked = True
            if getattr(self, '_skip_action_reinit', False):
                log.info("[WAM] skip_action_reinit=True — keeping action weights from checkpoint")
            else:
                _start_iter = getattr(self, '_start_iteration', None)
                if _start_iter == 0:
                    log.info(f"[WAM] Fresh start (iteration=0) — reinitializing action modules")
                    self.reinit_action_modules()
                elif _start_iter is not None:
                    log.info(f"[WAM] Resuming from iteration={_start_iter} — keeping learned action weights")
                else:
                    log.warning("[WAM] _start_iteration not set — skipping action reinit")

        # --- Condition mask ---
        if data_type == DataType.VIDEO:
            if condition_video_input_mask_B_C_T_H_W is None:
                B_, _, T_, H_, W_ = x_B_C_T_H_W.shape
                condition_video_input_mask_B_C_T_H_W = torch.zeros(
                    B_, 1, T_, H_, W_, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            elif condition_video_input_mask_B_C_T_H_W.shape[2:] != x_B_C_T_H_W.shape[2:]:
                log.warning(
                    f"[WAM] condition mask shape {condition_video_input_mask_B_C_T_H_W.shape} vs "
                    f"video shape {x_B_C_T_H_W.shape} — resizing mask to match"
                )
                B_, _, T_x, H_x, W_x = x_B_C_T_H_W.shape
                mask = torch.zeros(B_, 1, T_x, H_x, W_x, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
                T_mask = condition_video_input_mask_B_C_T_H_W.shape[2]
                T_copy = min(T_mask, T_x)
                mask[:, :, :T_copy] = condition_video_input_mask_B_C_T_H_W[:, :, :T_copy, :H_x, :W_x]
                condition_video_input_mask_B_C_T_H_W = mask
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B_, _, T_, H_, W_ = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros(
                B_, 1, T_, H_, W_, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1)

        timesteps_B_T = timesteps_B_T * self.timestep_scale

        # --- Patchify video ---
        x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        B, T, H, W, D = x_B_T_H_W_D.shape
        num_predicted = T - 1
        frame_seqlen = H * W

        # --- Cross-attention context ---
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        if img_context_emb is not None and self.extra_image_context_dim is not None:
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        # --- Timestep + state conditioning ---
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb, adaln_lora = self.t_embedder(timesteps_B_T)

            if state is not None:
                sf = state[:, 0, :] if state.ndim == 3 else state
                t_emb = t_emb + self.state_embedder_D(sf).unsqueeze(1)
                adaln_lora = adaln_lora + self.state_embedder_3D(sf).unsqueeze(1)

            t_emb = self.t_embedding_norm(t_emb)

        # Ensure float32 for blocks/final_layer that assert dtype (TE RMSNorm may return bf16)
        if self.use_wan_fp32_strategy:
            t_emb = t_emb.float()
            adaln_lora = adaln_lora.float()

        # --- Action-specific timestep embedding (decoupled noise) ---
        if timesteps_action is not None:
            with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                _ts_a = timesteps_action * self.timestep_scale
                if _ts_a.ndim == 1:
                    _ts_a = _ts_a.unsqueeze(1)
                t_emb_action, adaln_lora_action = self.t_embedder(_ts_a)
                if state is not None:
                    sf = state[:, 0, :] if state.ndim == 3 else state
                    t_emb_action = t_emb_action + self.state_embedder_D(sf).unsqueeze(1)
                    adaln_lora_action = adaln_lora_action + self.state_embedder_3D(sf).unsqueeze(1)
                t_emb_action = self.t_embedding_norm(t_emb_action)
            if self.use_wan_fp32_strategy:
                t_emb_action = t_emb_action.float()
                adaln_lora_action = adaln_lora_action.float()
        else:
            # Inference or coupled mode: action uses same timestep as video
            t_emb_action = t_emb
            adaln_lora_action = adaln_lora

        # --- Encode action tokens ---
        has_action = noisy_action_for_denoising is not None
        action_tokens = None  # (B, num_predicted, N_a, D)
        N_a = self.num_action_tokens_per_latent_frame
        if has_action:
            action_per_frame = noisy_action_for_denoising.reshape(
                B, num_predicted, N_a, self.action_dim_for_denoising)
            action_tokens = self.action_token_encoder(action_per_frame)

        # --- Language diffusion tokens (already in D_model space from model layer) ---
        has_language = (self.num_language_tokens > 0
                        and noisy_language_tokens is not None)
        lang_tokens = noisy_language_tokens if has_language else None  # (B, N_l, D)

        # --- Transformer blocks with blockwise causal joint attention ---
        for block in self.blocks:
            x_B_T_H_W_D, action_tokens, lang_tokens = self._joint_block_forward(
                block, x_B_T_H_W_D, action_tokens, lang_tokens,
                t_emb, adaln_lora, context_input, rope_emb, extra_pos_emb,
                t_emb_action, adaln_lora_action)

        # --- Cache hidden states for language head (if active) ---
        if getattr(self, '_cache_language_context', False) and has_action:
            self._last_video_context = x_B_T_H_W_D.detach()
            self._last_action_context = action_tokens.detach()
            self._language_context_valid = True

        # --- Video output ---
        video_out = self.unpatchify(
            self.final_layer(x_B_T_H_W_D, t_emb, adaln_lora_B_T_3D=adaln_lora))

        # --- Action output ---
        if has_action:
            action_pred = self.action_token_decoder(action_tokens)
            action_out = action_pred.reshape(B, -1, self.action_dim_for_denoising)
            # --- Language output (return velocity in D_model space) ---
            if has_language:
                return video_out, action_out, lang_tokens
            return video_out, action_out
        return video_out

    # ------------------------------------------------------------------
    # Per-block forward with blockwise causal joint attention
    # ------------------------------------------------------------------
    def _joint_block_forward(self, block, x_B_T_H_W_D, action_tokens, lang_tokens,
                             t_emb, adaln_lora, crossattn_emb, rope_emb, extra_pos_emb,
                             t_emb_action=None, adaln_lora_action=None):
        """One DiT block with blockwise causal joint attention."""

        if extra_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_pos_emb

        B, T, H, W, D = x_B_T_H_W_D.shape
        has_action = action_tokens is not None
        has_language = lang_tokens is not None
        num_predicted = T - 1
        N_a = self.num_action_tokens_per_latent_frame if has_action else 0
        N_l = self.num_language_tokens if has_language else 0

        # Action/language use their own timestep embedding (decoupled noise)
        _t_emb_act = t_emb_action if t_emb_action is not None else t_emb
        _adaln_act = adaln_lora_action if adaln_lora_action is not None else adaln_lora

        # ============ AdaLN modulation ============
        with amp.autocast("cuda", enabled=block.use_wan_fp32_strategy, dtype=torch.float32):
            if block.use_adaln_lora:
                shift_sa, scale_sa, gate_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, dim=-1)
                shift_ff, scale_ff, gate_ff = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, dim=-1)
                # Action-specific AdaLN (conditioned on action timestep, not video)
                shift_sa_act, scale_sa_act, gate_sa_act = (block.adaln_modulation_self_attn(_t_emb_act) + _adaln_act).chunk(3, dim=-1)
                shift_ff_act, scale_ff_act, gate_ff_act = (block.adaln_modulation_mlp(_t_emb_act) + _adaln_act).chunk(3, dim=-1)
            else:
                shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(t_emb).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(t_emb).chunk(3, dim=-1)
                shift_ff, scale_ff, gate_ff = block.adaln_modulation_mlp(t_emb).chunk(3, dim=-1)
                shift_sa_act, scale_sa_act, gate_sa_act = block.adaln_modulation_self_attn(_t_emb_act).chunk(3, dim=-1)
                shift_ff_act, scale_ff_act, gate_ff_act = block.adaln_modulation_mlp(_t_emb_act).chunk(3, dim=-1)

        def _expand_5d(v):
            return rearrange(v, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        # ============ Self-Attention: blockwise causal joint ============
        # -- Normalize video --
        norm_v = block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + _expand_5d(scale_sa)) + _expand_5d(shift_sa)
        norm_v_flat = rearrange(norm_v, "b t h w d -> b (t h w) d")

        # -- Video Q/K/V with RoPE --
        q_v, k_v, v_v = block.self_attn.compute_qkv(norm_v_flat, context=None, rope_emb=rope_emb)

        # -- Language Q/K/V (global, no RoPE) --
        q_l = k_l = v_l = None
        if has_language:
            shift_sa_l = shift_sa_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            scale_sa_l = scale_sa_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            gate_sa_l = gate_sa_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            norm_l = self.lang_ln_sa(lang_tokens) * (1 + scale_sa_l) + shift_sa_l
            q_l, k_l, v_l = block.self_attn.compute_qkv(norm_l, context=None, rope_emb=None)

        if has_action:
            # -- Normalize action (broadcast action timestep modulation to all predicted frames) --
            shift_sa_a = shift_sa_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            scale_sa_a = scale_sa_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            gate_sa_a = gate_sa_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            norm_a = self.action_ln_sa(action_tokens) * (1 + scale_sa_a) + shift_sa_a
            norm_a_flat = rearrange(norm_a, "b t n d -> b (t n) d")

            # -- Action Q/K/V (no RoPE) --
            q_a, k_a, v_a = block.self_attn.compute_qkv(norm_a_flat, context=None, rope_emb=None)

            # -- Blockwise causal attention (video + action + language) --
            if self.use_flex_attention:
                out_v_flat, out_a_flat, out_l = self._flex_blockwise_causal_attention(
                    q_v, k_v, v_v, q_a, k_a, v_a, T, H, W, N_a,
                    q_l, k_l, v_l, N_l)
            else:
                out_v_flat, out_a_flat, out_l = self._blockwise_causal_attention(
                    q_v, k_v, v_v, q_a, k_a, v_a, T, H, W, N_a,
                    q_l, k_l, v_l, N_l)

            # -- Project outputs --
            out_v_flat = block.self_attn.output_proj(out_v_flat)
            out_a_flat = block.self_attn.output_proj(out_a_flat)

            # -- Residual + gate --
            x_B_T_H_W_D = x_B_T_H_W_D + _expand_5d(gate_sa) * rearrange(
                out_v_flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            action_tokens = action_tokens + gate_sa_a * rearrange(
                out_a_flat, "b (t n) d -> b t n d", t=num_predicted, n=N_a)

            # -- Language residual + gate --
            if has_language:
                out_l = block.self_attn.output_proj(out_l)
                lang_tokens = lang_tokens + gate_sa_l * out_l
        else:
            # -- Standard full self-attention (no action tokens) --
            out_v = block.self_attn.compute_attention(
                q_v, k_v, v_v, video_size=VideoSize(T=T, H=H, W=W))
            x_B_T_H_W_D = x_B_T_H_W_D + _expand_5d(gate_sa) * rearrange(
                out_v, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

        # ============ Cross-Attention (video only — language skips this) ============
        norm_ca = block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + _expand_5d(scale_ca)) + _expand_5d(shift_ca)
        out_ca = rearrange(
            block.cross_attn(rearrange(norm_ca, "b t h w d -> b (t h w) d"),
                             crossattn_emb, rope_emb=rope_emb),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + _expand_5d(gate_ca) * out_ca

        # ============ MLP (video + action + language) ============
        norm_ff = block.layer_norm_mlp(x_B_T_H_W_D) * (1 + _expand_5d(scale_ff)) + _expand_5d(shift_ff)
        x_B_T_H_W_D = x_B_T_H_W_D + _expand_5d(gate_ff) * block.mlp(norm_ff)

        if has_action:
            shift_ff_a = shift_ff_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            scale_ff_a = scale_ff_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            gate_ff_a = gate_ff_act[:, -1:, None, :].expand(-1, num_predicted, N_a, -1).type_as(action_tokens)
            norm_a_ff = self.action_ln_mlp(action_tokens) * (1 + scale_ff_a) + shift_ff_a
            action_tokens = action_tokens + gate_ff_a * block.mlp(norm_a_ff)

        if has_language:
            shift_ff_l = shift_ff_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            scale_ff_l = scale_ff_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            gate_ff_l = gate_ff_act[:, -1:, :].expand(-1, N_l, -1).type_as(lang_tokens)
            norm_l_ff = self.lang_ln_mlp(lang_tokens) * (1 + scale_ff_l) + shift_ff_l
            lang_tokens = lang_tokens + gate_ff_l * block.mlp(norm_l_ff)

        return x_B_T_H_W_D, action_tokens, lang_tokens

    # ------------------------------------------------------------------
    # Flex attention: fused blockwise causal kernel (opt-in)
    # ------------------------------------------------------------------
    def _build_wam_block_mask(self, T: int, H: int, W: int, N_a: int, N_l: int,
                              device: torch.device) -> BlockMask:
        """Build and cache a BlockMask for WAM blockwise causal attention.

        Token layout: [video(T*S) | action((T-1)*N_a) | language(N_l) | padding]
        """
        key = (T, H, W, N_a, N_l, device)
        if self._flex_block_mask is not None and self._flex_mask_key == key:
            return self._flex_block_mask

        S = H * W
        video_end = T * S
        action_start = video_end
        action_end = action_start + (T - 1) * N_a
        lang_start = action_end
        lang_end = lang_start + N_l
        total_len = lang_end
        padded_len = ((total_len + 127) // 128) * 128

        def wam_mask(b, h, q_idx, kv_idx):
            # Token type classification
            q_is_video = q_idx < video_end
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_lang = (q_idx >= lang_start) & (q_idx < lang_end)

            kv_is_video = kv_idx < video_end
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_lang = (kv_idx >= lang_start) & (kv_idx < lang_end)

            # Frame group: video -> idx // S, action -> (idx - action_start) // N_a + 1
            q_video_frame = q_idx // S
            kv_video_frame = kv_idx // S
            kv_action_frame = (kv_idx - action_start) // N_a + 1

            # Frame for video/action queries
            q_frame = torch.where(q_is_video, q_video_frame,
                                  (q_idx - action_start) // N_a + 1)

            # Video/action query: attend to cond + video<=frame + action_frame + language
            va_rule = ((kv_is_video & (kv_video_frame <= q_frame))
                       | (kv_is_action & (kv_action_frame == q_frame))
                       | kv_is_lang)

            # Language query: attend to all real tokens
            lang_rule = kv_is_video | kv_is_action | kv_is_lang

            # Combine: video/action -> va_rule, language -> lang_rule, padding -> False
            return torch.where(q_is_lang, lang_rule,
                   torch.where(q_is_video | q_is_action, va_rule,
                               torch.tensor(False, device=q_idx.device)))

        block_mask = create_block_mask(
            wam_mask, B=None, H=None,
            Q_LEN=padded_len, KV_LEN=padded_len,
            _compile=True, device=device,
        )

        self._flex_block_mask = block_mask
        self._flex_mask_key = key
        log.info(f"[WAM flex_attention] Built BlockMask: T={T}, S={S}, N_a={N_a}, N_l={N_l}, "
                 f"total={total_len}, padded={padded_len}")
        return block_mask

    def _flex_blockwise_causal_attention(self, q_v, k_v, v_v, q_a, k_a, v_a, T, H, W, N_a,
                                         q_l=None, k_l=None, v_l=None, N_l=0):
        """Fused blockwise causal attention via flex_attention (single kernel per block)."""
        B, _, nh, hd = q_v.shape
        S = H * W
        num_pred = T - 1

        # Build or retrieve cached BlockMask
        block_mask = self._build_wam_block_mask(T, H, W, N_a, N_l, q_v.device)

        # Concatenate: [video | action | language] in (B, seq, nh, hd)
        parts = [q_v, q_a]
        k_parts = [k_v, k_a]
        v_parts = [v_v, v_a]
        has_lang = N_l > 0 and q_l is not None
        if has_lang:
            parts.append(q_l)
            k_parts.append(k_l)
            v_parts.append(v_l)
        q_cat = torch.cat(parts, dim=1)
        k_cat = torch.cat(k_parts, dim=1)
        v_cat = torch.cat(v_parts, dim=1)

        total_len = q_cat.shape[1]

        # Transpose to (B, nh, seq, hd) for flex_attention
        q_cat = q_cat.transpose(1, 2)
        k_cat = k_cat.transpose(1, 2)
        v_cat = v_cat.transpose(1, 2)

        # Pad to match BlockMask padded_len
        padded_len = ((total_len + 127) // 128) * 128
        pad_len = padded_len - total_len
        if pad_len > 0:
            pad = torch.zeros(B, nh, pad_len, hd, device=q_cat.device, dtype=q_cat.dtype)
            q_cat = torch.cat([q_cat, pad], dim=2)
            k_cat = torch.cat([k_cat, pad], dim=2)
            v_cat = torch.cat([v_cat, pad], dim=2)

        # Compiled flex_attention (cached)
        if self._compiled_flex_attention is None:
            self._compiled_flex_attention = torch.compile(flex_attention)
        out = self._compiled_flex_attention(query=q_cat, key=k_cat, value=v_cat,
                                            block_mask=block_mask)

        # Unpad and transpose back to (B, seq, nh*hd)
        out = out[:, :, :total_len].transpose(1, 2)
        D = nh * hd

        # Slice into video, action, language
        video_len = T * S
        action_len = num_pred * N_a
        out_video = out[:, :video_len].reshape(B, video_len, D)
        out_action = out[:, video_len:video_len + action_len].reshape(B, action_len, D)
        out_language = None
        if has_lang:
            out_language = out[:, video_len + action_len:video_len + action_len + N_l].reshape(B, N_l, D)

        return out_video, out_action, out_language

    # ------------------------------------------------------------------
    # Blockwise causal attention kernel (loop-based, default)
    # ------------------------------------------------------------------
    def _blockwise_causal_attention(self, q_v, k_v, v_v, q_a, k_a, v_a, T, H, W, N_a,
                                    q_l=None, k_l=None, v_l=None, N_l=0):
        """
        Blockwise causal joint attention (DreamZero-style) with optional language tokens.

        q/k/v_video:    (B, T*H*W, n_heads, head_dim)
        q/k/v_action:   (B, (T-1)*N_a, n_heads, head_dim)
        q/k/v_language: (B, N_l, n_heads, head_dim)  — global attention

        Returns:
            out_video:    (B, T*H*W, n_heads*head_dim)
            out_action:   (B, (T-1)*N_a, n_heads*head_dim)
            out_language: (B, N_l, n_heads*head_dim) or None
        """
        B, _, nh, hd = q_v.shape
        S = H * W
        num_pred = T - 1

        # -> (B, nh, seq, hd) for F.scaled_dot_product_attention
        qv, kv, vv = (t.transpose(1, 2) for t in (q_v, k_v, v_v))
        qa, ka, va = (t.transpose(1, 2) for t in (q_a, k_a, v_a))

        has_lang = N_l > 0 and q_l is not None
        if has_lang:
            ql, kl, vl = (t.transpose(1, 2) for t in (q_l, k_l, v_l))

        out_v = torch.empty_like(qv)
        out_a = torch.empty_like(qa)

        # Frame 0 (conditioning): self-attention + language KV
        if has_lang:
            k0 = torch.cat([kv[:, :, :S], kl], dim=2)
            v0 = torch.cat([vv[:, :, :S], vl], dim=2)
        else:
            k0, v0 = kv[:, :, :S], vv[:, :, :S]
        out_v[:, :, :S] = F.scaled_dot_product_attention(qv[:, :, :S], k0, v0)

        # Predicted frames: blockwise causal + language
        for i in range(num_pred):
            vs = (i + 1) * S          # video start
            ve = (i + 2) * S          # video end
            a_s = i * N_a             # action start
            a_e = (i + 1) * N_a      # action end

            # KV context: cond + video[0..i+1] + action[i] + all language
            k_ctx = torch.cat([kv[:, :, :ve], ka[:, :, a_s:a_e]], dim=2)
            v_ctx = torch.cat([vv[:, :, :ve], va[:, :, a_s:a_e]], dim=2)
            if has_lang:
                k_ctx = torch.cat([k_ctx, kl], dim=2)
                v_ctx = torch.cat([v_ctx, vl], dim=2)

            out_v[:, :, vs:ve] = F.scaled_dot_product_attention(
                qv[:, :, vs:ve], k_ctx, v_ctx)
            out_a[:, :, a_s:a_e] = F.scaled_dot_product_attention(
                qa[:, :, a_s:a_e], k_ctx, v_ctx)

        D = nh * hd
        out_video = out_v.transpose(1, 2).reshape(B, T * S, D)
        out_action = out_a.transpose(1, 2).reshape(B, num_pred * N_a, D)

        # Language: global attention over all video + action + language
        out_language = None
        if has_lang:
            k_all = torch.cat([kv, ka, kl], dim=2)
            v_all = torch.cat([vv, va, vl], dim=2)
            out_l = F.scaled_dot_product_attention(ql, k_all, v_all)
            out_language = out_l.transpose(1, 2).reshape(B, N_l, D)

        return out_video, out_action, out_language
