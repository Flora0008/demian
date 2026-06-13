# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JointActionVideoModel: Joint video + action rectified flow model for World Action Model (WAM).

Extends ActionVideo2WorldModelRectifiedFlow to jointly denoise video latents and
action tokens via flow matching. Both video and action share the same ODE solver
during inference, following the DreamZero architecture.

Training loss: L = L_video * 1.0 + L_action * action_loss_weight
Both losses use rectified flow weighting: w(sigma) = (1+sigma)^2 / sigma^2
"""

from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.text2world_model import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
    NUM_CONDITIONAL_FRAMES_KEY,
)


@attrs.define(slots=False)
class JointActionVideoModelConfig(Video2WorldModelRectifiedFlowConfig):
    """Configuration for joint video + action WAM model."""
    action_loss_weight: float = 1.0
    decouple_video_action_noise: bool = False
    action_dim_for_denoising: int = 7
    num_action_tokens_per_latent_frame: int = 4
    state_dim: int = 16
    # v2 training improvements
    loss_scale: float = 1.0  # Multiply total loss (cosmos-policy uses 10.0)
    action_loss_weight_mode: str = "uniform"  # "uniform" or "gaussian_bell" (DreamZero-style)
    # AR Language head (optional, default off for backward compatibility)
    language_head_enabled: bool = False
    language_head_config: Optional[dict] = None
    lm_loss_weight: float = 0.1
    lm_pretrain_only: bool = False  # When True, freeze DiT+WAM, only train LM head
    # Language diffusion (flow matching on caption embeddings, alternative to AR head)
    language_diffusion_enabled: bool = False
    num_language_tokens: int = 64
    language_loss_weight: float = 0.1
    disable_text_cross_attention: bool = False  # When True, skip Qwen text encoder; language diffusion tokens carry all language info


class JointActionVideoModel(ActionVideo2WorldModelRectifiedFlow):
    """
    Joint video + action rectified flow model.

    Overrides the parent model to:
    1. Add noise to both video and action during training
    2. Pass noisy action tokens to the network for joint denoising
    3. Compute joint loss (video MSE + action MSE with rectified flow weighting)
    4. Joint ODE solving during inference
    5. (Optional) Language head for autoregressive captioning loss
    """

    def set_up_model(self):
        super().set_up_model()

        # Language diffusion encoder/decoder
        if getattr(self.config, 'language_diffusion_enabled', False):
            import torch.nn as _nn
            D = self.net.model_channels
            embed_dim = 3584  # Qwen embedding dim
            vocab = 152064
            self._lang_diff_embed = _nn.Embedding(vocab, embed_dim)
            self._lang_diff_embed.weight.requires_grad = False
            self._lang_diff_proj_in = _nn.Sequential(
                _nn.Linear(embed_dim, D), _nn.SiLU(), _nn.Linear(D, D))
            _nn.init.zeros_(self._lang_diff_proj_in[-1].weight)
            _nn.init.zeros_(self._lang_diff_proj_in[-1].bias)
            self._lang_diff_proj_out = _nn.Linear(D, embed_dim)
            # Dedicated optimizer: the main FSDP optimizer only steps DTensor
            # (FSDP-managed) params. These non-FSDP modules need their own optimizer.
            self._lang_diff_optim = torch.optim.AdamW(
                list(self._lang_diff_proj_in.parameters())
                + list(self._lang_diff_proj_out.parameters()),
                lr=1e-4, weight_decay=0.1,
            )
        else:
            self._lang_diff_embed = None
            self._lang_diff_proj_in = None
            self._lang_diff_proj_out = None

        # Language head (optional, gated by config)
        self.language_head = None
        if getattr(self.config, 'language_head_enabled', False) and self.config.language_head_config:
            from cosmos_predict2._src.predict2.action.networks.ar_language_head import (
                ARLanguageHead, ARLanguageHeadConfig,
            )
            lh_cfg = ARLanguageHeadConfig(**(self.config.language_head_config or {}))
            if lh_cfg.qwen_path:
                self.language_head = ARLanguageHead.from_pretrained_qwen(lh_cfg, lh_cfg.qwen_path)
            else:
                self.language_head = ARLanguageHead(lh_cfg)
            self.language_head.configure_training_stage()
            self.language_head = self.language_head.cuda().to(torch.bfloat16)
            # Dedicated optimizer: the main FSDP optimizer only steps DTensor
            # (FSDP-managed) params inside self.net. The language_head lives on
            # self (outside FSDP boundary) so it needs its own optimizer.
            _lm_params = [p for p in self.language_head.parameters() if p.requires_grad]
            if _lm_params:
                self._lm_head_optim = torch.optim.AdamW(
                    _lm_params, lr=1e-4, weight_decay=0.1,
                )
            else:
                self._lm_head_optim = None

        # LM pretrain mode: only LM head gets meaningful gradients.
        # Cannot use requires_grad_(False) on DiT because FSDP expects all
        # params to participate in backward. Instead, loss=lm_loss only
        # (set in forward), and context is .detach()-ed so DiT gets zero grads.
        if getattr(self.config, 'lm_pretrain_only', False) and self.language_head is not None:
            from cosmos_predict2._src.imaginaire.utils import log as _log
            lh_trainable = sum(p.numel() for p in self.language_head.parameters() if p.requires_grad)
            _log.info(
                f"[LM Pretrain] loss=lm_only, DiT gets zero grads via detached context, "
                f"LM head trainable ({lh_trainable:,} params)"
            )

    def forward(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Training forward pass with joint video + action denoising.

        Computes text embeddings online (same as parent), but skips the LAM
        computation since action is a denoising target, not a conditioning input.
        """
        # --- Propagate start iteration to network for action module reinit ---
        if hasattr(self, '_start_iteration') and not getattr(self.net, '_action_reinit_checked', True):
            self.net._start_iteration = self._start_iteration

        # --- One-time diagnostic: check if lang_diff params are in the optimizer ---
        if not hasattr(self, '_lang_diff_optim_checked'):
            self._lang_diff_optim_checked = True
            import os as _os
            if int(_os.environ.get("RANK", "0")) == 0 and self._lang_diff_proj_in is not None:
                # Check if proj_in params are registered as modules
                _is_module = '_lang_diff_proj_in' in dict(self.named_modules())
                _n_params = sum(p.numel() for p in self._lang_diff_proj_in.parameters())
                _n_model = sum(p.numel() for p in self.parameters())
                # Check param identity in named_parameters
                _proj_in_last_id = id(self._lang_diff_proj_in[-1].weight)
                _found_in_params = any(id(p) == _proj_in_last_id for p in self.parameters())
                print(f"[WAM-DIAG] lang_diff_proj_in: is_module={_is_module}, "
                      f"n_params={_n_params}, n_model_params={_n_model}, "
                      f"found_in_self_params={_found_in_params}, "
                      f"device={self._lang_diff_proj_in[-1].weight.device}",
                      flush=True)

        # --- Timing instrumentation ---
        import time as _time
        _timings = {}
        _t0 = _time.monotonic()

        # --- Compute text embeddings online (required by conditioner) ---
        _skip_text = getattr(self.config, 'disable_text_cross_attention', False)
        if _skip_text:
            # No text encoder — language info comes from diffusion tokens only.
            # Provide zero embeddings so cross-attention is a no-op.
            B_ = data_batch[self.input_data_key].shape[0]
            data_batch["t5_text_embeddings"] = torch.zeros(
                B_, 1, self.net.model_channels, device="cuda", dtype=torch.bfloat16)
            data_batch["t5_text_mask"] = torch.zeros(B_, 1, device="cuda")
        elif "t5_text_embeddings" in data_batch and data_batch["t5_text_embeddings"] is not None:
            # Skip if pre-computed embeddings are already in the batch (from dataloader)
            text_embeddings = data_batch["t5_text_embeddings"]
            if not isinstance(text_embeddings, torch.Tensor):
                text_embeddings = torch.stack(text_embeddings)
            data_batch["t5_text_embeddings"] = text_embeddings.to(device="cuda")
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )
        elif (
            hasattr(self, 'config')
            and hasattr(self.config, 'text_encoder_config')
            and self.config.text_encoder_config is not None
            and getattr(self.config.text_encoder_config, 'compute_online', False)
        ):
            text_embeddings = self.text_encoder.compute_text_embeddings_online(
                data_batch, self.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )
        torch.cuda.synchronize()
        _timings["text_encoder"] = _time.monotonic() - _t0

        # --- Compute LAM features (required by conditioner for action field) ---
        _t1 = _time.monotonic()
        if hasattr(self, 'lam') and self.lam is not None and "lam_video" in data_batch:
            lam_video = rearrange(data_batch["lam_video"], "b (p t) h w c -> (b p) t h w c", t=2)
            lam_input = {"videos": lam_video}
            with torch.no_grad():
                outputs = self.lam.lam(lam_input)
            latent_action = outputs["z_rep"].squeeze().to(data_batch["action"].dtype).detach()
            latent_action = rearrange(latent_action, "(t b) d -> b t d", t=data_batch["action"].shape[1])
            data_batch["action"][:, :, -32:] = data_batch["action"][:, :, -32:] * latent_action
        torch.cuda.synchronize()
        _timings["lam"] = _time.monotonic() - _t1

        # --- Get video latent and condition ---
        _t2 = _time.monotonic()
        _, x0_video, condition = self.get_data_and_condition(data_batch)
        torch.cuda.synchronize()
        _timings["tokenize_condition"] = _time.monotonic() - _t2

        # --- Extract action and state from data_batch ---
        # The dataset produces a 384D multi-embodiment action vector.
        # RoboCasa 7D actions are at dims 169:176 (ee_pos(3) + ee_rot(3) + gripper(1)).
        action_dim = getattr(self.config, 'action_dim_for_denoising', 7)
        robocasa_action_offset = 169  # RoboCasa slot in multi-embodiment action vector
        full_action = data_batch["action"]  # (B, num_steps, 384)
        if full_action.shape[-1] > action_dim:
            action_target = full_action[:, :, robocasa_action_offset:robocasa_action_offset + action_dim]
        else:
            action_target = full_action  # Already the right dim
        state = data_batch.get("state", None)  # (B, 1, state_dim) or None

        batch_size = x0_video.size()[0]

        # --- Sample noise for video ---
        epsilon_video = torch.randn(x0_video.size(), **self.tensor_kwargs_fp32)
        t_B_video = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B_video = rearrange(t_B_video, "b -> b 1")

        # --- Sample noise for action ---
        epsilon_action = torch.randn_like(action_target)
        if getattr(self.config, "decouple_video_action_noise", False):
            # Decoupled: uniform timesteps for action (spread evenly across all noise levels)
            # instead of logitnormal+shift which biases 78% to high noise (bad for 7D actions)
            t_B_action = self.rectified_flow.sample_uniform_time(batch_size).to(**self.tensor_kwargs_fp32)
            t_B_action = rearrange(t_B_action, "b -> b 1")
        else:
            # Coupled: same timestep for video and action
            t_B_action = t_B_video

        # --- Broadcast for model parallelism ---
        x0_video, condition, epsilon_video, t_B_video = self.broadcast_split_for_model_parallelsim(
            x0_video, condition, epsilon_video, t_B_video
        )

        # --- Video: timesteps, sigmas, interpolation ---
        timesteps_video = self.rectified_flow.get_discrete_timestamp(t_B_video, self.tensor_kwargs_fp32)
        sigmas_video = self.rectified_flow.get_sigmas(timesteps_video, self.tensor_kwargs_fp32)
        timesteps_video = rearrange(timesteps_video, "b -> b 1")
        sigmas_video = rearrange(sigmas_video, "b -> b 1")
        xt_video, vt_video = self.rectified_flow.get_interpolation(epsilon_video, x0_video, sigmas_video)

        # --- Action: timesteps, sigmas, interpolation ---
        timesteps_action = self.rectified_flow.get_discrete_timestamp(t_B_action, self.tensor_kwargs_fp32)
        sigmas_action = self.rectified_flow.get_sigmas(timesteps_action, self.tensor_kwargs_fp32)
        # Expand for broadcasting: (B,) -> (B, 1, 1) to match (B, num_steps, action_dim)
        sigmas_action_exp = sigmas_action.unsqueeze(-1).unsqueeze(-1)
        xt_action = sigmas_action_exp * epsilon_action + (1 - sigmas_action_exp) * action_target
        vt_action = epsilon_action - action_target  # Rectified flow velocity target

        # --- Language diffusion: encode + noise ---
        has_lang_diffusion = getattr(self.config, 'language_diffusion_enabled', False)
        xt_lang = None
        vt_lang = None
        lang_embed_target = None  # Clean 3584D embedding for proj_out reconstruction loss
        lang_valid_mask = None
        if has_lang_diffusion and "lm_caption_ids" in data_batch:
            caption_ids = data_batch["lm_caption_ids"].cuda()  # (B, N_l)
            N_l = getattr(self.config, 'num_language_tokens', 64)
            # Truncate/pad to N_l
            if caption_ids.shape[1] > N_l:
                caption_ids = caption_ids[:, :N_l]
            elif caption_ids.shape[1] < N_l:
                pad = torch.full((batch_size, N_l - caption_ids.shape[1]),
                                 151645, dtype=torch.long, device=caption_ids.device)
                caption_ids = torch.cat([caption_ids, pad], dim=1)
            lang_valid_mask = data_batch.get("lm_caption_valid",
                                             torch.ones(batch_size, dtype=torch.bool)).cuda()
            # Encode: ids → frozen embedding → trainable projection → D_model
            # These modules live on the model (not self.net) to avoid FSDP wrapping.
            with torch.no_grad():
                lang_embed = self._lang_diff_embed(caption_ids)  # (B, N_l, 3584)
            lang_embed_target = lang_embed.detach()  # For proj_out reconstruction loss
            lang_target = self._lang_diff_proj_in(lang_embed)  # (B, N_l, D)
            # Flow matching: noise + interpolate (coupled with action timestep)
            epsilon_lang = torch.randn_like(lang_target)
            sigmas_lang = sigmas_action.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1) for (B, N_l, D)
            xt_lang = sigmas_lang * epsilon_lang + (1 - sigmas_lang) * lang_target
            vt_lang = epsilon_lang - lang_target

        # --- Enable language head context caching ---
        if self.language_head is not None:
            self.net._cache_language_context = True

        # --- Forward pass ---
        _t3 = _time.monotonic()
        _lm_pretrain = getattr(self.config, 'lm_pretrain_only', False)
        # Pass action timestep to network for correct AdaLN conditioning
        timesteps_action_for_net = rearrange(timesteps_action, "b -> b 1")
        result = self.denoise(
            noise=epsilon_video,
            xt_B_C_T_H_W=xt_video.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_video,
            condition=condition,
            state=state,
            noisy_action=xt_action.to(**self.tensor_kwargs),
            noisy_language=xt_lang.to(**self.tensor_kwargs) if xt_lang is not None else None,
            timesteps_action=timesteps_action_for_net,
        )
        torch.cuda.synchronize()
        _timings["denoise_forward"] = _time.monotonic() - _t3

        vt_pred_video = result
        vt_pred_action = None
        vt_pred_lang = None
        if isinstance(result, tuple):
            if len(result) == 3:
                vt_pred_video, vt_pred_action, vt_pred_lang = result
            elif len(result) == 2:
                vt_pred_video, vt_pred_action = result

        # --- Video loss ---
        time_weights_video = self.rectified_flow.train_time_weight(timesteps_video, self.tensor_kwargs_fp32)
        per_instance_video_loss = torch.mean(
            (vt_pred_video.float() - vt_video.float()) ** 2,
            dim=list(range(1, vt_pred_video.dim()))
        )
        video_loss = torch.mean(time_weights_video * per_instance_video_loss)

        # --- Action loss ---
        action_loss = torch.tensor(0.0, device=video_loss.device)
        if vt_pred_action is not None:
            timesteps_action_for_weight = rearrange(timesteps_action, "b -> b 1")
            # Use separate weighting mode for action loss (gaussian_bell upweights mid-range timesteps)
            _action_weight_mode = getattr(self.config, "action_loss_weight_mode", "uniform")
            if _action_weight_mode == "gaussian_bell":
                _num_steps = self.rectified_flow.noise_scheduler.config.num_train_timesteps
                _wts = torch.exp(-2.0 * ((timesteps_action_for_weight - _num_steps / 2.0) / _num_steps) ** 2)
                time_weights_action = _wts / (_wts.mean() + 1e-8)
            else:
                time_weights_action = self.rectified_flow.train_time_weight(
                    timesteps_action_for_weight, self.tensor_kwargs_fp32
                )
            per_instance_action_loss = torch.mean(
                (vt_pred_action.float() - vt_action.float()) ** 2,
                dim=list(range(1, vt_pred_action.dim()))
            )
            action_loss = torch.mean(time_weights_action * per_instance_action_loss)

        # --- Language head loss (optional) ---
        lm_loss = torch.tensor(0.0, device=video_loss.device)
        if self.language_head is not None:
            video_ctx = getattr(self.net, '_last_video_context', None)
            action_ctx = getattr(self.net, '_last_action_context', None)
            if (video_ctx is not None and action_ctx is not None
                    and "lm_instruction_ids" in data_batch
                    and "lm_caption_ids" in data_batch
                    and data_batch.get("lm_caption_valid", torch.tensor(True)).any()):
                valid_mask = data_batch.get("lm_caption_valid", None)
                if valid_mask is not None and not valid_mask.all():
                    # Only compute LM loss for samples with valid captions
                    idx = valid_mask.nonzero(as_tuple=True)[0]
                    lm_loss = self.language_head(
                        video_ctx[idx], action_ctx[idx],
                        data_batch["lm_instruction_ids"][idx].cuda(),
                        data_batch["lm_caption_ids"][idx].cuda(),
                    )
                else:
                    lm_loss = self.language_head(
                        video_ctx, action_ctx,
                        data_batch["lm_instruction_ids"].cuda(),
                        data_batch["lm_caption_ids"].cuda(),
                    )
            # Reset cache
            self.net._cache_language_context = False
            self.net._last_video_context = None
            self.net._last_action_context = None
            self.net._language_context_valid = False

        # --- Language diffusion loss (flow matching on caption embeddings) ---
        lang_diff_loss = torch.tensor(0.0, device=video_loss.device)
        if vt_pred_lang is not None and vt_lang is not None:
            per_instance_lang_loss = torch.mean(
                (vt_pred_lang.float() - vt_lang.float()) ** 2,
                dim=list(range(1, vt_pred_lang.dim()))
            )
            # Mask out invalid captions
            if lang_valid_mask is not None:
                per_instance_lang_loss = per_instance_lang_loss * lang_valid_mask.float()
            # Use same time weighting as action (gaussian_bell if configured)
            timesteps_lang_for_weight = rearrange(timesteps_action, "b -> b 1")
            _action_weight_mode = getattr(self.config, "action_loss_weight_mode", "uniform")
            if _action_weight_mode == "gaussian_bell":
                _num_steps = self.rectified_flow.noise_scheduler.config.num_train_timesteps
                _wts = torch.exp(-2.0 * ((timesteps_lang_for_weight - _num_steps / 2.0) / _num_steps) ** 2)
                time_weights_lang = _wts / (_wts.mean() + 1e-8)
            else:
                time_weights_lang = self.rectified_flow.train_time_weight(
                    timesteps_lang_for_weight, self.tensor_kwargs_fp32)
            lang_diff_loss = torch.mean(time_weights_lang * per_instance_lang_loss)
            # proj_out reconstruction loss: train proj_out to decode D_model → 3584D
            if self._lang_diff_proj_out is not None and lang_embed_target is not None:
                # Use the clean lang_target (D_model) as input, not the noisy prediction
                recon = self._lang_diff_proj_out(lang_target)  # (B, N_l, 3584)
                recon_loss = torch.mean((recon.float() - lang_embed_target.float()) ** 2)
                lang_diff_loss = lang_diff_loss + recon_loss

        # --- Joint loss ---
        action_loss_weight = getattr(self.config, "action_loss_weight", 1.0)
        lm_loss_weight = getattr(self.config, "lm_loss_weight", 0.1)
        lang_diff_weight = getattr(self.config, "language_loss_weight", 0.1)
        _loss_scale = getattr(self.config, "loss_scale", 1.0)
        if _lm_pretrain:
            loss = lm_loss + 0.0 * video_loss
        else:
            loss = (video_loss + action_loss_weight * action_loss
                    + lm_loss_weight * lm_loss + lang_diff_weight * lang_diff_loss)
        loss = loss * _loss_scale

        # Log separate losses for monitoring (use print for reliable slurm output)
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 10 == 1:
            import os as _os
            if int(_os.environ.get("RANK", "0")) == 0:
                extras = ""
                if lm_loss.item() > 0:
                    extras += f" lm={lm_loss.item():.4f}"
                if lang_diff_loss.item() > 0:
                    extras += f" lang_diff={lang_diff_loss.item():.4f}"
                # Log unscaled losses (before loss_scale) for comparability
                # Action module weight norms (track if decoder is being updated)
                _dec = self.net.action_token_decoder
                _enc = self.net.action_token_encoder
                _dec_w = _dec[-1].weight.data.norm().item()
                _enc_w = _enc[0].weight.data.norm().item()
                # Lang diff projection weight norms (if present)
                _ld_in_w = self._lang_diff_proj_in[-1].weight.data.norm().item() if self._lang_diff_proj_in is not None else 0.0
                _ld_out_w = self._lang_diff_proj_out.weight.data.norm().item() if self._lang_diff_proj_out is not None else 0.0
                _ld_in_g = 0.0
                _ld_out_g = 0.0
                if self._lang_diff_proj_in is not None and self._lang_diff_proj_in[-1].weight.grad is not None:
                    _ld_in_g = self._lang_diff_proj_in[-1].weight.grad.norm().item()
                if self._lang_diff_proj_out is not None and self._lang_diff_proj_out.weight.grad is not None:
                    _ld_out_g = self._lang_diff_proj_out.weight.grad.norm().item()
                # LM head weight/gradient norms (if present)
                _lm_w = 0.0
                _lm_g = 0.0
                if self.language_head is not None:
                    _lm_params = [p for p in self.language_head.parameters() if p.requires_grad]
                    if _lm_params:
                        _lm_w = torch.cat([p.data.flatten() for p in _lm_params]).norm().item()
                        _grads = [p.grad.flatten() for p in _lm_params if p.grad is not None]
                        if _grads:
                            _lm_g = torch.cat(_grads).norm().item()
                print(
                    f"[WAM] total={loss.item():.4f} "
                    f"video={video_loss.item():.4f} "
                    f"action={action_loss.item():.4f}{extras} "
                    f"(w_action={action_loss_weight}, loss_scale={_loss_scale}) "
                    f"dec_w={_dec_w:.4f} enc_w={_enc_w:.4f} "
                    f"lm_w={_lm_w:.4f} lm_g={_lm_g:.4f} "
                    f"ld_in={_ld_in_w:.6f}(g={_ld_in_g:.6f}) ld_out={_ld_out_w:.4f}(g={_ld_out_g:.6f})",
                    flush=True,
                )

        torch.cuda.synchronize()
        _timings["loss"] = _time.monotonic() - _t3 - _timings["denoise_forward"]
        _timings["total_forward"] = _time.monotonic() - _t0

        # Log timing breakdown every 100 steps (reduced from 10 to avoid log spam)
        if not hasattr(self, '_fwd_counter'):
            self._fwd_counter = 0
        self._fwd_counter += 1
        if self._fwd_counter % 100 == 1:
            import os as _os
            if int(_os.environ.get("RANK", "0")) == 0:
                parts = " | ".join(f"{k}={v:.3f}s" for k, v in _timings.items())
                print(f"[WAM-TIMING] {parts}", flush=True)

        output_batch = {
            "x0": x0_video,
            "xt": xt_video,
            "sigma": sigmas_video,
            "condition": condition,
            "model_pred": vt_pred_video,
            "edm_loss": loss,
            # Unscaled component losses for wandb logging
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "lm_loss": lm_loss.detach(),
            "lang_diff_loss": lang_diff_loss.detach(),
        }

        return output_batch, loss

    def on_before_zero_grad(self, optimizer, scheduler, iteration):
        """Step dedicated optimizers for non-FSDP params (lang_diff, language_head)."""
        if hasattr(self, '_lang_diff_optim') and self._lang_diff_optim is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self._lang_diff_proj_in.parameters()) + list(self._lang_diff_proj_out.parameters()),
                max_norm=1.0)
            self._lang_diff_optim.step()
            self._lang_diff_optim.zero_grad(set_to_none=True)
        if hasattr(self, '_lm_head_optim') and self._lm_head_optim is not None:
            _lm_params = [p for p in self.language_head.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(_lm_params, max_norm=1.0)
            self._lm_head_optim.step()
            self._lm_head_optim.zero_grad(set_to_none=True)

    def save_auxiliary_state(self, ckpt_dir: str, iteration: int):
        """Save non-FSDP optimizer states alongside DCP checkpoint (rank 0 only)."""
        import os
        if int(os.environ.get("RANK", "0")) != 0:
            return
        aux = {}
        if hasattr(self, '_lm_head_optim') and self._lm_head_optim is not None:
            aux['lm_head_optim'] = self._lm_head_optim.state_dict()
        if hasattr(self, '_lang_diff_optim') and self._lang_diff_optim is not None:
            aux['lang_diff_optim'] = self._lang_diff_optim.state_dict()
        if aux:
            save_path = os.path.join(ckpt_dir, f"iter_{iteration:09}", "aux_optim.pt")
            torch.save(aux, save_path)
            print(f"[aux] Saved auxiliary optimizer state to {save_path}", flush=True)

    def load_auxiliary_state(self, ckpt_dir: str, iteration: int):
        """Restore non-FSDP optimizer states from DCP checkpoint dir."""
        import os
        load_path = os.path.join(ckpt_dir, f"iter_{iteration:09}", "aux_optim.pt")
        if not os.path.exists(load_path):
            return
        aux = torch.load(load_path, map_location="cpu", weights_only=True)
        if 'lm_head_optim' in aux and hasattr(self, '_lm_head_optim') and self._lm_head_optim is not None:
            self._lm_head_optim.load_state_dict(aux['lm_head_optim'])
            print(f"[aux] Restored _lm_head_optim state from {load_path}", flush=True)
        if 'lang_diff_optim' in aux and hasattr(self, '_lang_diff_optim') and self._lang_diff_optim is not None:
            self._lang_diff_optim.load_state_dict(aux['lang_diff_optim'])
            print(f"[aux] Restored _lang_diff_optim state from {load_path}", flush=True)

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
        state: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
        noisy_language: Optional[torch.Tensor] = None,
        timesteps_action: Optional[torch.Tensor] = None,
    ):
        """
        Denoise with joint video + action tokens.

        Extends the parent denoise to pass state and noisy_action to the network.
        """
        # Conditioning frame replacement (same as parent)
        condition_video_mask = None
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
                1, C, 1, 1, 1
            ).type_as(xt_B_C_T_H_W)

            xt_B_C_T_H_W = (
                condition_state_in_B_C_T_H_W * condition_video_mask
                + xt_B_C_T_H_W * (1 - condition_video_mask)
            )

        # Forward pass through network with joint denoising
        condition_dict = condition.to_dict()
        # Ensure all inputs match model dtype (avoids bf16/fp32 mismatch at inference)
        _dtype = self.tensor_kwargs.get("dtype", None)
        if _dtype is not None:
            timesteps_B_T = timesteps_B_T.to(_dtype)
            if state is not None:
                state = state.to(_dtype)
            if noisy_action is not None:
                noisy_action = noisy_action.to(_dtype)
            if noisy_language is not None:
                noisy_language = noisy_language.to(_dtype)
            for _k, _v in condition_dict.items():
                if isinstance(_v, torch.Tensor) and _v.is_floating_point():
                    condition_dict[_k] = _v.to(_dtype)
        _extra_kwargs = {}
        if timesteps_action is not None:
            _ta = timesteps_action.to(_dtype) if _dtype is not None else timesteps_action
            _extra_kwargs["timesteps_action"] = _ta
        net_output = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            state=state,
            noisy_action_for_denoising=noisy_action,
            noisy_language_tokens=noisy_language,
            **condition_dict,
            **_extra_kwargs,
        )

        # Handle tuple output (video, action, [language]) or just video
        action_pred = None
        lang_pred = None
        if isinstance(net_output, tuple):
            if len(net_output) == 3:
                video_pred, action_pred, lang_pred = net_output
            elif len(net_output) == 2:
                video_pred, action_pred = net_output
            else:
                video_pred = net_output[0]
        else:
            video_pred = net_output.float()

        video_pred = video_pred.float()

        # Replace conditioning frame velocities with ground truth
        if condition.is_video and self.config.denoise_replace_gt_frames and condition_video_mask is not None:
            gt_frames_x0 = condition.gt_frames.type_as(video_pred)
            gt_frames_velocity = noise - gt_frames_x0
            video_pred = gt_frames_velocity * condition_video_mask + video_pred * (1 - condition_video_mask)

        if action_pred is not None:
            if lang_pred is not None:
                return video_pred, action_pred, lang_pred
            return video_pred, action_pred
        return video_pred

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
        num_action_steps: int | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Generate video + action samples via joint ODE solving.

        Returns:
            video_latents: (B, C, T, H, W) video latent
            latent_to_save: dict of intermediate latents
            action_pred: (B, num_action_steps, action_dim) predicted actions
        """
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
        noise_video = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape), torch.bfloat16,
            self.tensor_kwargs["device"], seed,
        )

        # Action noise
        action_dim = getattr(self.config, "action_dim_for_denoising", 7)
        n_action_tokens = getattr(self.config, "num_action_tokens_per_latent_frame", 4)
        T_latent = state_shape[1]
        num_predicted = T_latent - 1  # First frame is conditioning
        if num_action_steps is None:
            num_action_steps = num_predicted * n_action_tokens
        noise_action = torch.randn(
            n_sample, num_action_steps, action_dim,
            device=self.tensor_kwargs["device"], dtype=torch.bfloat16,
        )

        # Language noise (for joint denoising if language diffusion is enabled)
        has_lang_diff = getattr(self.config, 'language_diffusion_enabled', False) and self._lang_diff_proj_in is not None
        noise_language = None
        if has_lang_diff:
            N_l = getattr(self.config, 'num_language_tokens', 64)
            D = self.net.model_channels
            noise_language = torch.randn(
                n_sample, N_l, D,
                device=self.tensor_kwargs["device"], dtype=torch.bfloat16,
            )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps, device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )
        timesteps = self.sample_scheduler.timesteps

        # Get velocity function with state
        state = data_batch.get("state", None)
        velocity_fn = self._get_joint_velocity_fn(
            data_batch, guidance, is_negative_prompt, state,
            has_language=has_lang_diff,
        )

        if self.net.is_context_parallel_enabled:
            noise_video = broadcast_split_tensor(
                tensor=noise_video, seq_dim=2,
                process_group=self.get_context_parallel_group(),
            )

        latents_video = noise_video
        latents_action = noise_action
        latents_language = noise_language

        latent_to_save = {}
        timesteps_iter = timesteps if INTERNAL else tqdm.tqdm(
            timesteps, desc="Generating samples", total=len(timesteps)
        )

        for num_step, t in enumerate(timesteps_iter):
            if num_step in query_steps:
                latent_to_save[num_step] = latents_video

            timestep = torch.stack([t]).to(**self.tensor_kwargs)

            vel_result = velocity_fn(
                noise_video, latents_video, timestep.unsqueeze(0),
                noise_action, latents_action,
                noisy_language=latents_language,
            )
            v_video, v_action = vel_result[0], vel_result[1]
            v_language = vel_result[2] if len(vel_result) > 2 else None

            # Step video — keep batch dim consistent
            temp_x0 = self.sample_scheduler.step(
                v_video, t, latents_video,
                return_dict=False, generator=seed_g,
            )[0]
            latents_video = temp_x0

            # Step action (same ODE step as video)
            sigma_t = t / self.sample_scheduler.config.num_train_timesteps
            sigma_next = (
                timesteps[num_step + 1] / self.sample_scheduler.config.num_train_timesteps
                if num_step + 1 < len(timesteps)
                else 0.0
            )
            dt = sigma_next - sigma_t
            latents_action = latents_action + v_action * dt

            # Step language (same ODE as action)
            if v_language is not None and latents_language is not None:
                latents_language = latents_language + v_language * dt

        latent_to_save[num_step] = latents_video

        if self.net.is_context_parallel_enabled:
            latents_video = cat_outputs_cp(
                latents_video, seq_dim=2, cp_group=self.get_context_parallel_group()
            )

        # Decode denoised language tokens back to embedding space
        denoised_lang_embed = None
        if latents_language is not None and self._lang_diff_proj_out is not None:
            denoised_lang_embed = self._lang_diff_proj_out(latents_language)  # (B, N_l, 3584)

        return latents_video, latent_to_save, latents_action, denoised_lang_embed

    def _get_joint_velocity_fn(
        self,
        data_batch: Dict,
        guidance: float,
        is_negative_prompt: bool,
        state: Optional[torch.Tensor],
        has_language: bool = False,
    ) -> Callable:
        """Create a velocity function that returns both video and action velocities."""
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

        for cond in (condition, uncondition):
            cond_updated = cond.set_video_condition(
                gt_frames=x0,
                random_min_num_conditional_frames=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames=self.config.max_num_conditional_frames,
                num_conditional_frames=num_conditional_frames,
                conditional_frames_probs=self.config.conditional_frames_probs,
            )
            if cond is condition:
                condition = cond_updated.edit_for_inference(
                    is_cfg_conditional=True, num_conditional_frames=num_conditional_frames
                )
            else:
                uncondition = cond_updated.edit_for_inference(
                    is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
                )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        def velocity_fn(
            noise_video, noisy_video, timestep,
            noise_action, noisy_action,
            noisy_language=None,
        ):
            _lang_kw = {}
            if has_language and noisy_language is not None:
                _lang_kw["noisy_language"] = noisy_language
                _lang_kw["timesteps_action"] = timestep  # Same timestep for action & language
            # Conditional
            cond_result = self.denoise(
                noise_video, noisy_video, timestep, condition,
                state=state, noisy_action=noisy_action, **_lang_kw,
            )
            # Unconditional
            uncond_result = self.denoise(
                noise_video, noisy_video, timestep, uncondition,
                state=state, noisy_action=noisy_action, **_lang_kw,
            )

            if isinstance(cond_result, tuple):
                cond_v_video = cond_result[0]
                cond_v_action = cond_result[1]
                uncond_v_video = uncond_result[0]
                uncond_v_action = uncond_result[1]
                v_video = cond_v_video + guidance * (cond_v_video - uncond_v_video)
                # No CFG for actions — use pure conditional prediction (matches training
                # and DreamZero). Unconditional action velocity is noise; CFG would amplify it.
                v_action = cond_v_action
                # Language velocity (no CFG — language has no unconditional variant)
                v_lang = None
                if len(cond_result) > 2 and cond_result[2] is not None:
                    v_lang = cond_result[2]
                return v_video, v_action, v_lang
            else:
                v_video = cond_result + guidance * (cond_result - uncond_result)
                return v_video, torch.zeros_like(noisy_action), None

        return velocity_fn
