from typing import Literal

import pytest
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class _GemmaMLPExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dense_mlp: nn.Module | None = None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if dense_mlp is not None:
            self.load_from_dense(dense_mlp)

    @torch.no_grad()
    def load_from_dense(self, dense_mlp: nn.Module):
        self.gate_proj.weight.copy_(dense_mlp.gate_proj.weight)
        self.up_proj.weight.copy_(dense_mlp.up_proj.weight)
        self.down_proj.weight.copy_(dense_mlp.down_proj.weight)

    def forward(self, x: torch.Tensor, act_fn):
        return self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))


class ActionExpertMoEMLP(nn.Module):
    """Sparse token-level MoE for Gemma MLP blocks.

    Expert 0 keeps the dense MLP parameter names (`gate_proj`, `up_proj`, `down_proj`) so dense action expert
    checkpoints can still be loaded into MoE models before the remaining experts are copied from expert 0.
    """

    def __init__(self, dense_mlp: nn.Module, moe_config):
        super().__init__()
        self.config = dense_mlp.config
        self.hidden_size = dense_mlp.hidden_size
        self.intermediate_size = dense_mlp.intermediate_size
        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k
        self.router_jitter_noise = moe_config.router_jitter_noise
        self.load_balance_loss_weight = moe_config.load_balance_loss_weight
        self.router_z_loss_weight = moe_config.router_z_loss_weight

        self.gate_proj = dense_mlp.gate_proj
        self.up_proj = dense_mlp.up_proj
        self.down_proj = dense_mlp.down_proj
        self.act_fn = dense_mlp.act_fn
        self.extra_experts = nn.ModuleList(
            [
                _GemmaMLPExpert(self.hidden_size, self.intermediate_size, dense_mlp)
                for _ in range(self.num_experts - 1)
            ]
        )
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-2)

        self.latest_load_balance_loss = None
        self.latest_router_z_loss = None

    @property
    def is_moe(self) -> bool:
        return True

    @torch.no_grad()
    def initialize_experts_from_dense(self):
        for expert in self.extra_experts:
            expert.load_from_dense(self)

    def _expert_forward(self, expert_idx: int, x: torch.Tensor):
        if expert_idx == 0:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.extra_experts[expert_idx - 1](x, self.act_fn)

    def _router_logits(self, flat_x: torch.Tensor):
        router_x = flat_x
        if self.training and self.router_jitter_noise > 0:
            noise = torch.empty_like(router_x).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )
            router_x = router_x * noise
        return F.linear(router_x.to(self.router.weight.dtype), self.router.weight).float()

    def _update_aux_losses(self, router_logits: torch.Tensor, router_probs: torch.Tensor, selected_experts: torch.Tensor):
        selected_one_hot = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        tokens_per_expert = selected_one_hot.mean(dim=(0, 1))
        prob_per_expert = router_probs.mean(dim=0)
        self.latest_load_balance_loss = self.num_experts * torch.sum(tokens_per_expert * prob_per_expert)
        self.latest_router_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

    def forward(self, x: torch.Tensor):
        input_shape = x.shape
        flat_x = x.reshape(-1, input_shape[-1])

        router_logits = self._router_logits(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        self._update_aux_losses(router_logits, router_probs, selected_experts)

        combined = torch.zeros(flat_x.shape, device=flat_x.device, dtype=torch.float32)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = flat_x.index_select(0, token_idx)
            expert_output = self._expert_forward(expert_idx, expert_input)
            weights = routing_weights[token_idx, slot_idx].to(expert_output.dtype).unsqueeze(-1)
            combined.index_add_(0, token_idx, (expert_output * weights).float())

        return combined.reshape(input_shape).to(x.dtype)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        action_moe_config=None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.action_moe_config = action_moe_config

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        if self.action_moe_config is not None:
            self._replace_action_expert_mlps_with_moe()

        self.to_bfloat16_for_selected_params(precision)

    @property
    def has_action_moe(self) -> bool:
        return self.action_moe_config is not None

    def _replace_action_expert_mlps_with_moe(self):
        for layer in self.gemma_expert.model.layers:
            layer.mlp = ActionExpertMoEMLP(layer.mlp, self.action_moe_config)

    @torch.no_grad()
    def initialize_moe_experts_from_dense(self):
        if not self.has_action_moe:
            return
        for layer in self.gemma_expert.model.layers:
            layer.mlp.initialize_experts_from_dense()

    def moe_aux_loss(self):
        if not self.has_action_moe:
            return None

        load_balance_losses = []
        router_z_losses = []
        for layer in self.gemma_expert.model.layers:
            mlp = layer.mlp
            if mlp.latest_load_balance_loss is not None:
                load_balance_losses.append(mlp.latest_load_balance_loss)
            if mlp.latest_router_z_loss is not None:
                router_z_losses.append(mlp.latest_router_z_loss)

        if not load_balance_losses and not router_z_losses:
            return None

        device = self.gemma_expert.model.layers[0].mlp.router.weight.device
        dtype = torch.float32
        load_balance = (
            torch.stack(load_balance_losses).mean()
            if load_balance_losses
            else torch.zeros((), device=device, dtype=dtype)
        )
        router_z = (
            torch.stack(router_z_losses).mean() if router_z_losses else torch.zeros((), device=device, dtype=dtype)
        )
        return {
            "moe_load_balance_loss": load_balance,
            "moe_router_z_loss": router_z,
            "moe_aux_loss": self.action_moe_config.load_balance_loss_weight * load_balance
            + self.action_moe_config.router_z_loss_weight * router_z,
        }

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
