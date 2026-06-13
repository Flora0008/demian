"""
ARLanguageHead: Autoregressive language decoder for Cosmos WAM.

Takes post-block hidden states from JointActionVideoDiT (video + action tokens),
pools them spatially, projects to Qwen embedding space, and generates text
autoregressively using the first 4 layers of Qwen2.5-VL-7B-Instruct.

Uses prefix-token conditioning: pooled DiT context is prepended as prefix tokens,
then standard causal self-attention over [context | instruction | caption].
CE loss computed on caption portion only.

Adapted from DreamZero's ARLanguageHead (umT5-XXL cross-attention variant).
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ARLanguageHeadConfig:
    hidden_dim: int = 3584          # Qwen2.5-VL-7B hidden_size
    num_layers: int = 4             # first 4 of Qwen's 28 layers
    context_dim: int = 2048         # JointActionVideoDiT model_channels
    vocab_size: int = 152064        # Qwen2.5-VL-7B vocab
    max_gen_len: int = 128
    pool_h: int = 8                 # spatial pooling target height
    pool_w: int = 8                 # spatial pooling target width
    qwen_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Training stage: 1 = context_proj only, 2 = context_proj + LoRA
    lm_training_stage: int = 1
    lm_lora_rank: int = 16
    lm_lora_alpha: int = 16


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around an existing nn.Linear.

    Output = original(x) + B(A(x)) * scaling, where scaling = alpha / rank.
    B is zero-initialized so LoRA contribution starts at zero.
    """

    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original = original
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


class ARLanguageHead(nn.Module):
    """
    AR language decoder using Qwen2.5-VL-7B decoder layers with prefix conditioning.

    Architecture:
        context_pool (adaptive spatial pooling per frame)
        → context_proj (MLP 2048→3584→3584)
        → [context | instruction_embed | caption_embed]
        → Qwen decoder layers (causal self-attention)
        → lm_head → vocab logits

    Training:  forward(video_ctx, action_ctx, instruction_ids, caption_ids) → CE loss
    Inference: generate(video_ctx, action_ctx, instruction_ids) → token ids
    """

    def __init__(self, config: ARLanguageHeadConfig):
        super().__init__()
        self.config = config

        # Project DiT hidden states (2048) to Qwen dimension (3584)
        # Last layer zero-initialized so context starts invisible to the frozen
        # decoder (same principle as ControlNet).  Gradients can then teach
        # context_proj to produce useful embeddings incrementally.
        self.context_proj = nn.Sequential(
            nn.Linear(config.context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        nn.init.zeros_(self.context_proj[2].weight)
        nn.init.zeros_(self.context_proj[2].bias)

        # Placeholder — replaced by from_pretrained_qwen
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Decoder layers — will be populated from Qwen
        self.decoder_layers = nn.ModuleList()
        self.decoder_norm = None

        # Qwen2.5 has no dedicated pad token. Use EOS (151645) for padding
        # and as ignore_index in CE loss, so padded positions don't contribute.
        self.pad_token_id = 151645  # Qwen2.5 eos_token_id

    @classmethod
    def from_pretrained_qwen(cls, config: ARLanguageHeadConfig, qwen_path: str):
        """
        Initialize ARLanguageHead from a pretrained Qwen2.5-VL-7B checkpoint.

        Extracts:
          - model.embed_tokens → token_embed
          - model.layers[:num_layers] → decoder_layers
          - model.norm → decoder_norm
          - lm_head → lm_head (weight-tied with embed_tokens)
        """
        from transformers import AutoModel

        head = cls(config)

        logger.info(f"Loading Qwen from {qwen_path} for ARLanguageHead init...")
        # Qwen2.5-VL is a VL model — use AutoModel to load, then access the
        # language model backbone via .model (Qwen2_5_VLModel wraps Qwen2Model)
        full_model = AutoModel.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Navigate the VL wrapper to get the text backbone
        # Qwen2_5_VLModel has .embed_tokens, .layers, .norm at top level
        if hasattr(full_model, 'language_model'):
            # Some VL wrappers nest as .language_model.model
            lm = full_model.language_model.model if hasattr(full_model.language_model, 'model') else full_model.language_model
        elif hasattr(full_model, 'embed_tokens'):
            # Direct access (Qwen2_5_VLModel exposes these directly)
            lm = full_model
        elif hasattr(full_model, 'model'):
            lm = full_model.model
        else:
            raise RuntimeError(f"Cannot find LM backbone in {type(full_model)}")

        # 1. Token embedding
        head.token_embed = lm.embed_tokens
        # 2. LM head — weight-tie with token embedding
        head.lm_head.weight = head.token_embed.weight
        # 3. First num_layers decoder layers
        head.decoder_layers = nn.ModuleList(
            list(lm.layers[:config.num_layers])
        )
        # 4. Final RMS norm
        head.decoder_norm = lm.norm

        # Update vocab_size to match actual
        actual_vocab_size = head.token_embed.weight.shape[0]
        if actual_vocab_size != config.vocab_size:
            logger.info(
                f"Updating vocab_size from {config.vocab_size} to {actual_vocab_size}"
            )
            config.vocab_size = actual_vocab_size

        del full_model
        logger.info(
            f"ARLanguageHead initialized: {config.num_layers} Qwen layers, "
            f"hidden_dim={config.hidden_dim}, vocab_size={config.vocab_size}"
        )
        return head

    def configure_training_stage(self):
        """Configure requires_grad and LoRA based on lm_training_stage.

        Stage 1: Only context_proj is trainable. All Qwen components frozen.
        Stage 2: context_proj + LoRA on decoder layer self-attention (q/k/v/o_proj).
                 decoder_norm also unfrozen.
        """
        stage = self.config.lm_training_stage

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Unfreeze context_proj (always trainable)
        for p in self.context_proj.parameters():
            p.requires_grad = True

        if stage == 1:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"LM stage 1: context_proj only ({trainable:,} trainable)")

        elif stage == 2:
            self._inject_decoder_lora(
                rank=self.config.lm_lora_rank,
                alpha=self.config.lm_lora_alpha,
            )
            if self.decoder_norm is not None:
                for p in self.decoder_norm.parameters():
                    p.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(
                f"LM stage 2: context_proj + LoRA(rank={self.config.lm_lora_rank}) "
                f"+ decoder_norm ({trainable:,} trainable)"
            )
        else:
            raise ValueError(f"Unknown lm_training_stage: {stage}")

    def _inject_decoder_lora(self, rank: int, alpha: float):
        """Wrap q_proj, k_proj, v_proj, o_proj in each Qwen decoder layer with LoRA."""
        target_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        n_injected = 0

        for layer in self.decoder_layers:
            attn = layer.self_attn
            for name in target_names:
                original = getattr(attn, name)
                setattr(attn, name, LoRALinear(original, rank, alpha))
                n_injected += 1

        logger.info(
            f"Injected LoRA (rank={rank}) into {n_injected} projections "
            f"across {len(self.decoder_layers)} layers"
        )

    def _pool_context(
        self,
        video_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Pool video spatially and concatenate with action tokens.

        Args:
            video_ctx: (B, T, H, W, D) — post-block video hidden states
            action_ctx: (B, num_pred, N_a, D) — post-block action tokens

        Returns:
            context: (B, N_ctx, D) where N_ctx = T*pool_h*pool_w + num_pred*N_a
        """
        B, T, H, W, D = video_ctx.shape
        ph, pw = self.config.pool_h, self.config.pool_w

        # Spatial pooling per frame: (B*T, D, H, W) → (B*T, D, ph, pw)
        v = video_ctx.reshape(B * T, H, W, D).permute(0, 3, 1, 2)  # (B*T, D, H, W)
        v = F.adaptive_avg_pool2d(v, (ph, pw))  # (B*T, D, ph, pw)
        v = v.permute(0, 2, 3, 1).reshape(B, T * ph * pw, D)  # (B, T*ph*pw, D)

        # Flatten action tokens: (B, num_pred, N_a, D) → (B, num_pred*N_a, D)
        a = action_ctx.reshape(B, -1, D)

        return torch.cat([v, a], dim=1)  # (B, N_ctx, D)

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Create causal attention mask [1, 1, seq_len, seq_len]."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def _compute_rope_embeddings(self, position_ids, hidden_states):
        """Compute RoPE cos/sin embeddings for Qwen2.5-VL decoder layers.

        Qwen2.5-VL uses 3D RoPE with position_ids shape (3, B, seq_len) for
        temporal/height/width axes. For text-only decoding we replicate the
        same 1D positions across all 3 axes.
        """
        rotary_emb = self.decoder_layers[0].self_attn.rotary_emb
        # position_ids is (B, T) — expand to (3, B, T) for 3D RoPE
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)  # (3, B, T)
        cos, sin = rotary_emb(hidden_states, pos_3d)
        return cos, sin

    def _decoder_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run Qwen decoder layers with causal self-attention.

        Args:
            hidden_states: [B, T, hidden_dim] — concatenated [context | tokens]

        Returns:
            hidden_states: [B, T, hidden_dim]
        """
        B, T, _ = hidden_states.shape

        # Causal mask: [B, 1, T, T]
        causal_mask = self._make_causal_mask(T, hidden_states.device, hidden_states.dtype)

        # Position ids and RoPE embeddings
        position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        position_embeddings = self._compute_rope_embeddings(position_ids, hidden_states)

        for layer in self.decoder_layers:
            layer_out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = layer_out[0]

        if self.decoder_norm is not None:
            hidden_states = self.decoder_norm(hidden_states)

        return hidden_states

    def forward(
        self,
        video_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        instruction_ids: torch.Tensor,
        caption_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass. Returns CE loss on caption tokens only.

        Args:
            video_ctx: [B, T, H, W, 2048] post-block video hidden states (detached)
            action_ctx: [B, num_pred, N_a, 2048] post-block action tokens (detached)
            instruction_ids: [B, L_instr] tokenized task instruction
            caption_ids: [B, L_caption] tokenized caption

        Returns:
            loss: scalar CE loss averaged over caption tokens
        """
        B = video_ctx.shape[0]
        L_instr = instruction_ids.shape[1]

        param_dtype = next(self.context_proj.parameters()).dtype

        # 1. Pool and project context
        context = self._pool_context(video_ctx, action_ctx)  # (B, N_ctx, 2048)
        context = self.context_proj(context.to(dtype=param_dtype))  # (B, N_ctx, 3584)
        N_ctx = context.shape[1]

        # 2. Embed instruction and caption tokens
        # Decoder input: [instruction | caption[:-1]] (teacher forcing)
        decoder_input_ids = torch.cat(
            [instruction_ids, caption_ids[:, :-1]], dim=1
        )
        token_embeds = self.token_embed(decoder_input_ids)  # (B, L_instr+L_cap-1, 3584)

        # 3. Concatenate: [context | instruction | caption[:-1]]
        hidden_states = torch.cat([context, token_embeds], dim=1)

        # 4. Run decoder
        hidden_states = self._decoder_forward(hidden_states)

        # 5. Take logits only for the token portion (skip context prefix)
        token_logits = hidden_states[:, N_ctx:]  # (B, L_instr+L_cap-1, hidden_dim)
        logits = self.lm_head(token_logits)  # (B, L_instr+L_cap-1, vocab_size)

        # 6. Build targets: [ignore × L_instr | caption[1:]]
        ignore = torch.full(
            (B, L_instr), self.pad_token_id,
            device=caption_ids.device, dtype=caption_ids.dtype,
        )
        targets = torch.cat([ignore, caption_ids[:, 1:]], dim=1)

        # 7. CE loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        video_ctx: torch.Tensor,
        action_ctx: torch.Tensor,
        instruction_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        eos_token_id: int = 151645,  # Qwen2.5 EOS
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.

        Args:
            video_ctx: [B, T, H, W, 2048] post-block video hidden states
            action_ctx: [B, num_pred, N_a, 2048] post-block action tokens
            instruction_ids: [B, L_instr] tokenized task instruction

        Returns:
            generated_ids: [B, num_generated] caption token ids
        """
        B = video_ctx.shape[0]
        device = video_ctx.device
        param_dtype = next(self.context_proj.parameters()).dtype

        # Pool and project context once
        context = self._pool_context(video_ctx, action_ctx)
        context = self.context_proj(context.to(dtype=param_dtype))
        N_ctx = context.shape[1]

        # Build initial sequence: [context | instruction | start_token]
        instr_embeds = self.token_embed(instruction_ids)
        start_token = torch.full((B, 1), self.pad_token_id, dtype=torch.long, device=device)
        start_embeds = self.token_embed(start_token)

        hidden_input = torch.cat([context, instr_embeds, start_embeds], dim=1)

        all_generated = []

        for _ in range(max_new_tokens):
            hidden_states = self._decoder_forward(hidden_input)

            # Logits at last position
            next_logits = self.lm_head(hidden_states[:, -1:])  # (B, 1, vocab)
            next_logits = next_logits.squeeze(1)  # (B, vocab)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            all_generated.append(next_token)

            if (next_token == eos_token_id).all():
                break

            # Append next token embedding
            next_embeds = self.token_embed(next_token)
            hidden_input = torch.cat([hidden_input, next_embeds], dim=1)

        if all_generated:
            return torch.cat(all_generated, dim=1)
        return torch.empty(B, 0, dtype=torch.long, device=device)
