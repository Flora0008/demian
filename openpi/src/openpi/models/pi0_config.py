"""Pi0/Pi05 model configuration dataclass.

NOTE (DeMiAn release): the original JAX-only methods on ``Pi0Config`` —
``create`` (builds the flax ``Pi0``), ``inputs_spec`` (jax ShapeDtypeStruct
specs), and the nnx-based ``get_freeze_filter`` — have been replaced with
JAX-free stubs. The actual model used in this release is the PyTorch
``openpi.models_pytorch.pi0_pytorch.PI0Pytorch``, instantiated from this config
by ``scripts/train_pytorch.py``. The dataclass fields (including
``lm_loss_weight``, the DeMiAn-VLA auxiliary LM loss) are unchanged.
"""

import dataclasses

from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma


@dataclasses.dataclass(frozen=True)
class ActionExpertMoEConfig:
    """Optional sparse MoE replacement for the action expert feed-forward blocks."""

    num_experts: int = 4
    top_k: int = 2
    load_balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 1e-3
    router_jitter_noise: float = 0.0

    def __post_init__(self):
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.load_balance_loss_weight < 0:
            raise ValueError("load_balance_loss_weight must be >= 0")
        if self.router_z_loss_weight < 0:
            raise ValueError("router_z_loss_weight must be >= 0")
        if self.router_jitter_noise < 0:
            raise ValueError("router_jitter_noise must be >= 0")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_moe: ActionExpertMoEConfig | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # Weight for autoregressive language model loss on prefix tokens (0.0 = disabled).
    lm_loss_weight: float = 0.0
    # LM-only mode: skip flow matching / action loss entirely, train only the LM head.
    lm_only: bool = False
    # Use causal (prefix-LM) attention for reasoning tokens. When False, reasoning
    # tokens attend bidirectionally (same as task/image tokens).
    causal_reasoning: bool = True
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Inference-time torch.compile mode passthrough. Saved by newer training runs
    # and read at eval time; pi.py sets this via dataclasses.replace.
    pytorch_compile_mode: str | None = None

    # ---- Auxiliary supervision for primitive-aware training (Option-D ablations) ----
    # Physics regression head: predicts 12-dim normalized physics features (cumulative
    # displacement, net rotation, gripper transition, z-bias, path curvature, mean
    # speed) from the action expert's per-step hidden states. 0 disables.
    physics_aux_weight: float = 0.0
    # 12-character binary mask string controlling which physics dims contribute to the
    # auxiliary loss. "1" = dim included, "0" = dim's residual zeroed out (no gradient
    # through that dim). Default is all-ones. Used for per-dim group ablations.
    # Order: [disp_x, disp_y, disp_z, rot_rx, rot_ry, rot_rz, grip_none, grip_close,
    #        grip_open, z_bias, curvature, mean_speed].
    physics_dim_mask: str = "111111111111"
    # Optional clamp applied only to the action chunk used to derive physics-aux
    # targets. This is useful for quantile-normalized datasets where rare action
    # outliers can become very large normalized targets and destabilize the aux
    # gradient. None preserves historical behavior.
    physics_aux_action_clip: float | None = None
    # If true, physics targets are computed from raw pre-normalization actions
    # preserved by the data loader. The main action loss still uses normalized
    # actions.
    physics_aux_use_raw_actions: bool = False
    # Optional 12-dim scale override for physics targets. Useful when raw action
    # units differ across benchmarks (for example, meters/radians vs controller
    # normalized units).
    physics_aux_target_scale: tuple[float, ...] | None = None
    # Which deterministic action-summary target to compute. "robocasa_ee"
    # preserves the original dx/dy/dz/drx/dry/drz/gripper convention.
    # "molmobot_joint_delta" uses 7 cumulative joint deltas + gripper transition
    # + joint-space curvature and speed.
    physics_aux_action_layout: str = "robocasa_ee"
    # Pairwise physics regression head: predicts interval-level motion geometry
    # for every sub-trajectory inside the action chunk. 0 disables.
    pair_physics_aux_weight: float = 0.0
    # Minimum interval length for pairwise physics supervision. Boundary pair
    # (i, j) contributes to the pair loss only when j - i >= this value.
    pair_physics_min_gap: int = 1
    # Latent VAE bottleneck on pooled action-expert hidden states. Adds a small
    # Gaussian VAE that reconstructs the action chunk from a low-dim z. d_z=0 disables.
    latent_vae_dim: int = 0
    latent_vae_kl_weight: float = 0.0
    latent_vae_recon_weight: float = 1.0
    # Next-state regression head: predicts the (z-score-normalized) robot state at
    # frame+action_horizon from the action expert's per-step hidden states. 0 disables.
    next_state_aux_weight: float = 0.0
    # Robot state dim for next_state_aux. Robocasa = 16. Override per benchmark.
    next_state_aux_dim: int = 16

    # ---- Two-stage latent aux (motion-physics VAE) ----
    # When > 0, attach a head that predicts a Stage-1-pretrained latent target
    # encoded by a frozen MotionPhysicsVAE. See docs/two_stage_latent_aux_plan.md.
    latent_aux_weight: float = 0.0
    # Dimension of the Stage-1 VAE's latent (must match the checkpoint).
    latent_aux_dim: int = 16
    # Path to a Stage-1 VAE state_dict (loaded as a frozen target encoder).
    # Empty string disables the latent aux even if latent_aux_weight > 0.
    latent_aux_vae_path: str = ""
    # MotionPhysicsVAE hidden width — must match Stage-1.
    latent_aux_vae_hidden: int = 512
    # MotionPhysicsVAE state / action / horizon dims — typically inherited from
    # the dataset; overridden per-benchmark in TrainConfig.
    latent_aux_state_dim: int = 16
    latent_aux_action_dim: int = 7

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng):
        # JAX flax Pi0 model removed in this release. Use the PyTorch model
        # openpi.models_pytorch.pi0_pytorch.PI0Pytorch (see scripts/train_pytorch.py).
        raise NotImplementedError(
            "JAX Pi0 is not included in this release; use openpi.models_pytorch.pi0_pytorch.PI0Pytorch."
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        raise NotImplementedError(
            "inputs_spec relied on JAX (jax.ShapeDtypeStruct) and is not used by the PyTorch training path."
        )

    def get_freeze_filter(self):
        """Freeze filter (nnx-based) is JAX-trainer machinery and unused by the
        PyTorch trainer, which manages parameter freezing directly. Returns None."""
        return None
