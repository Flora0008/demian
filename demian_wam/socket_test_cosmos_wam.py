"""
Cosmos WAM Policy Server for RoboCasa evaluation.

Serves the WAM (World Action Model) as a WebSocket policy server.
The server receives observations (3 camera views + state) and returns
predicted actions via joint video + action denoising.

Usage (single-GPU, recommended for eval):
    python socket_test_cosmos_wam.py --port 6000 --single-gpu \
        --checkpoint-path <path>/model_regular.pt --experiment <name>

Usage (multi-GPU FSDP, legacy):
    torchrun --nproc_per_node=N socket_test_cosmos_wam.py \
        --port 6000 --checkpoint-path <dcp_iter_dir> --experiment <name>
"""

import dataclasses
import glob
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import tyro
from einops import rearrange


from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig

logger = logging.getLogger(__name__)

_ORIG_TORCH_FROM_NUMPY = torch.from_numpy


def _torch_from_numpy_compat(array: np.ndarray) -> torch.Tensor:
    """torch.from_numpy fallback for containers where torch was built against NumPy 1.x."""
    try:
        return _ORIG_TORCH_FROM_NUMPY(array)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise

    array = np.ascontiguousarray(array)
    dtype_map = {
        np.dtype("uint8"): torch.uint8,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("bool"): torch.bool,
    }
    torch_dtype = dtype_map.get(array.dtype)
    if torch_dtype is None:
        return torch.tensor(array.tolist())
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch_dtype).reshape(array.shape)


torch.from_numpy = _torch_from_numpy_compat


# torch.Tensor.numpy fallback for the same numpy 1.x/2.x ABI issue. The Reason1
# image-conditioning path (text_encoder.py: frame.permute(...).to(torch.uint8).numpy())
# calls tensor.numpy() which fails with "Numpy is not available" under the broken
# container. Round-trip via .tolist() → np.asarray to bypass the failing C bridge.
_ORIG_TENSOR_NUMPY = torch.Tensor.numpy


def _torch_tensor_numpy_compat(self, *args, **kwargs):
    try:
        return _ORIG_TENSOR_NUMPY(self, *args, **kwargs)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
    # Fall through: rebuild via tolist (slow path) preserving dtype where reasonable.
    cpu = self.detach().cpu()
    dtype_to_np = {
        torch.uint8: np.uint8, torch.int8: np.int8,
        torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64,
        torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
        torch.bool: np.bool_,
    }
    target_np = dtype_to_np.get(cpu.dtype, np.float32)
    return np.asarray(cpu.tolist(), dtype=target_np)


torch.Tensor.numpy = _torch_tensor_numpy_compat


def _torch_uint8_from_numpy(array: np.ndarray) -> torch.Tensor:
    """Create a uint8 tensor without using PyTorch's NumPy C-API bridge."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch.uint8).reshape(array.shape)


def _tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a float32 ndarray without tensor.numpy()."""
    return np.asarray(tensor.detach().float().cpu().tolist(), dtype=np.float32)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a uint8 tensor to bytes without tensor.numpy()."""
    return bytes(tensor.detach().cpu().tolist())


@dataclasses.dataclass
class Args:
    port: int = 6000
    checkpoint_path: str = ""
    """Path to WAM checkpoint. Single-GPU: path to model_regular.pt. Multi-GPU: path to DCP iter dir."""
    experiment: str = "dreamdojo_2b_480_640_robocasa_wam_baseline"
    """Registered experiment name for config lookup."""
    single_gpu: bool = False
    """Use single-GPU mode (no torchrun, no FSDP). Requires a converted .pt checkpoint."""
    save_video: bool = False
    output_dir: str = "runs/wam_server"
    num_steps: int = 35
    guidance: float = 1.5
    shift: float = 5.0
    seed: int = 1
    action_stats_paths: str = ""
    """Comma-separated list of LeRobot dataset paths (or direct stats.json paths).
    When provided, action predictions are de-normalized via inverse min_max
    using aggregated min(min) / max(max) across paths. Inverts the min_max
    transform applied by the training dataloader to action.{end_effector_position,
    end_effector_rotation, gripper_close}. Leave empty to return predictions in
    normalized [-1, 1] space (back-compat with prior eval runs)."""
    flow_inference_num_steps: int = -1
    """Override the model's `flow_inference_num_steps` config field after model
    load (default -1 = no override, keep YAML value). Smaller values reduce
    Euler-ODE noise accumulation at the cost of single-step approximation
    error. Use this to A/B test whether per-step prediction jitter (observed
    in trajectory comparison vs the deterministic head) is from ODE step
    variance vs underlying head capacity. Only applies to flow_action_expert
    / vlm_action_expert heads — deterministic paths ignore it."""
    robocasa_video_mode: str = "grid"
    """RoboCasa camera composition. `grid` matches the default 3-view 2x2-grid
    training/eval path. `left_reason1_right_wrist` matches left-video / Reason1
    right+wrist training: diffusion video is left agentview only, while Reason1
    receives separate right agentview and wrist images."""


# Module-level state: action denormalization stats (min, max) for the 7-D output
# slice [ee_pos(3), ee_rot(3), gripper(1)]. Set once at server startup from
# --action-stats-paths and applied to every model output before returning.
# None → denorm disabled.
_ACTION_DENORM_STATS: Optional[tuple] = None


def _load_aggregate_action_stats(stats_paths: list) -> tuple:
    """Load and aggregate per-dim action min/max for the 7-D output slice.

    For each path: opens `meta/stats.json` (or the file directly if the path
    already ends in stats.json), reads the raw 12-D `action` column's min/max,
    and slices [5:12] per `meta/modality.json` schema:
        base_motion=[0:4], control_mode=[4:5],
        end_effector_position=[5:8], end_effector_rotation=[8:11],
        gripper_close=[11:12]
    Aggregates across paths as element-wise min(min) / max(max). Returns
    (min_7, max_7) as float32 numpy arrays.

    Inverts the min_max transform at
    `groot_dreams/data/transform/state_action.py:141-156`:
        x_norm = 2 * (x - min) / (max - min) - 1
        x      = (x_norm + 1) / 2 * (max - min) + min
    """
    import json

    mins, maxs = [], []
    for p in stats_paths:
        sp = p if p.endswith("stats.json") else os.path.join(p, "meta", "stats.json")
        with open(sp) as f:
            stats = json.load(f)
        full_min = np.asarray(stats["action"]["min"], dtype=np.float32)
        full_max = np.asarray(stats["action"]["max"], dtype=np.float32)
        mins.append(full_min[5:12])
        maxs.append(full_max[5:12])
    agg_min = np.stack(mins).min(axis=0).astype(np.float32)
    agg_max = np.stack(maxs).max(axis=0).astype(np.float32)
    return agg_min, agg_max


def _denorm_action(action_norm: np.ndarray) -> np.ndarray:
    """Apply min_max inverse to action predictions; pass-through if stats unset.

    Inputs in [-1, 1] are mapped back to [min_d, max_d] per dimension d.
    For degenerate dims where max == min (e.g., gripper held at one extreme
    across the entire dataset), output is 0 — matches training-time forward
    behavior at state_action.py:160-161.
    """
    if _ACTION_DENORM_STATS is None:
        return action_norm
    a_min, a_max = _ACTION_DENORM_STATS
    spread = a_max - a_min
    safe = spread > 0
    out = np.zeros_like(action_norm, dtype=np.float32)
    if safe.any():
        out[..., safe] = (action_norm[..., safe] + 1.0) * 0.5 * spread[safe] + a_min[safe]
    return out


def init_distributed():
    """Initialize distributed process group and device mesh (called by torchrun)."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank}/{world_size} on cuda:{rank}")

    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    return mesh, rank, world_size


def _describe_linear(module) -> str:
    in_features = getattr(module, "in_features", "?")
    out_features = getattr(module, "out_features", "?")
    bias = getattr(module, "bias", None) is not None
    return f"{in_features}->{out_features}, bias={bias}"


def _log_crossattn_proj(model: torch.nn.Module, context: str) -> None:
    """Log the concrete cross-attention projection module selected by config."""
    net = getattr(model, "net", None)
    if net is None:
        logger.warning(f"[ARCH {context}] model.net is missing; cannot inspect crossattn_proj")
        return

    use_projection = getattr(net, "use_crossattn_projection", None)
    proj = getattr(net, "crossattn_proj", None)
    if proj is None:
        logger.warning(
            f"[ARCH {context}] model.net.crossattn_proj is missing "
            f"(use_crossattn_projection={use_projection})"
        )
        return

    details = [
        f"type={proj.__class__.__module__}.{proj.__class__.__name__}",
        f"use_crossattn_projection={use_projection}",
    ]
    if hasattr(proj, "out_features"):
        details.append(f"out_features={getattr(proj, 'out_features')}")

    if isinstance(proj, torch.nn.Sequential):
        child_types = [child.__class__.__name__ for child in proj]
        details.append(f"children={child_types}")
        for child in proj:
            if hasattr(child, "in_features") and hasattr(child, "out_features"):
                details.append(f"linear={_describe_linear(child)}")
                break
    else:
        for name in ("in_proj", "gate", "out_proj"):
            child = getattr(proj, name, None)
            if child is not None and hasattr(child, "in_features") and hasattr(child, "out_features"):
                details.append(f"{name}={_describe_linear(child)}")

    logger.info(f"[ARCH {context}] model.net.crossattn_proj: " + "; ".join(details))


def load_wam_model(checkpoint_path: str, experiment: str):
    """Load WAM model from checkpoint using distributed (FSDP) loading.

    Must be called after init_distributed().
    Launch with: torchrun --nproc_per_node=N socket_test_cosmos_wam.py ...
    """
    import importlib
    import torch.distributed.checkpoint as dcp
    from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override

    # Resolve config exactly as training does
    config_path = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
    config_module = get_config_module(config_path)
    config = importlib.import_module(config_module).make_config()
    config_overrides = ["--", f"experiment={experiment}"]
    tokenizer_pth = os.environ.get("TOKENIZER_PTH")
    if tokenizer_pth:
        config_overrides.append(f"+model.config.tokenizer.vae_pth={tokenizer_pth}")
        logger.info(f"Using local tokenizer checkpoint: {tokenizer_pth}")
    config = override(config, config_overrides)

    logger.info(f"Resolved config for experiment={experiment}")

    # Instantiate model — this calls fully_shard() internally, which is fine
    # because torchrun has already set up the distributed environment
    model = instantiate(config.model)
    logger.info(f"Model type: {type(model).__name__}")
    _log_crossattn_proj(model, "distributed-build")

    # Load DCP checkpoint (distributed — each rank loads its shard)
    logger.info(f"Loading DCP checkpoint from {checkpoint_path}")
    model_dir = os.path.join(checkpoint_path, "model")

    from torch.distributed.checkpoint import DefaultLoadPlanner
    state_dict = {"model": model.state_dict()}
    dcp.load(
        state_dict,
        checkpoint_id=model_dir,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    model.load_state_dict(state_dict["model"], strict=False)
    logger.info("DCP checkpoint loaded successfully")
    del state_dict

    model.eval()
    model.cuda()

    # Cast parameters to bf16: for FSDP DTensors, must replace the DTensor itself
    # (simple .data = .data.to(bf16) only updates the local shard, not DTensor metadata)
    from torch.distributed._tensor import DTensor
    with torch.no_grad():
        for mod in model.modules():
            for name, param in mod._parameters.items():
                if param is None or not param.is_floating_point():
                    continue
                if isinstance(param, DTensor):
                    local = param.to_local().to(torch.bfloat16)
                    new_dt = DTensor.from_local(
                        local,
                        device_mesh=param.device_mesh,
                        placements=param.placements,
                        run_check=False,
                    )
                    mod._parameters[name] = torch.nn.Parameter(new_dt, requires_grad=False)
                else:
                    param.data = param.data.to(torch.bfloat16)
            for name, buf in mod._buffers.items():
                if buf is None or not buf.is_floating_point():
                    continue
                if isinstance(buf, DTensor):
                    local = buf.to_local().to(torch.bfloat16)
                    mod._buffers[name] = DTensor.from_local(
                        local,
                        device_mesh=buf.device_mesh,
                        placements=buf.placements,
                        run_check=False,
                    )
                else:
                    buf.data = buf.data.to(torch.bfloat16)
    logger.info("WAM model loaded and cast to bf16.")
    return model


def _replace_te_rmsnorm(module: torch.nn.Module) -> None:
    """Replace all te.pytorch.RMSNorm instances with a pure-PyTorch RMSNorm.

    TE's custom CUDA kernel doesn't handle DTensors or certain inference scenarios.
    The pure-PyTorch version from minimal_v4_dit.py is mathematically identical.
    """
    import transformer_engine as te

    class PurePytorchRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return output.type_as(x) * self.weight

    # Collect all TE RMSNorm instances first (can't modify tree while iterating)
    to_replace = []
    for name, child in module.named_modules():
        if isinstance(child, te.pytorch.RMSNorm):
            to_replace.append((name, child))

    count = 0
    for name, old in to_replace:
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        new_norm = PurePytorchRMSNorm(old.weight.shape[0], eps=old.eps)
        new_norm.weight.data.copy_(old.weight.data)
        new_norm = new_norm.to(old.weight.device)
        setattr(parent, parts[-1], new_norm)
        count += 1

    logger.info(f"Replaced {count} te.pytorch.RMSNorm instances with pure-PyTorch RMSNorm")


def load_wam_model_single_gpu(checkpoint_path: str, experiment: str):
    """Load WAM model on a single GPU without FSDP.

    Requires a pre-converted regular .pt checkpoint (from scripts/convert_dcp_to_regular.py).
    No torchrun needed — runs as a plain python process.
    """
    import importlib

    # Minimal distributed init (megatron parallel_state needs it even for 1 GPU)
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group("nccl")
        torch.cuda.set_device(0)

    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()

    from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override

    config_path = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
    config_module = get_config_module(config_path)
    config = importlib.import_module(config_module).make_config()
    config_overrides = ["--", f"experiment={experiment}"]
    tokenizer_pth = os.environ.get("TOKENIZER_PTH")
    if tokenizer_pth:
        config_overrides.append(f"+model.config.tokenizer.vae_pth={tokenizer_pth}")
        logger.info(f"Using local tokenizer checkpoint: {tokenizer_pth}")
    config = override(config, config_overrides)

    # Disable FSDP so build_net() skips fully_shard()
    config.model.config.fsdp_shard_size = 1
    logger.info(f"Building model without FSDP for experiment={experiment}")

    model = instantiate(config.model)
    model.cuda()
    logger.info(f"Model instantiated: {type(model).__name__}")
    _log_crossattn_proj(model, "single-gpu-build")

    # Replace TE RMSNorm BEFORE loading weights (so parameter shapes match)
    _replace_te_rmsnorm(model)

    # Load the converted regular state dict
    logger.info(f"Loading regular checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    # The DCP state dict uses flattened keys like "net.blocks.0.self_attn.q_proj.weight"
    # which match the model's state dict keys.
    # Note: the model has a custom load_state_dict that may return None.
    # IMPORTANT: the custom load_state_dict only accepts keys starting with "net." or
    # "net_ema." — it silently discards everything else, including "action_head.*".
    # We load the action_head weights separately below.
    result = model.load_state_dict(state_dict, strict=False)
    if result is not None:
        missing, unexpected = result
        if missing:
            critical_missing = [k for k in missing if "_extra_state" not in k]
            if critical_missing:
                logger.warning(f"Missing keys ({len(critical_missing)}): {critical_missing[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    # Load action_head weights separately (bypassed by the custom load_state_dict).
    if hasattr(model, "action_head"):
        ah_sd = {
            k[len("action_head."):]: v
            for k, v in state_dict.items()
            if k.startswith("action_head.")
        }
        if ah_sd:
            model.action_head.load_state_dict(ah_sd, strict=True)
            logger.info(f"Loaded {len(ah_sd)} action_head keys into model.action_head")
        else:
            logger.warning("No action_head.* keys in state_dict — action_head stays at init weights!")

    del state_dict

    # Cast to bf16 and ensure everything is on CUDA.
    model = model.cuda().to(torch.bfloat16)

    # Keep t_embedder and t_embedding_norm in fp32 when use_wan_fp32_strategy=True,
    # because the DiT blocks assert emb_B_T_D.dtype == float32.
    if getattr(model.net, "use_wan_fp32_strategy", False):
        model.net.t_embedder.float()
        model.net.t_embedding_norm.float()
        logger.info("Kept t_embedder and t_embedding_norm in fp32 (use_wan_fp32_strategy=True)")

    model.eval()

    # DEBUG: verify action_head weights were loaded (not zero-initialized).
    if hasattr(model, "action_head"):
        ah_sd = model.action_head.state_dict()
        logger.info(f"[DEBUG] action_head keys after load: {list(ah_sd.keys())}")
        for k in ("mlp.2.weight", "mlp.2.bias", "action_queries", "cross_attn.in_proj_weight"):
            if k in ah_sd:
                v = ah_sd[k]
                logger.info(f"[DEBUG] action_head.{k}: norm={v.float().norm().item():.4f} mean_abs={v.float().abs().mean().item():.6f} dtype={v.dtype}")

    logger.info("WAM model loaded (single-GPU, no FSDP)")
    return model


class CosmosWAMPolicy:
    """
    WAM policy that wraps the Cosmos-based JointActionVideoModel.

    Handles:
    - Observation format conversion (roboarena → model format)
    - Frame accumulation for autoregressive chunked inference
    - Joint video+action denoising via ODE solving
    - Action format conversion
    """

    # Camera key mapping: client obs → model video key
    # Order MUST match training grid stitch order (left, right, wrist) for correct cell placement.
    # Support both naming conventions (roboarena: exterior_image_*, OpenPI: agentview_*)
    _VIDEO_KEY_MAPPING = {
        "observation/agentview_left": "robot0_agentview_left",
        "observation/agentview_right": "robot0_agentview_right",
        "observation/eye_in_hand": "robot0_eye_in_hand",
    }
    # Fallback aliases (roboarena / DreamZero convention)
    _VIDEO_KEY_ALIASES = {
        "observation/exterior_image_0_left": "observation/agentview_right",
        "observation/exterior_image_1_left": "observation/agentview_left",
        "observation/wrist_image_left": "observation/eye_in_hand",
    }

    # Target image size for the model (must match training resolution)
    TARGET_H = 256
    TARGET_W = 256

    def __init__(
        self,
        model,
        device="cuda",
        output_dir: str | None = None,
        num_steps: int = 35,
        guidance: float = 1.5,
        shift: float = 5.0,
        seed: int = 1,
        robocasa_video_mode: str = "grid",
    ):
        self._model = model
        self._device = device
        self._output_dir = output_dir
        self._num_steps = num_steps
        self._guidance = guidance
        self._shift = shift
        self._seed = seed
        if robocasa_video_mode not in {"grid", "left_reason1_right_wrist"}:
            raise ValueError(
                f"Unsupported robocasa_video_mode={robocasa_video_mode!r}; "
                "expected 'grid' or 'left_reason1_right_wrist'."
            )
        self._robocasa_video_mode = robocasa_video_mode

        # Optional: env-var override for action ODE step count. Lets a debug
        # eval run more (or fewer) Euler steps than what the trained model's
        # YAML set, without retraining. Most useful for diagnosing whether the
        # gripping-phase failure of flow-matching action heads is due to
        # discretization error in the small-t end of the ODE — increase to
        # 50/100 to test.
        _flow_steps_env = os.environ.get("FLOW_INFERENCE_NUM_STEPS")
        if _flow_steps_env is not None:
            new_steps = int(_flow_steps_env)
            if hasattr(self._model.config, "flow_inference_num_steps"):
                old_steps = self._model.config.flow_inference_num_steps
                self._model.config.flow_inference_num_steps = new_steps
                logger.info(
                    f"[OVERRIDE] flow_inference_num_steps {old_steps} -> {new_steps} "
                    f"(via FLOW_INFERENCE_NUM_STEPS env var)"
                )
            else:
                logger.warning(
                    f"FLOW_INFERENCE_NUM_STEPS={new_steps} set but model.config has no "
                    f"flow_inference_num_steps attr — ignoring"
                )

        # Frame buffer: accumulate frames for conditioning
        self._frame_buffer: list[np.ndarray] = []  # List of stitched frames (H, W, C)
        self._state_buffer: list[np.ndarray] = []
        self._reason1_conditioning_frames: torch.Tensor | None = None
        self._call_count = 0
        self._current_session_id: str | None = None

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
        logger.info(f"[WAM] RoboCasa video mode: {self._robocasa_video_mode}")

    def _resolve_camera_aliases(self, obs: dict) -> None:
        """Resolve alternate RoboCasa/OpenPI camera key names in-place."""
        for alias, canonical in self._VIDEO_KEY_ALIASES.items():
            if alias in obs and canonical not in obs:
                obs[canonical] = obs[alias]

    def _resize_camera(self, obs: dict, client_key: str, cell_size: int = 256) -> np.ndarray:
        """Return one camera as a uint8 HWC frame at `cell_size` square."""
        from PIL import Image

        img = obs.get(client_key)
        if img is None:
            logger.warning(f"[WAM] Camera key '{client_key}' not found in obs (keys: {list(obs.keys())})")
            return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        if img.ndim == 4:
            img = img[0]
        return np.asarray(
            Image.fromarray(img.astype(np.uint8)).resize(
                (cell_size, cell_size), Image.BILINEAR
            )
        )

    def _to_reason1_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC to CHW float in [-1, 1], matching training."""
        return (
            _torch_uint8_from_numpy(frame)
            .permute(2, 0, 1)
            .to(device=self._device, dtype=torch.float32)
            / 127.5
            - 1.0
        )

    def _append_observation_frames(self, obs: dict) -> None:
        """Append the diffusion-video frame and optional Reason1 image frames."""
        self._resolve_camera_aliases(obs)

        if self._robocasa_video_mode == "left_reason1_right_wrist":
            left = self._resize_camera(obs, "observation/agentview_left")
            right = self._resize_camera(obs, "observation/agentview_right")
            wrist = self._resize_camera(obs, "observation/eye_in_hand")
            self._frame_buffer.append(left)
            self._reason1_conditioning_frames = torch.stack(
                [self._to_reason1_frame(right), self._to_reason1_frame(wrist)],
                dim=0,
            ).unsqueeze(0)
        else:
            self._frame_buffer.append(self._stitch_cameras(obs))
            self._reason1_conditioning_frames = None

    def _attach_image_conditioning(self, data_batch: dict, site: str) -> None:
        """Attach Reason1 image-conditioning frames using the training layout."""
        if "reason1_conditioning_frames" in data_batch:
            data_batch["conditioning_frames"] = data_batch["reason1_conditioning_frames"].contiguous()
            _cf = data_batch["conditioning_frames"]
            logger.info(
                f"[VLM-DIAG {site}] conditioning_frames shape={tuple(_cf.shape)} "
                f"dtype={_cf.dtype} mean={_cf.float().mean().item():.4f} "
                f"std={_cf.float().std().item():.4f} "
                f"min={_cf.float().min().item():.2f} max={_cf.float().max().item():.2f}"
            )
            return

        _vt = data_batch.get("video", None)
        if _vt is not None and _vt.dim() == 5:
            data_batch["conditioning_frame"] = _vt[:, :, 0].contiguous()
            _cf = data_batch["conditioning_frame"]
            logger.info(
                f"[VLM-DIAG {site}] cond_frame shape={tuple(_cf.shape)} "
                f"dtype={_cf.dtype} mean={_cf.float().mean().item():.4f} "
                f"std={_cf.float().std().item():.4f} "
                f"min={_cf.float().min().item():.2f} max={_cf.float().max().item():.2f}"
            )

    def _stitch_cameras(self, obs: dict) -> np.ndarray:
        """Stitch 3 camera views into a 2×2 grid, then center-crop + resize to TARGET_H × TARGET_W.

        Training pipeline: 3 views (each 256×256) → 2×2 grid (512×512) → random crop to
        TARGET_H × TARGET_W → resize TARGET_H × TARGET_W.
        At inference we use a center crop instead of random crop for determinism.
        Output: (TARGET_H, TARGET_W, 3) uint8
        """
        from PIL import Image

        # Resolve aliases (roboarena → OpenPI key names)
        self._resolve_camera_aliases(obs)

        # Collect camera images, resize each to 256×256 (matching training data)
        cell_size = 256
        views = []
        for client_key in self._VIDEO_KEY_MAPPING:
            views.append(self._resize_camera(obs, client_key, cell_size=cell_size))

        # 2×2 grid (4th slot is zero-padded, matching VideoGridStitch)
        grid = np.zeros((cell_size * 2, cell_size * 2, 3), dtype=np.uint8)
        for idx, v in enumerate(views):
            r, c = divmod(idx, 2)
            grid[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = v

        # Aspect-ratio center crop (matching training's VideoCrop), then resize.
        # VideoCrop enforces target aspect ratio via center crop, then VideoResize scales.
        gh, gw = grid.shape[:2]  # 512, 512
        target_ratio = self.TARGET_W / self.TARGET_H  # 480/256 = 1.875
        grid_ratio = gw / gh  # 1.0
        if grid_ratio > target_ratio:
            crop_h = gh
            crop_w = int(gh * target_ratio)
        elif grid_ratio < target_ratio:
            crop_h = int(gw / target_ratio)
            crop_w = gw
        else:
            crop_h, crop_w = gh, gw
        y0 = (gh - crop_h) // 2
        x0 = (gw - crop_w) // 2
        cropped = grid[y0:y0 + crop_h, x0:x0 + crop_w]

        # Resize to model resolution (matching VideoResize in training)
        stitched = np.asarray(
            Image.fromarray(cropped.astype(np.uint8)).resize(
                (self.TARGET_W, self.TARGET_H), Image.BILINEAR
            )
        )
        return stitched

    # Model expects 13 raw frames → 4 latent frames (temporal compression 4x).
    # Formula: T_latent = 1 + (T_raw - 1) // 4.  For T_raw=13: T_latent=4.
    # 4 latent frames = 1 conditioning + 3 predicted; 3 predicted × 4 actions = 12 action steps.
    NUM_VIDEO_FRAMES = 13

    def _build_data_batch(self) -> dict:
        """Build data_batch dict from accumulated frames for model inference.

        The model requires a full-length video (13 frames) even at inference time.
        Only the first frame is a real conditioning frame; the remaining 12 are
        zero-padded placeholders that get replaced by noise during ODE solving.
        This mirrors DreamDojo's ``read_and_process_image`` which pads to
        ``model_required_frames`` with zeros.
        """
        # Stack conditioning frames: (T_cond, H, W, C) uint8
        frames = np.stack(self._frame_buffer, axis=0)
        T_cond = frames.shape[0]

        # Pad to full video length by REPEATING the last conditioning frame.
        # Zero-padding corrupts the VAE's temporal convolutions — adjacent zeros
        # bleed into the conditioning frame's latent representation. Repeating
        # the frame keeps the temporal conv output clean (the conditioning mask
        # still replaces non-conditioning latent frames with noise during ODE).
        if T_cond < self.NUM_VIDEO_FRAMES:
            last_frame = frames[-1:]  # (1, H, W, C)
            padding = np.repeat(last_frame, self.NUM_VIDEO_FRAMES - T_cond, axis=0)
            frames = np.concatenate([frames, padding], axis=0)

        video = (
            _torch_uint8_from_numpy(frames)
            .permute(3, 0, 1, 2)          # (C, T, H, W)
            .unsqueeze(0)                  # (1, C, T, H, W)
            .to(torch.uint8)
            .to(self._device)
        )

        # Action placeholder — actual actions are denoised from noise
        num_action_steps = 12  # 3 predicted latent frames × 4 actions per frame
        action_dim = 7
        data_batch = {
            "video": video,
            "fps": torch.tensor([4], device=self._device),  # Must match training fps (dataset returns fps=4)
            "padding_mask": torch.zeros(1, 1, self.TARGET_H, self.TARGET_W, device=self._device, dtype=torch.bfloat16),
            "action": torch.zeros(1, num_action_steps, action_dim, device=self._device, dtype=torch.bfloat16),
            "num_conditional_frames": 1,
        }

        # State conditioning — model-dependent.
        # Non-proprio models: training never provides state, so state_embedder weights
        # are untrained (random init); passing state would corrupt AdaLN. Skip.
        # Proprio models (use_proprio_conditioning=True): the trained proprio_conditioner
        # requires data_batch['state'] of shape (B, T, proprio_dim). Send the latest 16-D
        # state from the buffer.
        if getattr(self._model.config, "use_proprio_conditioning", False) and self._state_buffer:
            state = np.array(self._state_buffer[-1:], dtype=np.float32)  # (1, 16)
            data_batch["state"] = (
                torch.from_numpy(state)
                .unsqueeze(0)
                .to(device=self._device, dtype=torch.bfloat16)
            )  # (1, 1, 16)

        # Text conditioning (task description from observation)
        # Must be a list of strings (one per batch item), not a bare string
        data_batch["ai_caption"] = [getattr(self, "_current_prompt", "")]

        if self._reason1_conditioning_frames is not None:
            data_batch["reason1_conditioning_frames"] = self._reason1_conditioning_frames

        return data_batch

    @torch.no_grad()
    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations."""
        # Session tracking
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed, resetting state")
                self._reset_state()
            self._current_session_id = session_id

        self._call_count += 1

        # Build ai_caption to match the training-time text template:
        #   - baseline (no reasoning_text): ai_caption = "<task_desc>"
        #       (matches caption_mode=baseline which returns just task_text)
        #   - with reasoning_text: ai_caption = "Task: <task_desc> Reasoning: <caption>"
        #       (matches per-step/per-episode caption modes which produce
        #        f"Task: {task_text} Reasoning: {detail_text}" after label unification).
        task_desc = obs.get("prompt", "")
        reasoning = obs.get("reasoning_text", "")
        if reasoning:
            self._current_prompt = f"Task: {task_desc} Reasoning: {reasoning}"
        else:
            self._current_prompt = task_desc

        # Build diffusion-video frame and optional Reason1 frames.
        self._append_observation_frames(obs)

        # State
        state = obs.get("observation/state", np.zeros(16, dtype=np.float64))
        if state.ndim == 1:
            self._state_buffer.append(state[:16])

        # Need at least 1 conditioning frame for inference
        # Model expects 13 frames (1 cond + 12 predicted), but for inference
        # we only provide 1 conditioning frame and let the model predict the rest
        # Keep only the last frame as conditioning
        if len(self._frame_buffer) > 1:
            self._frame_buffer = self._frame_buffer[-1:]

        # Build data batch
        data_batch = self._build_data_batch()

        # Run joint video+action ODE solving
        t_start = time.time()

        # Compute text embeddings online
        if hasattr(self._model, 'text_encoder') and self._model.text_encoder is not None:
            if hasattr(self._model.config, 'text_encoder_config') and self._model.config.text_encoder_config is not None:
                if self._model.config.text_encoder_config.compute_online:
                    # Image-conditioned text encoders (Reason1 with image_conditioning=True,
                    # used by use_vlm_action_expert) need the current observation frame.
                    # Splice it from the video tensor before invoking the encoder.
                    if getattr(self._model.config.text_encoder_config, "image_conditioning", False):
                        self._attach_image_conditioning(data_batch, "site1")
                    text_emb = self._model.text_encoder.compute_text_embeddings_online(
                        data_batch, self._model.input_caption_key
                    )
                    data_batch["t5_text_embeddings"] = text_emb
                    data_batch["t5_text_mask"] = torch.ones(
                        text_emb.shape[0], text_emb.shape[1],
                        device=text_emb.device, dtype=text_emb.dtype,
                    )

        if "t5_text_embeddings" in data_batch:
            logger.info(f"[WAM] t5_text_embeddings shape: {data_batch['t5_text_embeddings'].shape}")
        else:
            logger.info("[WAM] t5_text_embeddings NOT in data_batch")
        logger.info(f"[WAM] ai_caption type={type(data_batch.get('ai_caption'))} val={repr(data_batch.get('ai_caption', 'MISSING'))[:80]}")

        _, _, action_pred, _ = self._model.generate_samples_with_latents_from_batch(
            data_batch,
            guidance=self._guidance,
            seed=self._seed,
            num_steps=self._num_steps,
            shift=self._shift,
        )

        t_elapsed = time.time() - t_start

        # DEBUG: log raw action tensor and cached_hidden state
        if self._call_count <= 2:
            logger.info(f"[DEBUG] raw action_pred: is_none={action_pred is None}")
            if action_pred is not None:
                logger.info(f"[DEBUG] raw action_pred: shape={tuple(action_pred.shape)} dtype={action_pred.dtype} norm={action_pred.float().norm().item():.6f} abs_max={action_pred.float().abs().max().item():.6f}")
                logger.info(f"[DEBUG] raw action_pred[0,0]: {action_pred[0, 0].float().cpu().tolist()}")
            ch = getattr(self._model, "_cached_hidden", "NOT_SET")
            if ch is None or ch == "NOT_SET":
                logger.info(f"[DEBUG] _cached_hidden: {ch}")
            else:
                logger.info(f"[DEBUG] _cached_hidden: shape={tuple(ch.shape)} norm={ch.float().norm().item():.4f}")

        # action_pred: (1, num_action_steps, action_dim) → (num_action_steps, action_dim)
        action_pred = _tensor_to_numpy_float32(action_pred[0])  # (12, 7)
        action_pred = _denorm_action(action_pred)

        if self._call_count % 10 == 1:
            logger.info(
                f"[WAM] call={self._call_count} actions={action_pred.shape} "
                f"time={t_elapsed:.2f}s prompt={self._current_prompt[:50]}"
            )

        logger.info(
            f"[WAM] call={self._call_count} actions={action_pred.shape} "
            f"time={t_elapsed:.2f}s prompt={self._current_prompt[:50]}"
        )

        return {"actions": action_pred.astype(np.float32)}

    def _reset_state(self):
        self._frame_buffer = []
        self._state_buffer = []
        self._reason1_conditioning_frames = None
        self._call_count = 0
        self._current_prompt = ""

    def reset(self, reset_info: dict) -> None:
        self._reset_state()


def _distributed_forward(model, data_batch, guidance, seed, num_steps):
    """Run model forward on all ranks (required for FSDP collectives)."""
    import torch.distributed as dist
    import pickle

    rank = dist.get_rank()

    # Broadcast data_batch from rank 0 to all ranks
    if rank == 0:
        serialized = pickle.dumps(data_batch)
        size = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
    else:
        size = torch.tensor([0], dtype=torch.int64, device="cuda")

    dist.broadcast(size, src=0)

    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).cuda()
    else:
        data_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")

    dist.broadcast(data_tensor, src=0)

    if rank != 0:
        data_batch = pickle.loads(_tensor_to_bytes(data_tensor))
        # Move tensors to local device
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                data_batch[k] = v.cuda()

    # All ranks run forward together
    _, _, action_pred, _ = model.generate_samples_with_latents_from_batch(
        data_batch,
        guidance=guidance,
        seed=seed,
        num_steps=num_steps,
    )

    return action_pred


class DistributedCosmosWAMPolicy(CosmosWAMPolicy):
    """WAM policy that uses distributed forward for FSDP compatibility."""

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Infer actions from observations (rank 0 only)."""
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed, resetting state")
                self._reset_state()
            self._current_session_id = session_id

        self._call_count += 1
        task_desc = obs.get("prompt", "")
        reasoning = obs.get("reasoning_text", "")
        if reasoning:
            self._current_prompt = f"Task: {task_desc} Reasoning: {reasoning}"
        else:
            self._current_prompt = task_desc

        self._append_observation_frames(obs)

        state = obs.get("observation/state", np.zeros(16, dtype=np.float64))
        if state.ndim == 1:
            self._state_buffer.append(state[:16])

        if len(self._frame_buffer) > 1:
            self._frame_buffer = self._frame_buffer[-1:]

        data_batch = self._build_data_batch()

        # Text embeddings
        if hasattr(self._model, 'text_encoder') and self._model.text_encoder is not None:
            if hasattr(self._model.config, 'text_encoder_config') and self._model.config.text_encoder_config is not None:
                if self._model.config.text_encoder_config.compute_online:
                    if getattr(self._model.config.text_encoder_config, "image_conditioning", False):
                        self._attach_image_conditioning(data_batch, "site2")
                    text_emb = self._model.text_encoder.compute_text_embeddings_online(
                        data_batch, self._model.input_caption_key
                    )
                    data_batch["t5_text_embeddings"] = text_emb
                    data_batch["t5_text_mask"] = torch.ones(
                        text_emb.shape[0], text_emb.shape[1],
                        device=text_emb.device, dtype=text_emb.dtype,
                    )

        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                data_batch[k] = v.to(torch.bfloat16)

        t_start = time.time()
        action_pred = _distributed_forward(
            self._model, data_batch, self._guidance, self._seed, self._num_steps,
        )
        t_elapsed = time.time() - t_start

        action_pred = _tensor_to_numpy_float32(action_pred[0])
        action_pred = _denorm_action(action_pred)

        logger.info(
            f"[WAM] call={self._call_count} actions={action_pred.shape} "
            f"time={t_elapsed:.2f}s prompt={self._current_prompt[:50]}"
        )

        return {"actions": action_pred.astype(np.float32)}


def main_single_gpu(args: Args):
    """Single-GPU inference — no torchrun, no FSDP, no distributed broadcast."""
    model = load_wam_model_single_gpu(args.checkpoint_path, args.experiment)

    # Apply post-load config overrides (no model rebuild required since
    # flow_inference_num_steps is read at every inference call inside
    # generate_samples_with_latents_from_batch — see action_head_video2world_model.py:928).
    if args.flow_inference_num_steps >= 0 and hasattr(model.config, "flow_inference_num_steps"):
        old = model.config.flow_inference_num_steps
        model.config.flow_inference_num_steps = args.flow_inference_num_steps
        logger.info(
            f"[WAM] Overrode model.config.flow_inference_num_steps: "
            f"{old} -> {args.flow_inference_num_steps}"
        )

    policy = CosmosWAMPolicy(
        model=model,
        output_dir=args.output_dir if args.save_video else None,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
        robocasa_video_mode=args.robocasa_video_mode,
    )

    server_config = PolicyServerConfig(
        image_resolution=(256, 256),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_session_id=True,
        action_space="cartesian_position",
    )

    server = WebsocketPolicyServer(
        policy=policy,
        server_config=server_config,
        host="0.0.0.0",
        port=args.port,
    )

    logger.info(f"WAM policy server starting on port {args.port} (single-GPU)")
    server.serve_forever()


def main_distributed(args: Args):
    """Multi-GPU FSDP inference — requires torchrun."""
    import torch.distributed as dist
    import pickle
    import queue
    import threading

    # Initialize distributed (torchrun sets env vars)
    mesh, rank, world_size = init_distributed()
    logger.info(f"Rank {rank}/{world_size} initialized")

    model = load_wam_model(args.checkpoint_path, args.experiment)

    # Shared queue: websocket thread puts data_batch, main thread consumes
    request_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    if rank == 0:
        policy = CosmosWAMPolicy(
            model=model,
            output_dir=args.output_dir if args.save_video else None,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
            robocasa_video_mode=args.robocasa_video_mode,
        )

        class QueuePolicy:
            """Thin wrapper: websocket handler puts obs in queue, waits for result."""
            def infer(self, obs):
                # Prepare data_batch on websocket thread (no CUDA ops)
                policy._call_count += 1
                session_id = obs.get("session_id", None)
                if session_id is not None and session_id != policy._current_session_id:
                    if policy._current_session_id is not None:
                        policy._reset_state()
                    policy._current_session_id = session_id
                _td = obs.get("prompt", "")
                _r = obs.get("reasoning_text", "")
                policy._current_prompt = f"Task: {_td} Reasoning: {_r}" if _r else _td

                policy._append_observation_frames(obs)
                state = obs.get("observation/state", np.zeros(16, dtype=np.float64))
                if state.ndim == 1:
                    policy._state_buffer.append(state[:16])
                if len(policy._frame_buffer) > 1:
                    policy._frame_buffer = policy._frame_buffer[-1:]

                data_batch = policy._build_data_batch()

                # Text embeddings
                if hasattr(model, 'text_encoder') and model.text_encoder is not None:
                    if hasattr(model.config, 'text_encoder_config') and model.config.text_encoder_config is not None:
                        if model.config.text_encoder_config.compute_online:
                            if getattr(model.config.text_encoder_config, "image_conditioning", False):
                                policy._attach_image_conditioning(data_batch, "site3")
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                text_emb = model.text_encoder.compute_text_embeddings_online(
                                    data_batch, model.input_caption_key
                                )
                            data_batch["t5_text_embeddings"] = text_emb.to(torch.bfloat16)
                            data_batch["t5_text_mask"] = torch.ones(
                                text_emb.shape[0], text_emb.shape[1],
                                device=text_emb.device, dtype=torch.bfloat16,
                            )

                for k, v in data_batch.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        data_batch[k] = v.to(torch.bfloat16)

                # Put request for main thread, wait for result
                request_queue.put(data_batch)
                result = result_queue.get()
                return result

            def reset(self, reset_info):
                policy._reset_state()

        server_config = PolicyServerConfig(
            image_resolution=(256, 256),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_session_id=True,
            action_space="cartesian_position",
        )

        queue_policy = QueuePolicy()
        server = WebsocketPolicyServer(
            policy=queue_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )

        # Run websocket server in background thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        logger.info(f"Starting WAM policy server on port {args.port}")

    # ALL ranks: main loop — rank 0 gets data from queue, broadcasts to workers
    logger.info(f"Rank {rank} entering main inference loop")
    while True:
        if rank == 0:
            # Wait for request from websocket handler
            data_batch = request_queue.get()
            t_start = time.time()

            # Serialize and broadcast
            serialized = pickle.dumps(data_batch)
            size = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
        else:
            size = torch.tensor([0], dtype=torch.int64, device="cuda")

        dist.broadcast(size, src=0)

        if rank == 0:
            data_tensor = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).cuda()
        else:
            data_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")

        dist.broadcast(data_tensor, src=0)

        if rank != 0:
            data_batch = pickle.loads(_tensor_to_bytes(data_tensor))
            for k, v in data_batch.items():
                if isinstance(v, torch.Tensor):
                    data_batch[k] = v.cuda()

        # All ranks run forward together (FSDP collectives)
        with torch.no_grad():
            _, _, action_pred, _ = model.generate_samples_with_latents_from_batch(
                data_batch,
                guidance=args.guidance,
                seed=args.seed,
                num_steps=args.num_steps,
            )

        if rank == 0:
            action_np = _tensor_to_numpy_float32(action_pred[0])
            action_np = _denorm_action(action_np)
            t_elapsed = time.time() - t_start
            logger.info(
                f"[WAM] call={policy._call_count} actions={action_np.shape} "
                f"time={t_elapsed:.2f}s"
            )
            result_queue.put({"actions": action_np.astype(np.float32)})


def main(args: Args):
    logging.basicConfig(level=logging.INFO)

    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required")

    if args.action_stats_paths.strip():
        global _ACTION_DENORM_STATS
        paths = [p.strip() for p in args.action_stats_paths.split(",") if p.strip()]
        agg_min, agg_max = _load_aggregate_action_stats(paths)
        _ACTION_DENORM_STATS = (agg_min, agg_max)
        logger.info(
            f"[WAM] Loaded action denorm stats from {len(paths)} dataset(s); "
            f"action predictions will be de-normalized before return."
        )
        logger.info(f"[WAM]   action min: {np.array2string(agg_min, precision=4)}")
        logger.info(f"[WAM]   action max: {np.array2string(agg_max, precision=4)}")
    else:
        logger.info(
            "[WAM] No --action-stats-paths provided; returning normalized [-1, 1] "
            "predictions (no denorm). Pass training dataset paths to invert min_max."
        )

    if args.flow_inference_num_steps >= 0:
        logger.info(
            f"[WAM] Will override model.config.flow_inference_num_steps "
            f"to {args.flow_inference_num_steps} after load."
        )

    if args.single_gpu:
        main_single_gpu(args)
    else:
        main_distributed(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
