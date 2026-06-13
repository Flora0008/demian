"""
Convert FSDP DCP checkpoint to a regular (non-sharded) state dict.

The training pipeline saves checkpoints in PyTorch Distributed Checkpoint (DCP) format
with FSDP DTensor shards. At inference time, te.pytorch.RMSNorm doesn't handle DTensors,
so we consolidate them into a single regular .pt file.

Usage:
    torchrun --nproc_per_node=1 scripts/convert_dcp_to_regular.py \
        --checkpoint-path /path/to/iter_000004500 \
        --experiment dreamdojo_2b_480_640_robocasa_wam_baseline \
        --output /path/to/iter_000004500/model_regular.pt
"""

import argparse
import importlib
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

# torch.from_numpy fallback for containers where torch was built against NumPy 1.x
# but NumPy 2.x is installed (newer eval container). Mirrors the
# same monkey-patch in socket_test_cosmos_wam.py so model instantiation -- which
# transitively constructs diffusers' FlowMatchEulerDiscreteScheduler -- doesn't
# crash on `torch.from_numpy(timesteps)`.
_ORIG_TORCH_FROM_NUMPY = torch.from_numpy
_DTYPE_MAP = {
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


def _torch_from_numpy_compat(array):
    try:
        return _ORIG_TORCH_FROM_NUMPY(array)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
    array = np.ascontiguousarray(array)
    torch_dtype = _DTYPE_MAP.get(array.dtype)
    if torch_dtype is None:
        return torch.tensor(array.tolist())
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch_dtype).reshape(array.shape)


torch.from_numpy = _torch_from_numpy_compat


# torch.Tensor.numpy fallback for the same numpy 1.x/2.x ABI mismatch. DCP's
# gather_object() at torch/distributed/distributed_c10d.py:3045 calls
# `tensor.numpy().tobytes()`, which crashes with the bare container's bridge.
_ORIG_TENSOR_NUMPY = torch.Tensor.numpy
_TENSOR_DTYPE_TO_NP = {
    torch.uint8: np.uint8, torch.int8: np.int8,
    torch.int16: np.int16, torch.int32: np.int32, torch.int64: np.int64,
    torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
    torch.bool: np.bool_,
}


def _torch_tensor_numpy_compat(self, *args, **kwargs):
    try:
        return _ORIG_TENSOR_NUMPY(self, *args, **kwargs)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
    cpu = self.detach().cpu()
    target_np = _TENSOR_DTYPE_TO_NP.get(cpu.dtype, np.float32)
    return np.asarray(cpu.tolist(), dtype=target_np)


torch.Tensor.numpy = _torch_tensor_numpy_compat


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to regular .pt")
    parser.add_argument("--checkpoint-path", required=True, help="Path to DCP checkpoint dir (e.g. iter_000004500)")
    parser.add_argument("--experiment", required=True, help="Registered experiment name")
    parser.add_argument("--output", default=None, help="Output .pt path (default: <checkpoint-path>/model_regular.pt)")
    parser.add_argument(
        "--tokenizer-pth",
        default=os.environ.get("TOKENIZER_PTH"),
        help="Optional local tokenizer checkpoint path to avoid downloading tokenizer.pth",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.checkpoint_path, "model_regular.pt")

    # --- 1. Initialize distributed (torchrun sets env vars) ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    logger.info(f"Rank {rank}, world_size {dist.get_world_size()}")

    # Megatron parallel_state needs dist to be initialized
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel()

    # --- 2. Build model with FSDP (same as training, so DCP keys match) ---
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_dir)

    from cosmos_predict2._src.imaginaire.lazy_config import instantiate
    from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override

    config_path = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
    config_module = get_config_module(config_path)
    config = importlib.import_module(config_module).make_config()
    config_overrides = ["--", f"experiment={args.experiment}"]
    if args.tokenizer_pth:
        config_overrides.append(f"+model.config.tokenizer.vae_pth={args.tokenizer_pth}")
        logger.info(f"Using local tokenizer checkpoint: {args.tokenizer_pth}")
    config = override(config, config_overrides)

    logger.info(f"Config resolved for experiment={args.experiment}")
    logger.info(f"fsdp_shard_size = {config.model.config.fsdp_shard_size}")

    model = instantiate(config.model)
    logger.info(f"Model instantiated: {type(model).__name__}")

    # --- 3. Load DCP checkpoint ---
    model_dir = os.path.join(args.checkpoint_path, "model")
    logger.info(f"Loading DCP from {model_dir}")

    from cosmos_predict2._src.predict2.checkpointer.dcp import ModelWrapper, dcp_load_state_dict
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    model_wrapper = ModelWrapper(model)
    state_dict = model_wrapper.state_dict()
    extra_state_keys = [k for k in list(state_dict.keys()) if k.endswith("._extra_state")]
    for key in extra_state_keys:
        state_dict.pop(key)
    if extra_state_keys:
        logger.info(f"Ignoring {len(extra_state_keys)} '*._extra_state' keys during DCP load")

    load_planner = DefaultLoadPlanner(allow_partial_load=True)
    storage_reader = FileSystemReader(model_dir)

    dist.barrier()
    dcp_load_state_dict(state_dict, storage_reader, load_planner)
    model_wrapper.load_state_dict(state_dict)
    logger.info("DCP checkpoint loaded into FSDP model")

    # --- 4. Extract full (non-sharded) state dict ---
    full_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    logger.info(f"Extracted full state dict: {len(full_state_dict)} keys")

    # Verify no DTensors remain
    from torch.distributed._tensor import DTensor
    dtensor_keys = [k for k, v in full_state_dict.items() if isinstance(v, DTensor)]
    if dtensor_keys:
        logger.warning(f"Found {len(dtensor_keys)} DTensor keys, converting to regular tensors...")
        for k in dtensor_keys:
            full_state_dict[k] = full_state_dict[k].to_local()

    # --- 4b. Merge action_head weights from aux_optim.pt (if present) ---
    # The action_head lives outside the FSDP boundary, so its weights are
    # NOT in the DCP model checkpoint. They are saved separately in aux_optim.pt.
    aux_path = os.path.join(args.checkpoint_path, "aux_optim.pt")
    if os.path.exists(aux_path):
        aux = torch.load(aux_path, map_location="cpu", weights_only=True)
        if "action_head_state_dict" in aux:
            ah_sd = aux["action_head_state_dict"]
            for k, v in ah_sd.items():
                full_state_dict[f"action_head.{k}"] = v
            logger.info(f"Merged {len(ah_sd)} action_head keys from {aux_path}")
        else:
            logger.warning(f"aux_optim.pt exists but has no action_head_state_dict (only optimizer)")

    # --- 5. Save on rank 0 ---
    if rank == 0:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(full_state_dict, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"Saved regular checkpoint to {output_path} ({size_mb:.1f} MB)")

    dist.barrier()
    dist.destroy_process_group()
    logger.info("Done.")


if __name__ == "__main__":
    main()
