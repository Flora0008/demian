# =============================================================================
# DeMiAn-VLA environment (openpi 0.5 / Pi0.5 PyTorch backbone) — training + eval.
#
# Equivalent to the internal `openpi-train.sqsh` / `molmobot-eval.sqsh` images
# (which are NOT distributed). Covers: VLA post-training (openpi +
# molmobot_pi0) and policy-side eval. The RoboCasa/robosuite and MolmoSpaces
# simulators are *external* repos — clone them and add to PYTHONPATH (see README
# "Setup"); they are not installed here.
#
# Build:
#   docker build -f docker/Dockerfile -t demian-vla:latest .
# (or convert to a .sqsh for pyxis/enroot with `enroot import`.)
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git git-lfs ffmpeg wget curl \
        libgl1 libglib2.0-0 libegl1 libosmesa6 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Torch (CUDA 12.x) first so the rest resolve against it.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.7.1

# openpi 0.5 runtime deps (mirrors openpi-migration/pyproject.toml). JAX is used
# only for tree utilities on CPU at runtime (JAX_PLATFORMS=cpu); flax/nnx back the
# config dataclasses. transformers is pinned to 4.53.2 to match the PaliGemma/Gemma
# patches in openpi/.../models_pytorch/transformers_replace.
RUN python -m pip install \
        "jax[cuda12]==0.5.3" jaxtyping==0.2.36 "flax==0.10.2" "ml_collections==1.0.0" \
        orbax-checkpoint==0.11.13 beartype==0.19.0 "augmax>=0.3.4" "equinox>=0.11.8" \
        "transformers==4.53.2" sentencepiece "numpy>=1.22.4,<2.0.0" "pillow>=11.0.0" \
        "opencv-python>=4.10.0.84" einops "tyro>=0.9.5" "wandb>=0.19.1" h5py decord \
        "tqdm-loggable>=0.2" "fsspec[gcs]>=2024.6.0" treescope rich polars peft \
        "huggingface-hub>=0.30.0,<1.0" hydra-core omegaconf safetensors scipy mediapy \
        websockets msgpack msgpack-numpy "qwen-vl-utils>=0.0.14"

# Apply the openpi PaliGemma/Gemma/SigLIP patches over the installed transformers.
# (Also done at runtime by the slurm scripts; baked here for convenience.)
COPY openpi/src/openpi/models_pytorch/transformers_replace /tmp/transformers_replace
RUN TRANSFORMERS_DIR=$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))') && \
    cp -r /tmp/transformers_replace/* "${TRANSFORMERS_DIR}/" && rm -rf /tmp/transformers_replace

ENV JAX_PLATFORMS=cpu \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
WORKDIR /workspace
