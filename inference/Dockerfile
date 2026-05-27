# Minimal CUDA image to run the SmolVLA policy server.
#
# This is a starting point, not a hardened production image. It is the
# shape of container we used while iterating on the policy server, with
# the workarounds that bit us during deployment baked in as build steps.
#
# Build with
#
#   docker build -f inference/Dockerfile -t smolvla-policy-server .
#
# Run with --runtime=nvidia (or with the nvidia-container-runtime daemon
# set as default), otherwise the container will not see the GPU even
# though the host has one.
#
#   docker run --rm --runtime=nvidia --gpus all \
#       -v /path/to/checkpoint:/checkpoint \
#       -p 8000:8000 \
#       smolvla-policy-server
#
# The CUDA tag is pinned to a 12.x runtime base, which matches the
# torch wheels we tested against (PyTorch 2.4 with cu121). If you bump
# torch, bump CUDA in lockstep here.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/root/.cache/huggingface

# Python 3.12 is the floor required by LeRobot v0.5.1.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install LeRobot with the SmolVLA extras and the small set of deps that
# this server needs on top of LeRobot. We pin Torch to the cu121 wheel
# that matches the CUDA runtime above.
RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install \
        "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121 \
    && python3.12 -m pip install \
        "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@v0.5.1" \
        websockets msgpack safetensors

# Pre-cache the SmolVLM-2 processor so the server can boot offline. This
# avoids a first-call stall (and the occasional HF rate-limit) when the
# container starts inside an evaluator without internet.
RUN python3.12 -c "__import__('transformers').AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')"

WORKDIR /opt/policy
COPY inference/policy_server.py /opt/policy/policy_server.py

# The server expects the checkpoint to be mounted at /checkpoint by the
# operator. The startup command reads $POLICY_CHECKPOINT_PATH so a
# different mount point can be used without rebuilding the image.
ENV POLICY_CHECKPOINT_PATH=/checkpoint
EXPOSE 8000

CMD ["sh", "-c", "python3.12 policy_server.py --checkpoint ${POLICY_CHECKPOINT_PATH} --host 0.0.0.0 --port 8000"]
