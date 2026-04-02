FROM ghcr.io/ggml-org/llama.cpp:full-cuda

# Pre-install SkyPilot dependencies for faster startup
# Note: we install both fuse and fuse3 to support different distros (e.g., Ubuntu 22.04 and 24.04)
# fuse3 must be installed after fuse to avoid conflicts.
RUN apt-get update && apt-get install -y rsync curl wget netcat-openbsd gcc patch pciutils \
    fuse openssh-server \
    && apt install -y fuse3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh

RUN ~/.local/bin/uv venv --seed ~/sky_runtime/skypilot-runtime --python 3.10

RUN VIRTUAL_ENV=~/sky_runtime/skypilot-runtime UV_LINK_MODE=copy UV_SYSTEM_PYTHON=false \
    ~/.local/bin/uv pip install "setuptools<70" \
    "ray[default]==2.9.3" "skypilot[kubernetes,remote]"
