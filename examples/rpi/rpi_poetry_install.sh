#!/bin/sh

# This script installs the Python and apt packages needed for the genAI program on a Raspberry Pi.
# For installing on Raspberry Pi OS -- use this, for other systems, use `poetry install`

# install what we need
sudo apt update && sudo apt upgrade -y && \
sudo apt install -y \
    libhdf5-dev \
    unzip \
    pkg-config \
    python3-pip \
    cmake \
    make \
    git \
    python-is-python3 \
    wget \
    patchelf && \

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# install the package
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
