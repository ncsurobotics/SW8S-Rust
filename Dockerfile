# Docker image for https://github.com/ncsurobotics/SW8S-Rust
FROM ubuntu:22.04

# Make sure apt doesn't prompt frontend
ARG DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.description="TEST Docker image for Seawolf 8 (Aquapack NCSU)"

# System setup
RUN apt-get update \
  # Install system dependencies
  && apt-get install -y git \
    curl \
    gcc \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    libopencv-dev \
    lld \
    llvm \
    sudo \
  # Create the user (adapted from https://github.com/Homebrew/brew/blob/master/Dockerfile)
  && useradd -m -s /bin/bash aquapack \
  && echo 'aquapack ALL=(ALL) NOPASSWD:ALL' >>/etc/sudoers


RUN apt-get install nano
# User setup
USER aquapack
WORKDIR /home/aquapack
# RUN  \
  # Install Rust via rustup (apt's Rust is too old)
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
  && . "$HOME/.cargo/env" && echo '. "$HOME/.cargo/env"' >> ~/.bashrc \
  # Clone the repo
  && git clone https://github.com/ncsurobotics/SW8S-Rust
