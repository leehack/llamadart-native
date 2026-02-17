# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

COPY ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources

RUN dpkg --add-architecture arm64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      ccache \
      cmake \
      g++ \
      gcc \
      gcc-aarch64-linux-gnu \
      g++-aarch64-linux-gnu \
      git \
      glslc \
      libopenblas-dev \
      libopenblas-dev:arm64 \
      libvulkan-dev \
      libvulkan-dev:arm64 \
      ninja-build \
      nvidia-cuda-toolkit \
      pkg-config \
      python3 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENV CCACHE_DIR=/work/.ccache
