#!/usr/bin/env bash
set -euo pipefail

# build_linux.sh <backend> [arch] [clean]
# backends: cpu | vulkan | cuda | zendnn

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "deps/llama.cpp/CMakeLists.txt" ]; then
  echo "Error: missing submodule deps/llama.cpp. Run: git submodule update --init --recursive" >&2
  exit 1
fi

BACKEND="${1:-vulkan}"
ARCH="${2:-}"
CLEAN="${3:-}"

if [ -z "$ARCH" ] || [ "$ARCH" = "clean" ]; then
  CLEAN="$ARCH"
  ARCH="$(uname -m)"
fi

case "$ARCH" in
  aarch64|arm64)
    TARGET_ARCH="arm64"
    ;;
  x86_64|x64)
    TARGET_ARCH="x64"
    ;;
  *)
    echo "Unsupported arch '$ARCH' (expected x64/x86_64/arm64/aarch64)" >&2
    exit 1
    ;;
esac

BUILD_DIR="build-linux-${TARGET_ARCH}-${BACKEND}"
if [ "$CLEAN" = "clean" ]; then
  rm -rf "$BUILD_DIR"
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_SHARED_LINKER_FLAGS=-s
  -DGGML_NATIVE=OFF
  -DGGML_OPENMP=ON
)

# Cross-compile hints
if [ "$TARGET_ARCH" = "arm64" ] && [ "$(uname -m)" != "aarch64" ] && [ "$(uname -m)" != "arm64" ]; then
  if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    CMAKE_ARGS+=(
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc
      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
      -DCMAKE_SYSTEM_NAME=Linux
      -DCMAKE_SYSTEM_PROCESSOR=aarch64
    )
  else
    echo "Cross compiler aarch64-linux-gnu-gcc not found." >&2
    exit 1
  fi
fi

case "$BACKEND" in
  cpu)
    CMAKE_ARGS+=(-DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_ZENDNN=OFF)
    ;;
  vulkan)
    CMAKE_ARGS+=(-DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_ZENDNN=OFF)
    ;;
  cuda)
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "CUDA backend requested but nvcc not found in PATH." >&2
      exit 1
    fi
    if [ "$TARGET_ARCH" != "x64" ]; then
      echo "CUDA backend currently supports Linux x64 only." >&2
      exit 1
    fi
    CMAKE_ARGS+=(-DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_ZENDNN=OFF)
    ;;
  zendnn)
    if [ "$TARGET_ARCH" != "x64" ]; then
      echo "ZenDNN backend currently supports Linux x64 only." >&2
      exit 1
    fi
    CMAKE_ARGS+=(-DGGML_ZENDNN=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF)
    ;;
  *)
    echo "Invalid backend '$BACKEND'. Use cpu|vulkan|cuda|zendnn." >&2
    exit 1
    ;;
esac

echo "Building Linux target=${TARGET_ARCH} backend=${BACKEND}"

mkdir -p "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

OUT_DIR="bin/linux/$TARGET_ARCH"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

cp -L "$BUILD_DIR/libllamadart.so" "$OUT_DIR/libllamadart.so" 2>/dev/null || \
find "$BUILD_DIR" -name "libllamadart.so" -exec cp -L {} "$OUT_DIR/libllamadart.so" \;

echo "Linux build complete: $OUT_DIR/libllamadart.so"
