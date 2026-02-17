#!/usr/bin/env bash
set -euo pipefail

# build_android.sh <ABI> [clean]
# Env: ANDROID_GPU_BACKEND=vulkan|opencl|cpu (default: vulkan)

if [ "$1" = "all" ]; then
  ABIS=("arm64-v8a" "x86_64")
  CLEAN="${2:-}"
else
  ABIS=("${1:-arm64-v8a}")
  CLEAN="${2:-}"
fi

ANDROID_GPU_BACKEND="${ANDROID_GPU_BACKEND:-vulkan}"

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  POSSIBLE_NDKS=(
    "$HOME/Library/Android/sdk/ndk/26.3.11579264"
    "$HOME/Library/Android/sdk/ndk/27.0.12077973"
    "$HOME/Library/Android/sdk/ndk/25.1.8937393"
    "/usr/local/lib/android/sdk/ndk-bundle"
  )
  for ndk in "${POSSIBLE_NDKS[@]}"; do
    if [ -d "$ndk" ]; then
      export ANDROID_NDK_HOME="$ndk"
      break
    fi
  done
fi

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  echo "Error: ANDROID_NDK_HOME not set and could not be auto-detected." >&2
  exit 1
fi

echo "Using NDK: $ANDROID_NDK_HOME"
echo "Android backend: $ANDROID_GPU_BACKEND"

for ABI in "${ABIS[@]}"; do
  echo "========================================"
  echo "Building Android ABI=$ABI backend=$ANDROID_GPU_BACKEND"
  echo "========================================"

  BUILD_DIR="build-android-$ABI"
  if [ "$CLEAN" = "clean" ]; then
    rm -rf "$BUILD_DIR"
  fi

  TOOLCHAIN_FILE="$(pwd)/$BUILD_DIR/android-host-toolchain.cmake"
  mkdir -p "$BUILD_DIR"

  MAKE_PROG="$(which ninja 2>/dev/null || which make 2>/dev/null || true)"
  if [ -n "$MAKE_PROG" ]; then
    echo "set(CMAKE_MAKE_PROGRAM \"$MAKE_PROG\" CACHE STRING \"make program\" FORCE)" > "$TOOLCHAIN_FILE"
  else
    : > "$TOOLCHAIN_FILE"
  fi
  echo "set(CMAKE_SYSTEM_NAME \"$(uname)\")" >> "$TOOLCHAIN_FILE"
  echo "set(Threads_FOUND TRUE)" >> "$TOOLCHAIN_FILE"
  echo "set(CMAKE_THREAD_LIBS_INIT \"-pthread\")" >> "$TOOLCHAIN_FILE"
  echo "set(CMAKE_USE_PTHREADS_INIT TRUE)" >> "$TOOLCHAIN_FILE"

  ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-23}"
  GGML_VULKAN="OFF"
  GGML_OPENCL="OFF"

  case "$ANDROID_GPU_BACKEND" in
    vulkan)
      GGML_VULKAN="ON"
      if [ "$ANDROID_API_LEVEL" -lt 28 ]; then
        echo "Raising ANDROID_API_LEVEL from $ANDROID_API_LEVEL to 28 for Vulkan symbols"
        ANDROID_API_LEVEL=28
      fi
      ;;
    opencl)
      GGML_OPENCL="ON"
      if [ "$ANDROID_API_LEVEL" -lt 28 ]; then
        ANDROID_API_LEVEL=28
      fi
      ;;
    cpu)
      ;;
    *)
      echo "Invalid ANDROID_GPU_BACKEND '$ANDROID_GPU_BACKEND'. Use vulkan|opencl|cpu." >&2
      exit 1
      ;;
  esac

  if [ "$ABI" = "arm64-v8a" ]; then
    ARCH_PATH="aarch64-linux-android"
  elif [ "$ABI" = "x86_64" ]; then
    ARCH_PATH="x86_64-linux-android"
  elif [ "$ABI" = "x86" ]; then
    ARCH_PATH="i686-linux-android"
  else
    ARCH_PATH="$ABI"
  fi

  EXTRA_CMAKE_ARGS=()
  if [ "$ABI" = "arm64-v8a" ]; then
    EXTRA_CMAKE_ARGS+=(-DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm)
  fi

  VULKAN_ARGS=()
  if [ "$GGML_VULKAN" = "ON" ]; then
    GLSLC="$(find "$ANDROID_NDK_HOME" -name glslc | head -n 1)"
    [ -z "$GLSLC" ] && echo "Warning: glslc not found in NDK."

    VULKAN_LIB="$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -path "*/sysroot/usr/lib/$ARCH_PATH/$ANDROID_API_LEVEL/libvulkan.so" | head -n 1)"
    if [ -z "$VULKAN_LIB" ]; then
      VULKAN_LIB="$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -path "*/sysroot/usr/lib/$ARCH_PATH/*/libvulkan.so" | sort -V | tail -n 1)"
    fi
    if [ -z "$VULKAN_LIB" ]; then
      VULKAN_LIB="$(find "$ANDROID_NDK_HOME" -name libvulkan.so | grep "/$ARCH_PATH/" | head -n 1 || true)"
    fi
    if [ -z "$VULKAN_LIB" ]; then
      echo "Error: libvulkan.so not found for ABI $ABI" >&2
      exit 1
    fi

    VULKAN_INC_DIR="$(pwd)/Vulkan-Headers/include"
    if [ ! -d "$VULKAN_INC_DIR/vulkan" ]; then
      echo "Error: Vulkan-Headers submodule missing." >&2
      exit 1
    fi

    VULKAN_ARGS=(
      -DVulkan_LIBRARY="$VULKAN_LIB"
      -DVulkan_INCLUDE_DIR="$VULKAN_INC_DIR"
      -DVulkan_GLSLC_EXECUTABLE="$GLSLC"
      -DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN="$TOOLCHAIN_FILE"
    )
  fi

  if [ "$GGML_OPENCL" = "ON" ]; then
    OPENCL_INC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/CL"
    OPENCL_LIB_GLOB="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/$ARCH_PATH"

    if [ ! -d "$OPENCL_INC" ]; then
      echo "Error: OpenCL headers missing at $OPENCL_INC" >&2
      echo "Install OpenCL-Headers into NDK sysroot first." >&2
      exit 1
    fi

    OPENCL_LIB="$(find "$OPENCL_LIB_GLOB" -name libOpenCL.so 2>/dev/null | head -n 1 || true)"
    if [ -z "$OPENCL_LIB" ]; then
      echo "Error: libOpenCL.so missing for $ABI in NDK sysroot." >&2
      echo "Build/install OpenCL ICD Loader first." >&2
      exit 1
    fi

    EXTRA_CMAKE_ARGS+=(-DOpenCL_LIBRARY="$OPENCL_LIB")
  fi

  cmake -G Ninja -S . -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-$ANDROID_API_LEVEL" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SHARED_LINKER_FLAGS=-s \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_VULKAN="$GGML_VULKAN" \
    -DGGML_OPENCL="$GGML_OPENCL" \
    "${EXTRA_CMAKE_ARGS[@]}" \
    "${VULKAN_ARGS[@]}"

  cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

  if [ "$ABI" = "arm64-v8a" ]; then
    TARGET_ARCH="arm64"
  elif [ "$ABI" = "x86_64" ]; then
    TARGET_ARCH="x64"
  else
    TARGET_ARCH="$ABI"
  fi

  JNI_LIBS_DIR="bin/android/$TARGET_ARCH"
  rm -rf "$JNI_LIBS_DIR"
  mkdir -p "$JNI_LIBS_DIR"

  cp -L "$BUILD_DIR/libllamadart.so" "$JNI_LIBS_DIR/libllamadart.so" 2>/dev/null || \
  find "$BUILD_DIR" -name libllamadart.so -exec cp -L {} "$JNI_LIBS_DIR/libllamadart.so" \;

  echo "Android build complete: $JNI_LIBS_DIR/libllamadart.so"
done
