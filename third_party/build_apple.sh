#!/usr/bin/env bash
set -euo pipefail

# build_apple.sh <target> [clean] [backend]
# backend: metal|cpu (default: metal)
# Targets: macos-arm64, macos-x86_64, ios-device-arm64, ios-sim-arm64, ios-sim-x86_64

TARGET="${1:-}"
CLEAN="${2:-}"
BACKEND="${3:-${APPLE_BACKEND:-metal}}"

if [ -z "$TARGET" ]; then
  echo "Usage: build_apple.sh <target> [clean] [backend]" >&2
  exit 1
fi

IOS_MIN_OS_VERSION=16.4
MACOS_MIN_OS_VERSION=11.0

case "$BACKEND" in
  metal)
    APPLE_BACKEND_ARGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=OFF)
    ;;
  cpu)
    APPLE_BACKEND_ARGS=(-DGGML_METAL=OFF)
    ;;
  *)
    echo "Invalid Apple backend '$BACKEND'. Use metal|cpu." >&2
    exit 1
    ;;
esac

build_target() {
    local TYPE="$1"   # STATIC or SHARED
    local BUILD_DIR="$2"
    local OUT_NAME="$3"
    local ARCH="$4"
    local EXTRA_ARGS="$5"
    local DEP_TARGET="$6"

    if [ "$CLEAN" = "clean" ]; then
      rm -rf "$BUILD_DIR"
    fi

    local SHARED_FLAG="OFF"
    if [ "$TYPE" = "SHARED" ]; then
      SHARED_FLAG="ON"
    fi

    cmake -G Ninja -S . -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMADART_SHARED="$SHARED_FLAG" \
      -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
      -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEP_TARGET" \
      "${APPLE_BACKEND_ARGS[@]}" \
      $EXTRA_ARGS

    cmake --build "$BUILD_DIR" --config Release -j "$(sysctl -n hw.logicalcpu)"

    mkdir -p "$(dirname "$OUT_NAME")"
    if [ "$TYPE" = "STATIC" ]; then
      LIBS="$(find "$BUILD_DIR" -name "*.a" ! -name "libllamadart.a")"
      libtool -static -o "$OUT_NAME" ${LIBS} 2>/dev/null
    else
      cp "$BUILD_DIR/libllamadart.dylib" "$OUT_NAME"
    fi
}

if [[ "$TARGET" == macos-* ]]; then
    ARCH="${TARGET#macos-}"
    [ "$ARCH" = "x64" ] && ARCH="x86_64"

    build_target "STATIC" "build-macos-$ARCH-static" "bin/macos/$ARCH/libllamadart.a" "$ARCH" "" "$MACOS_MIN_OS_VERSION"
    build_target "SHARED" "build-macos-$ARCH-shared" "bin/macos/$ARCH/libllamadart.dylib" "$ARCH" "" "$MACOS_MIN_OS_VERSION"

    echo "macOS build complete for $ARCH backend=$BACKEND"

elif [[ "$TARGET" == ios-* ]]; then
    if [ "$TARGET" = "ios-device-arm64" ]; then
        SDK="iphoneos"
        ARCH="arm64"
        OUT_BASE="bin/ios/libllamadart-ios-arm64"
    elif [ "$TARGET" = "ios-sim-arm64" ]; then
        SDK="iphonesimulator"
        ARCH="arm64"
        OUT_BASE="bin/ios/libllamadart-ios-arm64-sim"
    elif [ "$TARGET" = "ios-sim-x86_64" ] || [ "$TARGET" = "ios-sim-x64" ]; then
        SDK="iphonesimulator"
        ARCH="x86_64"
        OUT_BASE="bin/ios/libllamadart-ios-x86_64-sim"
    else
        echo "Invalid iOS target '$TARGET'" >&2
        exit 1
    fi

    EXTRA_IOS_ARGS="-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=$SDK -DIOS=ON"

    build_target "STATIC" "build-ios-$TARGET-static" "${OUT_BASE}.a" "$ARCH" "$EXTRA_IOS_ARGS" "$IOS_MIN_OS_VERSION"
    build_target "SHARED" "build-ios-$TARGET-shared" "${OUT_BASE}.dylib" "$ARCH" "$EXTRA_IOS_ARGS" "$IOS_MIN_OS_VERSION"

    echo "iOS build complete for $TARGET backend=$BACKEND"
else
    echo "Invalid target '$TARGET'." >&2
    exit 1
fi
