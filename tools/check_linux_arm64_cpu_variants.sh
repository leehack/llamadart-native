#!/usr/bin/env bash
# Audits the ISA of a Linux arm64 bundle's CPU modules, then checks under QEMU
# which module ggml selects on each emulated CPU and that it computes there.
#
# Usage: tools/check_linux_arm64_cpu_variants.sh <bundle-dir>
# Run after tools/build.py linux --arch arm64, which leaves the KleidiAI
# sources under build/. Environment: OBJDUMP and READELF (LLVM tools, default
# llvm-objdump/llvm-readelf), CC (AArch64 C compiler, default
# aarch64-linux-gnu-gcc), QEMU_SYSROOT (qemu-aarch64 -L, for cross hosts).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <bundle-dir>" >&2
  exit 2
fi
BUNDLE="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJDUMP="${OBJDUMP:-llvm-objdump}"
READELF="${READELF:-llvm-readelf}"
CC="${CC:-aarch64-linux-gnu-gcc}"

audit() {
  python3 "$ROOT/tools/validate_linux_arm64_cpu_isa.py" \
    --objdump "$OBJDUMP" --readelf "$READELF" \
    --llama-source "$ROOT/third_party/llama.cpp" \
    --kleidiai-source "$ROOT/build/$2/_deps/kleidiai-src" \
    --variant "$1" "$BUNDLE/$3"
}

audit baseline linux-arm64-full libggml-cpu.so
variant_list="$(python3 "$ROOT/tools/build.py" list --linux-arm64-cpu-variant-libs)"
if [ -z "$variant_list" ]; then
  echo "tools/build.py listed no Linux arm64 CPU variants" >&2
  exit 1
fi
mapfile -t variant_libs <<< "$variant_list"
for lib in "${variant_libs[@]}"; do
  variant="${lib#libggml-cpu-}"
  variant="${variant%.so}"
  audit "$variant" "linux-arm64-cpu-$variant" "$lib"
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$CC" -O2 -std=c11 -Wall -Wextra -I "$ROOT/third_party/llama.cpp/ggml/include" \
  "$ROOT/tools/linux_cpu_variant_smoke.c" -o "$work/smoke" \
  -L "$BUNDLE" -l:libggml.so -l:libggml-base.so -ldl -lm

qemu_args=()
if [ -n "${QEMU_SYSROOT:-}" ]; then
  qemu_args=(-L "$QEMU_SYSROOT")
fi
# cortex-a53/a72: ARMv8.0. neoverse-n1: DOTPROD+FP16, no I8MM.
# neoverse-v1/n2: I8MM (with SVE, which no shipped variant uses).
while read -r cpu expected; do
  echo "== qemu-aarch64 -cpu $cpu: expect $expected"
  qemu-aarch64 -cpu "$cpu" "${qemu_args[@]}" -E LD_LIBRARY_PATH="$BUNDLE" \
    "$work/smoke" "$BUNDLE" "$expected" < /dev/null
done <<'EOF'
cortex-a53 libggml-cpu.so
cortex-a72 libggml-cpu.so
neoverse-n1 libggml-cpu-linux_armv8.2_2.so
neoverse-v1 libggml-cpu-linux_armv8.6_1.so
neoverse-n2 libggml-cpu-linux_armv8.6_1.so
EOF
