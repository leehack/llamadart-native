#!/usr/bin/env bash
set -euo pipefail

tag="${1:?Usage: generate_assets_manifest.sh <tag> <assets_dir> <output_json> <output_checksums>}"
assets_dir="${2:?Usage: generate_assets_manifest.sh <tag> <assets_dir> <output_json> <output_checksums>}"
output_json="${3:?Usage: generate_assets_manifest.sh <tag> <assets_dir> <output_json> <output_checksums>}"
output_checksums="${4:?Usage: generate_assets_manifest.sh <tag> <assets_dir> <output_json> <output_checksums>}"

if [ ! -d "$assets_dir" ]; then
  echo "Assets directory not found: $assets_dir" >&2
  exit 1
fi

hash_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "No SHA256 tool found (sha256sum/shasum)." >&2
    exit 1
  fi
}

strip_ext() {
  local base="$1"
  case "$base" in
    *.dylib)
      echo "${base%.dylib}"
      ;;
    *.dll)
      echo "${base%.dll}"
      ;;
    *.so)
      echo "${base%.so}"
      ;;
    *.so.*)
      echo "${base%%.so*}"
      ;;
    *)
      echo "${base%.*}"
      ;;
  esac
}

infer_meta() {
  local file="$1"
  local base stem platform="unknown" arch="unknown" backend="unknown" module="core" libid=""

  base="$(basename "$file")"
  stem="$(strip_ext "$base")"
  libid="$stem"

  # New naming convention: <libname>-<platform>-<arch>.<ext>
  case "$stem" in
    *-windows-x64)
      platform="windows"; arch="x64"; libid="${stem%-windows-x64}" ;;
    *-linux-x64)
      platform="linux"; arch="x64"; libid="${stem%-linux-x64}" ;;
    *-linux-arm64)
      platform="linux"; arch="arm64"; libid="${stem%-linux-arm64}" ;;
    *-macos-arm64)
      platform="macos"; arch="arm64"; libid="${stem%-macos-arm64}" ;;
    *-macos-x86_64)
      platform="macos"; arch="x86_64"; libid="${stem%-macos-x86_64}" ;;
    *-ios-arm64-sim)
      platform="ios"; arch="arm64-sim"; libid="${stem%-ios-arm64-sim}" ;;
    *-ios-x86_64-sim)
      platform="ios"; arch="x86_64-sim"; libid="${stem%-ios-x86_64-sim}" ;;
    *-ios-arm64)
      platform="ios"; arch="arm64"; libid="${stem%-ios-arm64}" ;;
    *-android-arm64)
      platform="android"; arch="arm64"; libid="${stem%-android-arm64}" ;;
    *-android-x64)
      platform="android"; arch="x64"; libid="${stem%-android-x64}" ;;
    *)
      ;;
  esac

  # Backward compatibility for old naming: libllamadart-<platform>-<arch>-<backend>
  if [ "$platform" = "unknown" ]; then
    local legacy
    legacy="${stem#libllamadart-}"
    case "$legacy" in
      windows-x64-*)
        platform="windows"; arch="x64"; backend="${legacy#windows-x64-}"; module="backend-$backend" ;;
      linux-x64-*)
        platform="linux"; arch="x64"; backend="${legacy#linux-x64-}"; module="backend-$backend" ;;
      linux-arm64-*)
        platform="linux"; arch="arm64"; backend="${legacy#linux-arm64-}"; module="backend-$backend" ;;
      macos-arm64-*)
        platform="macos"; arch="arm64"; backend="${legacy#macos-arm64-}"; module="backend-$backend" ;;
      macos-x86_64-*)
        platform="macos"; arch="x86_64"; backend="${legacy#macos-x86_64-}"; module="backend-$backend" ;;
      ios-arm64-sim-*)
        platform="ios"; arch="arm64-sim"; backend="${legacy#ios-arm64-sim-}"; module="backend-$backend" ;;
      ios-x86_64-sim-*)
        platform="ios"; arch="x86_64-sim"; backend="${legacy#ios-x86_64-sim-}"; module="backend-$backend" ;;
      ios-arm64-*)
        platform="ios"; arch="arm64"; backend="${legacy#ios-arm64-}"; module="backend-$backend" ;;
      android-arm64-*)
        platform="android"; arch="arm64"; backend="${legacy#android-arm64-}"; module="backend-$backend" ;;
      android-x64-*)
        platform="android"; arch="x64"; backend="${legacy#android-x64-}"; module="backend-$backend" ;;
      *)
        ;;
    esac

    echo "$platform|$arch|$backend|$module"
    return
  fi

  local id_no_lib
  id_no_lib="${libid#lib}"

  case "$id_no_lib" in
    ggml-base|ggml|llama|llamadart)
      backend="core"
      module="core"
      ;;
    mtmd)
      backend="mtmd"
      module="mtmd"
      ;;
    ggml-*)
      local ggml_backend
      ggml_backend="${id_no_lib#ggml-}"
      case "$ggml_backend" in
        cpu*) backend="cpu" ;;
        vulkan*) backend="vulkan" ;;
        opencl*) backend="opencl" ;;
        cuda*) backend="cuda" ;;
        metal*) backend="metal" ;;
        zendnn*) backend="zendnn" ;;
        blas*) backend="blas" ;;
        sycl*) backend="sycl" ;;
        hip*) backend="hip" ;;
        rpc*) backend="rpc" ;;
        webgpu*) backend="webgpu" ;;
        hexagon*) backend="hexagon" ;;
        cann*) backend="cann" ;;
        musa*) backend="musa" ;;
        virtgpu*) backend="virtgpu" ;;
        zdnn*) backend="zdnn" ;;
        *) backend="$ggml_backend" ;;
      esac
      module="backend-$backend"
      ;;
    *)
      backend="core"
      module="core"
      ;;
  esac

  echo "$platform|$arch|$backend|$module"
}

files="$(find "$assets_dir" -maxdepth 1 -type f | sort)"
if [ -z "$files" ]; then
  echo "No assets found in $assets_dir" >&2
  exit 1
fi

count="$(printf '%s\n' "$files" | sed '/^$/d' | wc -l | tr -d ' ')"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

: > "$output_checksums"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  b="$(basename "$f")"
  sha="$(hash_file "$f")"
  printf '%s  %s\n' "$sha" "$b" >> "$output_checksums"
done <<EOF_FILES
$files
EOF_FILES

{
  echo "{"
  echo "  \"tag\": \"$tag\"," 
  echo "  \"generated_at\": \"$timestamp\"," 
  echo "  \"hook_contract_version\": 1,"
  echo "  \"artifacts\": ["

  i=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    i=$((i + 1))

    b="$(basename "$f")"
    sha="$(hash_file "$f")"
    size="$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")"

    meta="$(infer_meta "$f")"
    platform="${meta%%|*}"
    rest="${meta#*|}"
    arch="${rest%%|*}"
    rest="${rest#*|}"
    backend="${rest%%|*}"
    module="${rest#*|}"

    comma=","
    if [ "$i" -eq "$count" ]; then
      comma=""
    fi

    echo "    {"
    echo "      \"module\": \"$module\"," 
    echo "      \"platform\": \"$platform\"," 
    echo "      \"arch\": \"$arch\"," 
    echo "      \"backend\": \"$backend\"," 
    echo "      \"file\": \"$b\"," 
    echo "      \"sha256\": \"$sha\"," 
    echo "      \"size\": $size"
    echo "    }$comma"
  done <<EOF_FILES
$files
EOF_FILES

  echo "  ]"
  echo "}"
} > "$output_json"

echo "Generated $output_json and $output_checksums"
