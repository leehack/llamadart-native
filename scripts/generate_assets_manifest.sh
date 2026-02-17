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

infer_meta() {
  local file="$1"
  local base stem platform="unknown" arch="unknown" backend="unknown" module="core"

  base="$(basename "$file")"
  stem="${base#libllamadart-}"
  stem="${stem%.*}"

  case "$stem" in
    windows-x64-*)
      platform="windows"; arch="x64"; backend="${stem#windows-x64-}" ;;
    linux-x64-*)
      platform="linux"; arch="x64"; backend="${stem#linux-x64-}" ;;
    linux-arm64-*)
      platform="linux"; arch="arm64"; backend="${stem#linux-arm64-}" ;;
    macos-arm64-*)
      platform="macos"; arch="arm64"; backend="${stem#macos-arm64-}" ;;
    macos-x86_64-*)
      platform="macos"; arch="x86_64"; backend="${stem#macos-x86_64-}" ;;
    ios-arm64-sim-*)
      platform="ios"; arch="arm64-sim"; backend="${stem#ios-arm64-sim-}" ;;
    ios-x86_64-sim-*)
      platform="ios"; arch="x86_64-sim"; backend="${stem#ios-x86_64-sim-}" ;;
    ios-arm64-*)
      platform="ios"; arch="arm64"; backend="${stem#ios-arm64-}" ;;
    android-arm64-*)
      platform="android"; arch="arm64"; backend="${stem#android-arm64-}" ;;
    android-x64-*)
      platform="android"; arch="x64"; backend="${stem#android-x64-}" ;;
    *) ;;
  esac

  case "$backend" in
    cpu|vulkan|cuda|metal|opencl|zendnn|blas)
      module="backend-$backend"
      ;;
    mtmd)
      module="mtmd"
      ;;
    *)
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
