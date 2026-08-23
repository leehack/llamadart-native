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

json_string() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

infer_meta() {
  local file="$1"
  local base platform="unknown" arch="unknown" backend="unknown" module="core"

  base="$(basename "$file")"

  case "$base" in
    "llamadart-native-apple-xcframework-$tag.zip")
      platform="apple"; arch="universal"; backend="core"; module="spm-xcframework" ;;
    "llamadart-native-headers-$tag.tar.gz")
      platform="all"; arch="universal"; backend="core"; module="headers" ;;
    "llamadart-native-windows-x64-$tag.tar.gz")
      platform="windows"; arch="x64"; backend="core" ;;
    "llamadart-native-windows-arm64-$tag.tar.gz")
      platform="windows"; arch="arm64"; backend="core" ;;
    "llamadart-native-linux-x64-$tag.tar.gz")
      platform="linux"; arch="x64"; backend="core" ;;
    "llamadart-native-linux-arm64-$tag.tar.gz")
      platform="linux"; arch="arm64"; backend="core" ;;
    "llamadart-native-macos-arm64-$tag.tar.gz")
      platform="macos"; arch="arm64"; backend="core" ;;
    "llamadart-native-macos-x86_64-$tag.tar.gz")
      platform="macos"; arch="x86_64"; backend="core" ;;
    "llamadart-native-ios-arm64-$tag.tar.gz")
      platform="ios"; arch="arm64"; backend="core" ;;
    "llamadart-native-ios-arm64-sim-$tag.tar.gz")
      platform="ios"; arch="arm64-sim"; backend="core" ;;
    "llamadart-native-ios-x86_64-sim-$tag.tar.gz")
      platform="ios"; arch="x86_64-sim"; backend="core" ;;
    "llamadart-native-android-arm64-$tag.tar.gz")
      platform="android"; arch="arm64"; backend="core" ;;
    "llamadart-native-android-x64-$tag.tar.gz")
      platform="android"; arch="x64"; backend="core" ;;
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
llama_cpp_tag="${LLAMADART_LLAMA_CPP_TAG:-$tag}"
llama_cpp_commit="${LLAMADART_LLAMA_CPP_COMMIT:-}"
native_commit="${LLAMADART_NATIVE_COMMIT:-}"
correlation_id="${LLAMADART_CORRELATION_ID:-}"
smoke_policy="${LLAMADART_SMOKE_POLICY:-}"
smoke_conclusion="${LLAMADART_SMOKE_CONCLUSION:-}"
workflow_run_id="${LLAMADART_WORKFLOW_RUN_ID:-}"
workflow_run_url="${LLAMADART_WORKFLOW_RUN_URL:-}"
workflow_head_sha="${LLAMADART_WORKFLOW_HEAD_SHA:-}"

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
  # Keep tag as an immutable compatibility alias for existing consumers while
  # making native artifact-version ownership explicit for new manifests.
  printf '  "tag": %s,\n' "$(json_string "$tag")"
  printf '  "native_release_tag": %s,\n' "$(json_string "$tag")"
  printf '  "llama_cpp_tag": %s,\n' "$(json_string "$llama_cpp_tag")"
  if [ -n "$llama_cpp_commit" ]; then
    printf '  "llama_cpp_commit": %s,\n' "$(json_string "$llama_cpp_commit")"
  fi
  if [ -n "$native_commit" ]; then
    printf '  "native_commit": %s,\n' "$(json_string "$native_commit")"
  fi
  if [ -n "$correlation_id" ]; then
    printf '  "correlation_id": %s,\n' "$(json_string "$correlation_id")"
  fi
  if [ -n "$smoke_policy" ]; then
    printf '  "smoke_policy": %s,\n' "$(json_string "$smoke_policy")"
  fi
  if [ -n "$smoke_conclusion" ]; then
    printf '  "smoke_conclusion": %s,\n' "$(json_string "$smoke_conclusion")"
  fi
  if [ -n "$workflow_run_id" ]; then
    printf '  "workflow_run_id": %s,\n' "$(json_string "$workflow_run_id")"
  fi
  if [ -n "$workflow_run_url" ]; then
    printf '  "workflow_run_url": %s,\n' "$(json_string "$workflow_run_url")"
  fi
  if [ -n "$workflow_head_sha" ]; then
    printf '  "workflow_head_sha": %s,\n' "$(json_string "$workflow_head_sha")"
  fi
  printf '  "generated_at": %s,\n' "$(json_string "$timestamp")"
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
    printf '      "module": %s,\n' "$(json_string "$module")"
    printf '      "platform": %s,\n' "$(json_string "$platform")"
    printf '      "arch": %s,\n' "$(json_string "$arch")"
    printf '      "backend": %s,\n' "$(json_string "$backend")"
    printf '      "file": %s,\n' "$(json_string "$b")"
    printf '      "sha256": %s,\n' "$(json_string "$sha")"
    echo "      \"size\": $size"
    echo "    }$comma"
  done <<EOF_FILES
$files
EOF_FILES

  echo "  ]"
  echo "}"
} > "$output_json"

echo "Generated $output_json and $output_checksums"
