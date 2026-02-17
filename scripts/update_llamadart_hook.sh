#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:?Usage: update_llamadart_hook.sh <llamadart_repo_dir> <tag> <native_repo_slug>}"
tag="${2:?Usage: update_llamadart_hook.sh <llamadart_repo_dir> <tag> <native_repo_slug>}"
native_repo_slug="${3:?Usage: update_llamadart_hook.sh <llamadart_repo_dir> <tag> <native_repo_slug>}"

hook_file="$repo_dir/hook/build.dart"
if [ ! -f "$hook_file" ]; then
  echo "hook/build.dart not found at $hook_file" >&2
  exit 1
fi

HOOK_FILE="$hook_file" TAG="$tag" NATIVE_REPO="$native_repo_slug" python3 - <<'PY'
from pathlib import Path
import os
import re

hook = Path(os.environ["HOOK_FILE"])
tag = os.environ["TAG"]
native_repo = os.environ["NATIVE_REPO"]

text = hook.read_text(encoding="utf-8")

text = re.sub(
    r"const _llamaCppTag = '.*?';",
    f"const _llamaCppTag = '{tag}';",
    text,
)

pattern = r"(const _baseUrl\s*=\s*\n\s*')https://github.com/[^']+/releases/download/\$_llamaCppTag(';)"
replacement = r"\1" + f"https://github.com/{native_repo}/releases/download/$_llamaCppTag" + r"\2"
text = re.sub(pattern, replacement, text)

hook.write_text(text, encoding="utf-8")
PY

echo "Updated $hook_file for tag=$tag repo=$native_repo_slug"
