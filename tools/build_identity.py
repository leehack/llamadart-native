#!/usr/bin/env python3
"""Writes the build identity libllamadart reports; see docs/build_identity.md."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import llama_cpp_patches

FORMAT = "llamadart-native-build/1"
# No path separator, drive colon or space can appear, so no path can either.
VALUE = re.compile(r"[A-Za-z0-9._+,-]+")
DIGEST_LENGTH = 16
# Pinned so that the digest of a dirty tree does not follow diff settings in
# the user's git config. Options that have no command-line form are pinned as
# config in _git.
DIFF_OPTIONS = (
    "--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv",
    "--no-renames", "--no-relative", "--submodule=short", "--diff-algorithm=myers",
    "--no-indent-heuristic", "--unified=3", "--src-prefix=a/", "--dst-prefix=b/",
    "--inter-hunk-context=0",
)
DIFF_CONFIG = ("diff.suppressBlankEmpty=false",)


def _empty_order_file() -> str:
    # An empty diff.orderFile value is an error, so the setting is pinned to a
    # file that orders nothing.
    handle, name = tempfile.mkstemp(prefix="llamadart-empty-order-")
    os.close(handle)
    atexit.register(os.unlink, name)
    return name


_EMPTY_ORDER_FILE: str | None = None


class Unidentified(Exception):
    """The build cannot name one of its inputs."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:DIGEST_LENGTH]


def _git(repo: Path, *args: str) -> bytes:
    # A container build may run as another user than the checkout's owner.
    command = ["git", "-c", "safe.directory=*", "-c", "core.quotePath=true"]
    global _EMPTY_ORDER_FILE
    if _EMPTY_ORDER_FILE is None:
        _EMPTY_ORDER_FILE = _empty_order_file()
    for setting in (f"diff.orderFile={_EMPTY_ORDER_FILE}", *DIFF_CONFIG):
        command += ["-c", setting]
    command += ["-C", str(repo), *args]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise Unidentified(f"cannot run git: {error.strerror}") from error
    if completed.returncode != 0:
        raise Unidentified(f"git {args[0]} failed in {repo.name}")
    return completed.stdout


def revision(repo: Path, ignore_submodules: str | None = None) -> str:
    """The commit checked out in repo, with a digest of any tracked change."""
    top = _git(repo, "rev-parse", "--show-toplevel").decode().strip()
    # A source tree copied into another checkout would report that one.
    try:
        same = os.path.samefile(top, repo)
    except OSError as error:
        raise Unidentified(f"cannot compare {repo.name} with git's top level") from error
    if not same:
        raise Unidentified(f"{repo.name} is not the top of a git checkout")
    commit = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    options = list(DIFF_OPTIONS)
    if ignore_submodules:
        options.append(f"--ignore-submodules={ignore_submodules}")
    diff = _git(repo, "diff", *options, "HEAD", "--")
    return f"{commit}+dirty.{_digest(diff)}" if diff else commit


def patches_digest(source: Path) -> str:
    """Digest of the sources the carried patches that fit this llama.cpp produce."""
    patched, _ = llama_cpp_patches.apply_available(
        source / "third_party" / "llama.cpp", source / "patches" / "llama.cpp")
    if not patched:
        return "none"
    data = b"".join(
        path.encode() + b"\0" + text.encode() + b"\0"
        for path, text in sorted(patched.items()))
    return _digest(data)


def identity(source: Path, *, config: str, compiler: str, target: str,
             backends: str, options: str) -> str:
    """The identity string, or "" when the build cannot name its inputs."""
    upstream = source / "third_party" / "llama.cpp"
    try:
        fields = [
            # Submodule commits are part of the owner tree; their worktrees
            # are not, and llama.cpp's has its own field.
            ("owner", revision(source, ignore_submodules="dirty")),
            ("llama.cpp", revision(upstream)),
            ("ggml", _git(upstream, "rev-parse", "--verify", "HEAD:ggml").decode().strip()),
        ]
    except Unidentified as error:
        print(f"build identity: unidentified build: {error}", file=sys.stderr)
        return ""
    fields += [
        ("patches", patches_digest(source)),
        ("config", config or "none"),
        ("compiler", compiler),
        ("target", target),
        ("backends", backends or "none"),
        ("options", options),
    ]
    for key, value in fields:
        if VALUE.fullmatch(value) is None:
            raise ValueError(f"build identity: {key} must match {VALUE.pattern}: {value!r}")
    return ";".join([FORMAT] + [f"{key}={value}" for key, value in fields])


def write_header(output: Path, value: str) -> bool:
    """Writes the header unless it already holds value; returns whether it wrote."""
    text = (
        "// Generated by tools/build_identity.py.\n"
        "#pragma once\n"
        f"#define LLAMA_DART_BUILD_IDENTITY \"{value}\"\n"
    )
    # An unchanged header keeps its timestamp, so nothing is rebuilt.
    if output.is_file() and output.read_text(encoding="utf-8") == text:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("config", "compiler", "target", "backends", "options"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        value = identity(
            args.source.resolve(), config=args.config, compiler=args.compiler,
            target=args.target, backends=args.backends, options=args.options)
    except (ValueError, llama_cpp_patches.PatchError) as error:
        print(error, file=sys.stderr)
        return 1
    write_header(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
