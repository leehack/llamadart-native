#!/usr/bin/env python3
"""Inspect a failed sdkmanager r28c download without retrying or changing it."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time

NDK_VERSION = "28.2.13676358"
ARCHIVE_NAME = "android-ndk-r28c-linux.zip"
EXPECTED_SIZE = 722261334
EXPECTED_SHA1 = "a7b54a5de87fecd125a17d54f73c446199e72a64"
EXPECTED_SOURCE = "https://dl.google.com/android/repository/repository2-3.xml"
METADATA_PARTS = ("ndk", NDK_VERSION, ".installer", ".installData")
MAX_METADATA_BYTES = 65536
HASH_TIMEOUT_SECONDS = 30
ZIP_TIMEOUT_SECONDS = 60


class EvidenceIssue(Exception):
    def __init__(self, status: str, reason: str):
        self.status, self.reason = status, reason


def _open_relative(root_fd: int, parts: tuple[str, ...]) -> int:
    """Walk only beneath an already-open SDK root; never follow a symlink."""
    current = os.dup(root_fd)
    try:
        for index, part in enumerate(parts):
            if part in ("", ".", "..") or "/" in part:
                raise EvidenceIssue("rejected", "invalid-contained-path")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index != len(parts) - 1:
                flags |= os.O_DIRECTORY
            following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
        if not stat.S_ISREG(os.fstat(current).st_mode):
            raise EvidenceIssue("rejected", "evidence-not-regular-file")
        result, current = current, -1
        return result
    finally:
        if current >= 0:
            os.close(current)


def _property_value(value: str) -> str:
    result = []
    index = 0
    while index < len(value):
        character = value[index]
        index += 1
        if character == "\\":
            if index == len(value):
                raise EvidenceIssue("rejected", "invalid-installer-metadata")
            character = value[index]
            index += 1
            if character == "u":
                digits = value[index:index + 4]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                    raise EvidenceIssue("rejected", "invalid-installer-metadata")
                character = chr(int(digits, 16))
                index += 4
            else:
                character = {"t": "\t", "r": "\r", "n": "\n", "f": "\f"}.get(character, character)
        result.append(character)
    return "".join(result)


def _operation_parts(data: bytes, sdk_input: Path, sdk_root: Path) -> tuple[str, str]:
    paths = []
    for line in data.decode("latin-1").splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        if len(stripped) - len(stripped.rstrip("\\")) & 1:
            raise EvidenceIssue("rejected", "invalid-installer-metadata")
        match = re.match(r"((?:\\.|[^:=\s])+)(?:\s*[:=]\s*|\s+)(.*)$", stripped)
        if match and _property_value(match.group(1)) == "path":
            paths.append(_property_value(match.group(2)))
    if len(paths) != 1 or len(paths[0]) > 4096 or any(ord(c) < 32 for c in paths[0]):
        raise EvidenceIssue("rejected", "invalid-installer-metadata")
    operation = Path(paths[0])
    if not operation.is_absolute() or ".." in operation.parts:
        raise EvidenceIssue("rejected", "installer-path-outside-sdk-temp")
    for root in (sdk_input, sdk_root):
        try:
            relative = operation.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) == 2 and relative.parts[0] == ".temp" and re.fullmatch(r"PackageOperation(?:0[1-9]|[1-9][0-9])", relative.parts[1]):
            return relative.parts
    raise EvidenceIssue("rejected", "installer-path-outside-sdk-temp")


def _zip_integrity(descriptor: int) -> dict:
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        result = subprocess.run(
            ["/usr/bin/unzip", "-t", f"/dev/fd/{descriptor}"],
            pass_fds=(descriptor,), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=ZIP_TIMEOUT_SECONDS, check=False,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        return {"status": "pass" if result.returncode == 0 else "fail", "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except OSError:
        return {"status": "unavailable"}


def _archive_evidence(descriptor: int) -> dict:
    before = os.fstat(descriptor)
    result = {"size": before.st_size, "size_matches": before.st_size == EXPECTED_SIZE}
    if before.st_size > EXPECTED_SIZE:
        return {**result, "hash_status": "skipped-oversize", "zip_integrity": {"status": "skipped-oversize"}}
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    deadline = time.monotonic() + HASH_TIMEOUT_SECONDS
    count = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        count += len(chunk)
        if count > EXPECTED_SIZE or time.monotonic() > deadline:
            raise EvidenceIssue("error", "archive-read-bound-exceeded")
        sha1.update(chunk)
        sha256.update(chunk)
    after = os.fstat(descriptor)
    if count != before.st_size or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise EvidenceIssue("error", "archive-changed-during-read")
    result.update(hash_status="complete", sha1=sha1.hexdigest(), sha256=sha256.hexdigest(), sha1_matches=sha1.hexdigest() == EXPECTED_SHA1)
    result["zip_integrity"] = _zip_integrity(descriptor)
    final = os.fstat(descriptor)
    if (final.st_size, final.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise EvidenceIssue("error", "archive-changed-during-read")
    return result


def capture(sdk_path: Path, setup_outcome: str) -> dict:
    """Return bounded metadata only; errors never echo SDK metadata or payloads."""
    receipt = {
        "schema": 1, "kind": "android-ndk-install-diagnostics",
        "setup_outcome": setup_outcome, "read_only": True,
        "expected": {"ndk_version": NDK_VERSION, "archive": ARCHIVE_NAME,
                     "size": EXPECTED_SIZE, "sha1": EXPECTED_SHA1, "source": EXPECTED_SOURCE},
    }
    if setup_outcome != "failure":
        return {**receipt, "status": "skipped", "reason": "setup-did-not-fail"}
    root_fd = metadata_fd = archive_fd = -1
    stage = "sdk-root"
    try:
        sdk_input = sdk_path.absolute()
        sdk_root = sdk_input.resolve(strict=True)
        root_fd = os.open(sdk_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stage = "installer-metadata"
        metadata_fd = _open_relative(root_fd, METADATA_PARTS)
        if os.fstat(metadata_fd).st_size > MAX_METADATA_BYTES:
            raise EvidenceIssue("rejected", "installer-metadata-too-large")
        data = os.read(metadata_fd, MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise EvidenceIssue("rejected", "installer-metadata-too-large")
        operation = _operation_parts(data, sdk_input, sdk_root)
        stage = "retained-archive"
        archive_fd = _open_relative(root_fd, (*operation, ARCHIVE_NAME))
        archive = _archive_evidence(archive_fd)
        return {**receipt, "status": "captured", "operation_directory": "/".join(operation), "archive": archive}
    except EvidenceIssue as error:
        return {**receipt, "status": error.status, "reason": error.reason}
    except OSError as error:
        status = "unavailable" if error.errno == errno.ENOENT else "rejected" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "error"
        return {**receipt, "status": status, "reason": stage + "-unavailable", "errno": error.errno}
    except (ValueError, RuntimeError):
        return {**receipt, "status": "rejected", "reason": "invalid-installer-metadata"}
    finally:
        for descriptor in (archive_fd, metadata_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--setup-outcome", choices=("failure", "success", "skipped", "cancelled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.dumps(capture(args.sdk_root, args.setup_outcome), sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as output:
        output.write(report)
        staged = Path(output.name)
    staged.replace(args.output)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
