#!/usr/bin/env python3
"""Prefer the configured official HTTPS archive in the Ubuntu 24.04 CI image.

Failure-only safe receipts: docs/apt_mirror_diagnostics.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys

# actions/runner-images ubuntu24/20261101.360, configure-apt-sources.sh.
ORIGINAL = (
    b"http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n"
    b"https://archive.ubuntu.com/ubuntu/\tpriority:2\n"
    b"https://security.ubuntu.com/ubuntu/\tpriority:3\n"
)
PREFERRED = (
    b"http://azure.archive.ubuntu.com/ubuntu/\tpriority:2\n"
    b"https://archive.ubuntu.com/ubuntu/\tpriority:1\n"
    b"https://security.ubuntu.com/ubuntu/\tpriority:3\n"
)

MIRRORLIST_PATH = Path("/etc/apt/apt-mirrors.txt")
MAX_BYTES = 4096
MAX_LINES = 16
MAX_LINE_BYTES = 256
MAX_RECEIPT_BYTES = 8192
REFUSAL = "Unexpected APT mirrorlist; refusing to change mirror priorities"
MIRROR_IDS = {
    b"http://azure.archive.ubuntu.com/ubuntu/": "azure_archive",
    b"https://archive.ubuntu.com/ubuntu/": "ubuntu_archive",
    b"https://security.ubuntu.com/ubuntu/": "ubuntu_security",
}
# Classification is diagnostic only. This does not expand accepted byte formats.
KNOWN_ROW = re.compile(
    rb"([ \t]{0,32})(" + b"|".join(re.escape(url) for url in MIRROR_IDS) +
    rb")([ \t]{1,32})priority:([1-9][0-9]{0,2})([ \t]{0,32})(\r\n|\n)?"
)


class MirrorlistRefused(ValueError):
    def __init__(self, diagnostic: dict | None = None) -> None:
        super().__init__(REFUSAL)
        self.diagnostic = diagnostic


def _whitespace(value: bytes) -> str:
    return value.replace(b" ", b"s").replace(b"\t", b"t").decode("ascii")


def _diagnostic(current: bytes) -> dict | None:
    # Only LF delimits rows; an unexpected bare CR remains unclassified.
    parts = current.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    if len(lines) > MAX_LINES or any(len(line) > MAX_LINE_BYTES for line in lines):
        return None
    rows = []
    for line in lines:
        match = KNOWN_ROW.fullmatch(line)
        if not match:
            rows.append({"mirror_id": "unclassified"})
            continue
        leading, url, separator, priority, trailing, ending = match.groups()
        # Encode only already-validated whitespace, never raw configuration.
        rows.append({
            "mirror_id": MIRROR_IDS[url], "priority": int(priority),
            "leading_ws": _whitespace(leading), "separator_ws": _whitespace(separator),
            "trailing_ws": _whitespace(trailing),
            "ending": {b"\n": "lf", b"\r\n": "crlf", None: "none"}[ending],
        })
    crlf = current.count(b"\r\n")
    return {
        "schema": "llamadart-apt-mirror-diagnostic/1",
        "byte_count": len(current), "line_count": len(lines),
        "trailing_lf": current.endswith(b"\n"),
        "newlines": {"lf": current.count(b"\n") - crlf,
                     "crlf": crlf, "bare_cr": current.count(b"\r") - crlf},
        "lines": rows,
    }


def prioritize_archive(path: Path) -> None:
    # Refuse links and avoid blocking on a FIFO before checking the file type.
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("APT mirrorlist must be a regular file")
        if before.st_size > MAX_BYTES:
            raise MirrorlistRefused()
        current = os.read(descriptor, MAX_BYTES + 1)
        after = os.fstat(descriptor)
        if (len(current) > MAX_BYTES or len(current) != before.st_size or
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns) !=
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns)):
            raise MirrorlistRefused()
        if current == PREFERRED:
            return
        if current != ORIGINAL:
            raise MirrorlistRefused(_diagnostic(current))
        # Keep the existing inode, owner and mode; only the two priorities change.
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.write(descriptor, PREFERRED) != len(PREFERRED):
            raise OSError("incomplete mirror priority write")
    finally:
        os.close(descriptor)


def main() -> int:
    # Production accepts no path/configuration argument and reads no environment.
    if len(sys.argv) != 1:
        print(f"Cannot prioritize Ubuntu archive: {REFUSAL}", file=sys.stderr)
        return 1
    diagnostic = None
    try:
        prioritize_archive(MIRRORLIST_PATH)
        return 0
    except MirrorlistRefused as refusal:
        diagnostic = refusal.diagnostic
    except (OSError, ValueError):
        pass
    # Identical generic refusal for unsafe paths, read failures and size limits.
    # Never print exception text, paths, arbitrary URLs, hashes or environment.
    print(f"Cannot prioritize Ubuntu archive: {REFUSAL}", file=sys.stderr)
    if diagnostic is not None:
        receipt = json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
        if len(receipt) <= MAX_RECEIPT_BYTES:
            print("APT mirrorlist diagnostic: " + receipt, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
