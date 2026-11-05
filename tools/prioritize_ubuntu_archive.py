#!/usr/bin/env python3
"""Prefer the configured official HTTPS archive in the Ubuntu 24.04 CI image."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
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


def prioritize_archive(path: Path) -> None:
    # Refuse links and avoid blocking on a FIFO before checking the file type.
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("APT mirrorlist must be a regular file")
    with os.fdopen(descriptor, "r+b") as mirrors:
        current = mirrors.read()
        if current == PREFERRED:
            return
        if current != ORIGINAL:
            raise ValueError("Unexpected APT mirrorlist; refusing to change mirror priorities")
        # Keep the existing inode, owner and mode; only the two priorities change.
        mirrors.seek(0)
        mirrors.write(PREFERRED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("/etc/apt/apt-mirrors.txt"))
    args = parser.parse_args()
    try:
        prioritize_archive(args.path)
    except (OSError, ValueError) as error:
        print(f"Cannot prioritize Ubuntu archive: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
