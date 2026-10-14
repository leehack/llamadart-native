"""Derive framework Info.plist OS metadata from a Mach-O binary's build version."""

from __future__ import annotations

import struct


LC_BUILD_VERSION = 0x32
FAT_MAGIC = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
MH_MAGIC = 0xFEEDFACE
MH_MAGIC_64 = 0xFEEDFACF

# Mach-O platform -> (CFBundleSupportedPlatforms entry, minimum-version key).
# iOS bundles use MinimumOSVersion; macOS bundles use LSMinimumSystemVersion.
PLATFORMS = {
    1: ("MacOSX", "LSMinimumSystemVersion"),
    2: ("iPhoneOS", "MinimumOSVersion"),
    7: ("iPhoneSimulator", "MinimumOSVersion"),
}


def _thin_build_versions(data: bytes, offset: int) -> list[tuple[int, int]]:
    magic = struct.unpack_from("<I", data, offset)[0]
    if magic == MH_MAGIC_64:
        header_size = 32
    elif magic == MH_MAGIC:
        header_size = 28
    else:
        raise ValueError(f"not a little-endian Mach-O image at offset {offset}")
    ncmds = struct.unpack_from("<I", data, offset + 16)[0]
    versions = []
    command = offset + header_size
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, command)
        if size < 8:
            raise ValueError("malformed Mach-O load command")
        if cmd == LC_BUILD_VERSION:
            versions.append(struct.unpack_from("<II", data, command + 8))
        command += size
    return versions


def build_versions(data: bytes) -> list[tuple[int, int]]:
    """Return the (platform, minos) of every LC_BUILD_VERSION in every architecture."""
    try:
        magic = struct.unpack_from(">I", data, 0)[0]
        if magic not in (FAT_MAGIC, FAT_MAGIC_64):
            return _thin_build_versions(data, 0)
        count = struct.unpack_from(">I", data, 4)[0]
        stride, layout = (20, ">iiIII") if magic == FAT_MAGIC else (32, ">iiQQII")
        versions = []
        for index in range(count):
            offset = struct.unpack_from(layout, data, 8 + index * stride)[2]
            arch_versions = _thin_build_versions(data, offset)
            if not arch_versions:
                raise ValueError(f"architecture {index} has no LC_BUILD_VERSION")
            versions.extend(arch_versions)
        return versions
    except struct.error as error:
        raise ValueError(f"truncated Mach-O binary ({error})") from error


def format_version(encoded: int) -> str:
    """Format a Mach-O xxxx.yy.zz nibble-encoded version the way Xcode writes it."""
    major, minor, patch = encoded >> 16, (encoded >> 8) & 0xFF, encoded & 0xFF
    return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"


def framework_os_metadata(data: bytes) -> dict[str, object]:
    """Return the Info.plist platform keys that describe this framework binary.

    Raises ValueError unless every architecture targets the same supported
    platform and deployment minimum.
    """
    versions = set(build_versions(data))
    if not versions:
        raise ValueError("binary has no LC_BUILD_VERSION load command")
    if len(versions) != 1:
        raise ValueError(
            "architectures disagree on platform or deployment minimum: "
            + ", ".join(f"{p}/{format_version(m)}" for p, m in sorted(versions))
        )
    platform, minos = versions.pop()
    if platform not in PLATFORMS:
        raise ValueError(f"unsupported Mach-O platform {platform}")
    supported_platform, minimum_key = PLATFORMS[platform]
    return {
        "CFBundleSupportedPlatforms": [supported_platform],
        minimum_key: format_version(minos),
    }
