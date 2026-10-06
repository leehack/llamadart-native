#!/usr/bin/env python3
"""Validate the privacy manifest of every slice in an Apple XCFramework zip."""

from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


MANIFEST_NAME = "PrivacyInfo.xcprivacy"

FILE_TIMESTAMP = "NSPrivacyAccessedAPICategoryFileTimestamp"
SYSTEM_BOOT_TIME = "NSPrivacyAccessedAPICategorySystemBootTime"
DISK_SPACE = "NSPrivacyAccessedAPICategoryDiskSpace"
ACTIVE_KEYBOARDS = "NSPrivacyAccessedAPICategoryActiveKeyboards"
USER_DEFAULTS = "NSPrivacyAccessedAPICategoryUserDefaults"

# Apple's approved reasons per required-reason API category:
# https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacyaccessedapitypes/nsprivacyaccessedapitypereasons
APPROVED_REASONS = {
    FILE_TIMESTAMP: frozenset({"DDA9.1", "C617.1", "3B52.1", "0A2A.1"}),
    SYSTEM_BOOT_TIME: frozenset({"35F9.1", "8FFB.1", "3D61.1"}),
    DISK_SPACE: frozenset({"85F4.1", "E174.1", "7D9E.1", "B728.1"}),
    ACTIVE_KEYBOARDS: frozenset({"3EC4.1", "54BD.1"}),
    USER_DEFAULTS: frozenset({"CA92.1", "1C8F.1", "C56D.1", "AC6B.1"}),
}

# Undefined Mach-O symbols and Objective-C selectors that reach the APIs listed in
# https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacyaccessedapitypes/nsprivacyaccessedapitype
REQUIRED_REASON_SYMBOLS = {
    FILE_TIMESTAMP: frozenset(
        {
            "_stat",
            "_fstat",
            "_fstatat",
            "_lstat",
            "_getattrlist",
            "_getattrlistbulk",
            "_fgetattrlist",
            "_getattrlistat",
            "_NSFileCreationDate",
            "_NSFileModificationDate",
            "_NSURLContentModificationDateKey",
            "_NSURLCreationDateKey",
        }
    ),
    SYSTEM_BOOT_TIME: frozenset({"_mach_absolute_time"}),
    DISK_SPACE: frozenset(
        {
            "_statfs",
            "_statvfs",
            "_fstatfs",
            "_fstatvfs",
            "_getattrlist",
            "_fgetattrlist",
            "_getattrlistat",
            "_NSFileSystemFreeSize",
            "_NSFileSystemSize",
            "_NSURLVolumeAvailableCapacityKey",
            "_NSURLVolumeAvailableCapacityForImportantUsageKey",
            "_NSURLVolumeAvailableCapacityForOpportunisticUsageKey",
            "_NSURLVolumeTotalCapacityKey",
        }
    ),
    ACTIVE_KEYBOARDS: frozenset({"_OBJC_CLASS_$_UITextInputMode"}),
    USER_DEFAULTS: frozenset({"_OBJC_CLASS_$_NSUserDefaults"}),
}
REQUIRED_REASON_SELECTORS = {
    FILE_TIMESTAMP: frozenset({"fileModificationDate"}),
    SYSTEM_BOOT_TIME: frozenset({"systemUptime"}),
    ACTIVE_KEYBOARDS: frozenset({"activeInputModes"}),
}
OBJC_CLASS_PREFIX = "_OBJC_CLASS_$_"


def validate_manifest(data: bytes) -> tuple[list[str], set[str]]:
    """Return manifest errors and the required-reason categories it declares."""
    try:
        manifest = plistlib.loads(data)
    except Exception as error:  # plistlib raises several unrelated types
        return [f"is not a valid property list ({type(error).__name__})"], set()
    if not isinstance(manifest, dict):
        return ["root object is not a dictionary"], set()

    errors: list[str] = []
    if manifest.get("NSPrivacyTracking") is not False:
        errors.append("NSPrivacyTracking must be false")
    for key in ("NSPrivacyTrackingDomains", "NSPrivacyCollectedDataTypes"):
        if manifest.get(key) != []:
            errors.append(f"{key} must be an empty array")

    declared: set[str] = set()
    accessed = manifest.get("NSPrivacyAccessedAPITypes")
    if not isinstance(accessed, list):
        return [*errors, "NSPrivacyAccessedAPITypes must be an array"], declared
    for entry in accessed:
        category = entry.get("NSPrivacyAccessedAPIType") if isinstance(entry, dict) else None
        if not isinstance(category, str) or category not in APPROVED_REASONS:
            errors.append(f"unknown required-reason API category: {category!r}")
            continue
        if category in declared:
            errors.append(f"{category} is declared more than once")
        declared.add(category)
        reasons = entry.get("NSPrivacyAccessedAPITypeReasons")
        if not isinstance(reasons, list) or not reasons:
            errors.append(f"{category} must declare at least one reason")
            continue
        for reason in reasons:
            if not isinstance(reason, str) or reason not in APPROVED_REASONS[category]:
                errors.append(f"{category} declares unapproved reason {reason!r}")
    return errors, declared


def manifest_member(framework: str, names: set[str]) -> str:
    """Return where Apple expects the manifest for this framework's layout."""
    versioned = any(name.startswith(f"{framework}/Versions/A/") for name in names)
    if versioned:
        return f"{framework}/Versions/A/Resources/{MANIFEST_NAME}"
    return f"{framework}/{MANIFEST_NAME}"


def normalize_symbol(name: str) -> str:
    if name.startswith(OBJC_CLASS_PREFIX):
        return name
    # x86_64 macOS binds variants such as _stat$INODE64.
    return name.split("$", 1)[0]


def required_categories(
    symbols: set[str], selectors: set[str]
) -> dict[str, list[str]]:
    """Map each required-reason category a binary uses to its evidence."""
    normalized = {normalize_symbol(symbol) for symbol in symbols}
    used: dict[str, list[str]] = {}
    for category in APPROVED_REASONS:
        evidence = sorted(
            (normalized & REQUIRED_REASON_SYMBOLS[category])
            | (selectors & REQUIRED_REASON_SELECTORS.get(category, frozenset()))
        )
        if evidence:
            used[category] = evidence
    return used


def audit_errors(declared: set[str], used: dict[str, list[str]]) -> list[str]:
    errors = [
        f"uses {category} via {', '.join(evidence)} but the manifest does not declare it"
        for category, evidence in used.items()
        if category not in declared
    ]
    errors.extend(
        f"manifest declares {category} but the binary uses none of its APIs"
        for category in sorted(declared - set(used))
    )
    return errors


def tool_output(command: list[str]) -> str:
    if not shutil.which(command[0]):
        raise ValueError(f"Mach-O inspection tool not found: {command[0]}")
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout


def binary_api_references(binary: Path) -> tuple[set[str], set[str]]:
    """Return undefined symbols and Objective-C selectors across all architectures."""
    # Without -arch all, nm reads only the host architecture of a fat binary.
    undefined = tool_output(["nm", "-u", "-arch", "all", str(binary)])
    methnames = tool_output(
        ["otool", "-arch", "all", "-v", "-s", "__TEXT", "__objc_methname", str(binary)]
    )
    symbols = {
        line.split()[-1]
        for line in undefined.splitlines()
        if line.strip() and not line.rstrip().endswith(":")
    }
    selectors = {
        match.group(1)
        for match in re.finditer(r"^[0-9a-f]+\s+(\S+)$", methnames, re.MULTILINE)
    }
    return symbols, selectors


def validate_archive(archive: Path, *, audit_imports: bool = False) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        roots = sorted(
            name
            for name in names
            if name.endswith(".xcframework/Info.plist") and name.count("/") == 1
        )
        if len(roots) != 1:
            return [f"expected exactly one XCFramework Info.plist, found {len(roots)}"]
        root = roots[0].rsplit("/", 1)[0]
        try:
            slices = [
                (
                    library["LibraryIdentifier"],
                    library["LibraryPath"],
                    library.get("BinaryPath", ""),
                )
                for library in plistlib.loads(bundle.read(roots[0]))["AvailableLibraries"]
            ]
        except Exception as error:
            return [f"{roots[0]} is not a valid XCFramework manifest ({type(error).__name__})"]
        if not slices:
            return [f"{roots[0]} lists no libraries"]

        for identifier, library_path, binary_path in slices:
            slice_root = f"{root}/{identifier}"
            expected = manifest_member(f"{slice_root}/{library_path}", names)
            found = {
                name
                for name in names
                if name.startswith(f"{slice_root}/") and name.endswith(f"/{MANIFEST_NAME}")
            }
            for stray in sorted(found - {expected}):
                errors.append(f"{identifier}: unexpected privacy manifest at {stray}")
            if expected not in found:
                errors.append(f"{identifier}: missing privacy manifest at {expected}")
                continue

            manifest_errors, declared = validate_manifest(bundle.read(expected))
            errors.extend(f"{identifier}: {expected} {error}" for error in manifest_errors)
            if manifest_errors or not audit_imports:
                continue

            binary_member = f"{slice_root}/{binary_path}"
            if binary_member not in names:
                errors.append(f"{identifier}: missing binary {binary_member}")
                continue
            with tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / "binary"
                binary.write_bytes(bundle.read(binary_member))
                used = required_categories(*binary_api_references(binary))
            errors.extend(
                f"{identifier}: {error}" for error in audit_errors(declared, used)
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--audit-imports",
        action="store_true",
        help=(
            "also require the declared categories to match the required-reason "
            "APIs each slice binary references (needs nm and otool)"
        ),
    )
    args = parser.parse_args()
    try:
        errors = validate_archive(args.archive, audit_imports=args.audit_imports)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        errors = [str(error)]
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"Validated Apple privacy manifests in {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
