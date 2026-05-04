#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
DEFAULT_COMPONENTS = (
    "cuda_cccl",
    "cuda_crt",
    "cuda_cudart",
    "cuda_nvcc",
    "cuda_nvprof",
    "cuda_nvrtc",
    "cuda_nvtx",
    "cuda_profiler_api",
    "libcublas",
    "libnvfatbin",
    "libnvjitlink",
    "libnvvm",
    "visual_studio_integration",
)
REQUIRED_COMPONENTS = {
    "cuda_cudart",
    "cuda_nvcc",
    "libcublas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install selected CUDA redistributable archives."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="CUDA redistrib manifest version, for example 12.8.1.",
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=("linux-x86_64", "windows-x86_64"),
        help="CUDA redistrib platform key.",
    )
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="Directory where CUDA files will be flattened.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help=(
            "Component to install. May be repeated. Defaults to the minimal "
            "build/runtime set used by llamadart-native."
        ),
    )
    return parser.parse_args()


def download_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as file:
            shutil.copyfileobj(response, file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_contents(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zip_file:
            destination_root = destination.resolve()
            for member in zip_file.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination_root):
                    raise RuntimeError(
                        f"Refusing to extract unsafe ZIP path: {member.filename}"
                    )
            zip_file.extractall(destination)
        return

    if archive.name.endswith((".tar.xz", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as tar_file:
            destination_root = destination.resolve()
            for member in tar_file.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination_root):
                    raise RuntimeError(
                        f"Refusing to extract unsafe tar path: {member.name}"
                    )
            tar_file.extractall(destination)
        return

    raise RuntimeError(f"Unsupported archive type: {archive}")


def component_entry(
    manifest: dict[str, Any],
    component: str,
    platform_key: str,
) -> dict[str, Any] | None:
    component_data = manifest.get(component)
    if not isinstance(component_data, dict):
        return None
    entry = component_data.get(platform_key)
    if not isinstance(entry, dict):
        return None
    return entry


def install_component(
    component: str,
    entry: dict[str, Any],
    destination: Path,
    temp_dir: Path,
) -> None:
    relative_path = entry.get("relative_path")
    expected_sha256 = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise RuntimeError(f"Manifest entry for {component} is incomplete")

    archive_url = f"{BASE_URL}/{relative_path}"
    archive = temp_dir / Path(relative_path).name
    extracted = temp_dir / f"{component}-extract"

    print(f"Downloading {component}: {archive_url}")
    download_file(archive_url, archive)

    actual_sha256 = sha256(archive)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {component}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    extracted.mkdir()
    extract_archive(archive, extracted)
    roots = [path for path in extracted.iterdir()]
    if len(roots) == 1 and roots[0].is_dir():
        copy_tree_contents(roots[0], destination)
    else:
        copy_tree_contents(extracted, destination)


def main() -> None:
    args = parse_args()
    manifest_url = f"{BASE_URL}/redistrib_{args.version}.json"
    components = tuple(args.components or DEFAULT_COMPONENTS)

    print(f"Loading CUDA redistrib manifest: {manifest_url}")
    manifest = download_json(manifest_url)
    args.destination.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cuda-redist-") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        for component in components:
            entry = component_entry(manifest, component, args.platform)
            if entry is None:
                if component in REQUIRED_COMPONENTS:
                    raise RuntimeError(
                        f"Required CUDA component {component} is unavailable "
                        f"for {args.platform} in {args.version}"
                    )
                print(f"Skipping unavailable optional component: {component}")
                continue
            install_component(component, entry, args.destination, temp_dir)
            installed.append(component)

    print(
        "Installed CUDA components into "
        f"{args.destination}: {', '.join(installed)}"
    )


if __name__ == "__main__":
    main()
