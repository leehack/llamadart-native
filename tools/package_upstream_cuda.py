#!/usr/bin/env python3
"""Validate and repackage an exact-tag upstream Windows CUDA backend.

The upstream llama.cpp release splits ``ggml-cuda.dll`` from the CUDA runtime
DLLs. This tool verifies both source archives, checks the PE contracts against
the exact locally built ``ggml-base.dll``, and emits a collision-free optional
CUDA pack for llamadart-native consumers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import mmap
from pathlib import Path
import re
import shutil
import struct
import tarfile
import tempfile
import zipfile

from cuda_pack_contract import (
    CUDA_VARIANTS,
    CudaContractError,
    inspect_device_code,
)


PE_MACHINE_AMD64 = 0x8664
PE32_PLUS_MAGIC = 0x20B
WINDOWS_EXTERNAL_IMPORTS = frozenset(
    {
        "kernel32.dll",
        "msvcp140.dll",
        "nvcuda.dll",
        "vcomp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    }
)


class PackagingError(RuntimeError):
    """Raised when an upstream asset violates the CUDA pack contract."""


@dataclass(frozen=True)
class PeInfo:
    machine: int
    optional_magic: int
    exports: frozenset[str]
    imports: dict[str, frozenset[str]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    actual = sha256(path)
    normalized = expected.removeprefix("sha256:").lower()
    if actual != normalized:
        raise PackagingError(
            f"SHA-256 mismatch for {path.name}: expected {normalized}, got {actual}"
        )
    return actual


class PeReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._source = path.open("rb")
        try:
            self.data = mmap.mmap(self._source.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            self._source.close()
            raise PackagingError(f"Unable to map PE file: {path.name}")
        self.sections: list[tuple[int, int, int, int]] = []

    def close(self) -> None:
        self.data.close()
        self._source.close()

    def _unpack(self, fmt: str, offset: int) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(self.data):
            raise PackagingError(f"Malformed PE structure in {self.path.name}")
        return struct.unpack_from(fmt, self.data, offset)

    def _cstring(self, offset: int) -> str:
        if offset < 0 or offset >= len(self.data):
            raise PackagingError(f"Malformed PE string in {self.path.name}")
        end = self.data.find(b"\0", offset)
        if end < 0:
            raise PackagingError(f"Unterminated PE string in {self.path.name}")
        try:
            return self.data[offset:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise PackagingError(f"Non-ASCII PE string in {self.path.name}") from error

    def _rva_offset(self, rva: int) -> int:
        if rva == 0:
            return 0
        for virtual_address, virtual_size, raw_offset, raw_size in self.sections:
            span = max(virtual_size, raw_size)
            if virtual_address <= rva < virtual_address + span:
                offset = raw_offset + rva - virtual_address
                if offset >= len(self.data):
                    break
                return offset
        raise PackagingError(
            f"PE RVA 0x{rva:x} is outside file-backed sections in {self.path.name}"
        )

    def read(self) -> PeInfo:
        if len(self.data) < 0x40 or self.data[:2] != b"MZ":
            raise PackagingError(f"Not a PE file: {self.path.name}")
        (pe_offset,) = self._unpack("<I", 0x3C)
        if self.data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise PackagingError(f"Invalid PE signature: {self.path.name}")

        coff_offset = pe_offset + 4
        machine, section_count, _, _, _, optional_size, _ = self._unpack(
            "<HHIIIHH", coff_offset
        )
        optional_offset = coff_offset + 20
        (optional_magic,) = self._unpack("<H", optional_offset)
        if optional_magic != PE32_PLUS_MAGIC:
            raise PackagingError(
                f"Expected PE32+ image in {self.path.name}, got 0x{optional_magic:x}"
            )
        if optional_size < 128:
            raise PackagingError(f"Truncated PE optional header in {self.path.name}")

        section_offset = optional_offset + optional_size
        for index in range(section_count):
            offset = section_offset + index * 40
            _, virtual_size, virtual_address, raw_size, raw_offset = self._unpack(
                "<8sIIII", offset
            )
            self.sections.append(
                (virtual_address, virtual_size, raw_offset, raw_size)
            )

        export_rva, _ = self._unpack("<II", optional_offset + 112)
        import_rva, _ = self._unpack("<II", optional_offset + 120)
        return PeInfo(
            machine=machine,
            optional_magic=optional_magic,
            exports=frozenset(self._read_exports(export_rva)),
            imports=self._read_imports(import_rva),
        )

    def _read_exports(self, directory_rva: int) -> set[str]:
        if directory_rva == 0:
            return set()
        offset = self._rva_offset(directory_rva)
        fields = self._unpack("<IIHHIIIIIII", offset)
        number_of_names = fields[7]
        names_rva = fields[9]
        names_offset = self._rva_offset(names_rva)
        exports: set[str] = set()
        for index in range(number_of_names):
            (name_rva,) = self._unpack("<I", names_offset + index * 4)
            exports.add(self._cstring(self._rva_offset(name_rva)))
        return exports

    def _read_imports(self, directory_rva: int) -> dict[str, frozenset[str]]:
        if directory_rva == 0:
            return {}
        offset = self._rva_offset(directory_rva)
        imports: dict[str, frozenset[str]] = {}
        while True:
            original_thunk, timestamp, forwarder, name_rva, first_thunk = self._unpack(
                "<IIIII", offset
            )
            if not any((original_thunk, timestamp, forwarder, name_rva, first_thunk)):
                break
            dll_name = self._cstring(self._rva_offset(name_rva)).lower()
            thunk_rva = original_thunk or first_thunk
            thunk_offset = self._rva_offset(thunk_rva)
            symbols: set[str] = set()
            index = 0
            while True:
                (entry,) = self._unpack("<Q", thunk_offset + index * 8)
                if entry == 0:
                    break
                if not entry & (1 << 63):
                    symbol_offset = self._rva_offset(entry) + 2
                    symbols.add(self._cstring(symbol_offset))
                index += 1
            imports[dll_name] = frozenset(symbols)
            offset += 20
        return imports


def inspect_pe(path: Path) -> PeInfo:
    reader = PeReader(path)
    try:
        info = reader.read()
    finally:
        reader.close()
    if info.machine != PE_MACHINE_AMD64:
        raise PackagingError(
            f"Expected x86-64 PE image in {path.name}, got machine 0x{info.machine:x}"
        )
    return info


def find_unique_member(archive: zipfile.ZipFile, expected_name: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and Path(name).name.lower() == expected_name.lower()
    ]
    if len(matches) != 1:
        raise PackagingError(
            f"Expected exactly one {expected_name} in {Path(archive.filename).name}, "
            f"found {len(matches)}"
        )
    return matches[0]


def extract_member(archive_path: Path, member_name: str, output: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        member = find_unique_member(archive, member_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, output.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def validate_backend(
    backend_path: Path,
    core_path: Path,
    cuda_major: str,
) -> tuple[PeInfo, PeInfo]:
    backend = inspect_pe(backend_path)
    core = inspect_pe(core_path)
    if "ggml_backend_init" not in backend.exports:
        raise PackagingError("Upstream CUDA backend does not export ggml_backend_init")

    core_imports = backend.imports.get("ggml-base.dll")
    if not core_imports:
        raise PackagingError("Upstream CUDA backend does not import ggml-base.dll")
    missing = sorted(core_imports - core.exports)
    if missing:
        raise PackagingError(
            "CUDA backend imports symbols absent from the exact core: "
            + ", ".join(missing)
        )

    expected_cublas = f"cublas64_{cuda_major}.dll"
    if expected_cublas not in backend.imports:
        raise PackagingError(
            f"CUDA backend does not import the expected runtime {expected_cublas}"
        )
    return backend, core


def validate_dependency_closure(images: dict[str, PeInfo]) -> set[str]:
    available = {name.lower() for name in images}
    external: set[str] = set()
    missing: set[str] = set()
    for image in images.values():
        for imported_name in image.imports:
            name = imported_name.lower()
            if name in available:
                continue
            if name.startswith("api-ms-win-") or name in WINDOWS_EXTERNAL_IMPORTS:
                external.add(name)
                continue
            missing.add(name)
    if missing:
        raise PackagingError(
            "CUDA pack has unresolved non-system DLL imports: "
            + ", ".join(sorted(missing))
        )
    return external


def write_deterministic_tar_gz(source_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(source_dir.iterdir()):
                    info = archive.gettarinfo(str(path), arcname=path.name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        archive.addfile(info, source)


def package(args: argparse.Namespace) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag) is None:
        raise PackagingError(f"Invalid llama.cpp tag: {args.tag}")
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", args.native_release_tag
    ) is None:
        raise PackagingError(
            f"Invalid llamadart-native release tag: {args.native_release_tag}"
        )
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.llama_commit) is None:
        raise PackagingError("llama.cpp commit must be a full 40-character SHA")
    variant = CUDA_VARIANTS.get(args.cuda_version)
    if variant is None:
        raise PackagingError(
            "Only the audited CUDA 12.4 and CUDA 13.3 packs are supported"
        )
    cuda_major = str(variant.cuda_major)

    expected_backend_name = (
        f"llama-{args.tag}-bin-win-cuda-{args.cuda_version}-x64.zip"
    )
    expected_runtime_name = (
        f"cudart-llama-bin-win-cuda-{args.cuda_version}-x64.zip"
    )
    for actual, expected in (
        (args.backend_archive.name, expected_backend_name),
        (args.runtime_archive.name, expected_runtime_name),
    ):
        if actual != expected:
            raise PackagingError(
                f"Upstream asset version mismatch: expected {expected}, got {actual}"
            )

    backend_archive_sha = require_sha256(
        args.backend_archive, args.backend_sha256
    )
    runtime_archive_sha = require_sha256(
        args.runtime_archive, args.runtime_sha256
    )

    with tempfile.TemporaryDirectory(prefix="llamadart-cuda-pack-") as directory:
        staging = Path(directory)
        backend_name = f"ggml-cuda-{cuda_major}.dll"
        backend_path = staging / backend_name
        extract_member(args.backend_archive, "ggml-cuda.dll", backend_path)
        device_code = inspect_device_code(args.cuobjdump, backend_path, variant)

        runtime_names = [
            f"cudart64_{cuda_major}.dll",
            f"cublas64_{cuda_major}.dll",
            f"cublasLt64_{cuda_major}.dll",
        ]
        for runtime_name in runtime_names:
            extract_member(args.runtime_archive, runtime_name, staging / runtime_name)

        backend, core = validate_backend(backend_path, args.core_dll, cuda_major)
        images = {backend_name: backend, "ggml-base.dll": core}
        for runtime_name in runtime_names:
            images[runtime_name] = inspect_pe(staging / runtime_name)
        external_imports = validate_dependency_closure(images)

        files = []
        for path in sorted(staging.iterdir()):
            files.append(
                {"name": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            )

        metadata = {
            "contract_version": 3,
            "native_release_tag": args.native_release_tag,
            "llama_cpp_tag": args.tag,
            "llama_cpp_commit": args.llama_commit,
            "platform": "windows",
            "arch": "x64",
            "backend": "cuda",
            "cuda_version": args.cuda_version,
            "cuda_major": int(cuda_major),
            "backend_library": backend_name,
            "core_compatibility": {
                "library": "ggml-base.dll",
                "sha256": sha256(args.core_dll),
            },
            "compatibility": {
                "minimum_compute_capability": variant.minimum_compute_capability,
                "minimum_driver_family": variant.minimum_driver_family,
                "minimum_driver_api": variant.minimum_driver_api,
            },
            "device_code": device_code,
            "source_assets": [
                {
                    "name": args.backend_archive.name,
                    "sha256": backend_archive_sha,
                    "url": "https://github.com/ggml-org/llama.cpp/releases/"
                    f"download/{args.tag}/{args.backend_archive.name}",
                },
                {
                    "name": args.runtime_archive.name,
                    "sha256": runtime_archive_sha,
                    "url": "https://github.com/ggml-org/llama.cpp/releases/"
                    f"download/{args.tag}/{args.runtime_archive.name}",
                },
            ],
            "backend_exports": sorted(backend.exports),
            "backend_imports": {
                name: sorted(symbols) for name, symbols in sorted(backend.imports.items())
            },
            "external_imports": sorted(external_imports),
            "files": files,
        }
        metadata_path = staging / "cuda-pack.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        output = args.output_dir / (
            "llamadart-native-windows-x64-"
            f"cuda{cuda_major}-{args.native_release_tag}.tar.gz"
        )
        write_deterministic_tar_gz(staging, output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--native-release-tag", required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--backend-archive", required=True, type=Path)
    parser.add_argument("--backend-sha256", required=True)
    parser.add_argument("--runtime-archive", required=True, type=Path)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--core-dll", required=True, type=Path)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        output = package(parse_args())
    except (CudaContractError, PackagingError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}")
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
