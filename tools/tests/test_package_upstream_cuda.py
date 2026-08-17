#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import package_upstream_cuda as subject  # noqa: E402


def write_zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageUpstreamCudaTest(unittest.TestCase):
    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "asset.zip"
            archive.write_bytes(b"asset")
            with self.assertRaisesRegex(subject.PackagingError, "SHA-256 mismatch"):
                subject.require_sha256(archive, "0" * 64)

    def test_member_lookup_rejects_ambiguous_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "backend.zip"
            write_zip(
                archive_path,
                {"ggml-cuda.dll": b"one", "nested/ggml-cuda.dll": b"two"},
            )
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(subject.PackagingError, "exactly one"):
                    subject.find_unique_member(archive, "ggml-cuda.dll")

    def test_backend_rejects_core_version_skew(self) -> None:
        backend = subject.PeInfo(
            machine=subject.PE_MACHINE_AMD64,
            optional_magic=subject.PE32_PLUS_MAGIC,
            exports=frozenset({"ggml_backend_init"}),
            imports={
                "ggml-base.dll": frozenset({"ggml_present", "ggml_missing"}),
                "cublas64_13.dll": frozenset(),
            },
        )
        core = subject.PeInfo(
            machine=subject.PE_MACHINE_AMD64,
            optional_magic=subject.PE32_PLUS_MAGIC,
            exports=frozenset({"ggml_present"}),
            imports={},
        )
        with mock.patch.object(subject, "inspect_pe", side_effect=[backend, core]):
            with self.assertRaisesRegex(subject.PackagingError, "ggml_missing"):
                subject.validate_backend(Path("backend"), Path("core"), "13")

    def test_dependency_closure_rejects_unknown_runtime(self) -> None:
        image = subject.PeInfo(
            machine=subject.PE_MACHINE_AMD64,
            optional_magic=subject.PE32_PLUS_MAGIC,
            exports=frozenset(),
            imports={"unexpected-runtime.dll": frozenset()},
        )
        with self.assertRaisesRegex(
            subject.PackagingError, "unexpected-runtime.dll"
        ):
            subject.validate_dependency_closure({"backend.dll": image})

    def test_dependency_closure_allows_nvidia_driver(self) -> None:
        image = subject.PeInfo(
            machine=subject.PE_MACHINE_AMD64,
            optional_magic=subject.PE32_PLUS_MAGIC,
            exports=frozenset(),
            imports={
                "nvcuda.dll": frozenset(),
                "api-ms-win-crt-runtime-l1-1-0.dll": frozenset(),
            },
        )
        external = subject.validate_dependency_closure({"backend.dll": image})
        self.assertEqual(
            external,
            {"nvcuda.dll", "api-ms-win-crt-runtime-l1-1-0.dll"},
        )

    def test_package_rejects_asset_from_a_different_tag(self) -> None:
        args = argparse.Namespace(
            tag="b-new",
            native_release_tag="native-test",
            llama_commit="1" * 40,
            cuda_version="13.3",
            backend_archive=Path("llama-b-old-bin-win-cuda-13.3-x64.zip"),
            backend_sha256="0" * 64,
            runtime_archive=Path("cudart-llama-bin-win-cuda-13.3-x64.zip"),
            runtime_sha256="0" * 64,
            core_dll=Path("ggml-base.dll"),
            cuobjdump=Path("cuobjdump.exe"),
            output_dir=Path("output"),
        )
        with self.assertRaisesRegex(subject.PackagingError, "version mismatch"):
            subject.package(args)

    def test_packages_only_variant_specific_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend_archive = root / "llama-b-test-bin-win-cuda-13.3-x64.zip"
            runtime_archive = (
                root / "cudart-llama-bin-win-cuda-13.3-x64.zip"
            )
            core = root / "ggml-base.dll"
            core.write_bytes(b"core")
            write_zip(
                backend_archive,
                {"ggml-cuda.dll": b"backend", "ggml-base.dll": b"wrong-core"},
            )
            write_zip(
                runtime_archive,
                {
                    "cudart64_13.dll": b"cudart",
                    "cublas64_13.dll": b"cublas",
                    "cublasLt64_13.dll": b"cublas-lt",
                    "cublas64_12.dll": b"wrong-variant",
                },
            )
            backend_info = subject.PeInfo(
                machine=subject.PE_MACHINE_AMD64,
                optional_magic=subject.PE32_PLUS_MAGIC,
                exports=frozenset({"ggml_backend_init"}),
                imports={
                    "ggml-base.dll": frozenset({"ggml_abort"}),
                    "cublas64_13.dll": frozenset(),
                },
            )
            core_info = subject.PeInfo(
                machine=subject.PE_MACHINE_AMD64,
                optional_magic=subject.PE32_PLUS_MAGIC,
                exports=frozenset({"ggml_abort"}),
                imports={},
            )
            args = argparse.Namespace(
                tag="b-test",
                native_release_tag="native-test",
                llama_commit="1" * 40,
                cuda_version="13.3",
                backend_archive=backend_archive,
                backend_sha256=digest(backend_archive),
                runtime_archive=runtime_archive,
                runtime_sha256=digest(runtime_archive),
                core_dll=core,
                cuobjdump=Path("cuobjdump.exe"),
                output_dir=root / "output",
            )
            inspections = [
                backend_info,
                core_info,
                core_info,
                core_info,
                core_info,
            ]
            device_code = {
                "inspector": {
                    "name": "NVIDIA cuobjdump",
                    "sha256": "b6f56c1eb5edd046949f9c947e730a1bf0ed5beff6fc20f8ccafd8a1f5d2eff1",
                    "version": "cuobjdump release 13.3, V13.3.29",
                },
                "ptx_architectures": ["75", "80", "90"],
                "sass_architectures": ["86", "89", "120a", "121a"],
            }
            with (
                mock.patch.object(subject, "inspect_pe", side_effect=inspections),
                mock.patch.object(
                    subject, "inspect_device_code", return_value=device_code
                ),
            ):
                output = subject.package(args)

            self.assertTrue(output.is_file())
            with subject.tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertEqual(
                names,
                {
                    "cuda-pack.json",
                    "ggml-cuda-13.dll",
                    "cudart64_13.dll",
                    "cublas64_13.dll",
                    "cublasLt64_13.dll",
                },
            )
            with subject.tarfile.open(output, "r:gz") as archive:
                metadata = subject.json.loads(
                    archive.extractfile("cuda-pack.json").read()
                )
            self.assertEqual(
                output.name,
                "llamadart-native-windows-x64-cuda13-native-test.tar.gz",
            )
            self.assertEqual(metadata["contract_version"], 3)
            self.assertEqual(metadata["native_release_tag"], "native-test")
            self.assertEqual(
                metadata["core_compatibility"],
                {"library": "ggml-base.dll", "sha256": digest(core)},
            )
            self.assertEqual(metadata["device_code"], device_code)
            self.assertEqual(
                metadata["compatibility"],
                {
                    "minimum_compute_capability": 75,
                    "minimum_driver_family": 580,
                    "minimum_driver_api": 13000,
                },
            )

    def test_archive_is_reproducible_across_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "payload.dll").write_bytes(b"payload")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            subject.write_deterministic_tar_gz(source, first)
            subject.write_deterministic_tar_gz(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
