#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import cuda_pack_contract  # noqa: E402
import verify_cuda_pack as subject  # noqa: E402


def add_bytes(archive: tarfile.TarFile, name: str, contents: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    archive.addfile(info, io.BytesIO(contents))


def write_test_pack(
    path: Path,
    version: str,
    *,
    corrupt_hash: bool = False,
    extra_name: str | None = None,
) -> None:
    variant = cuda_pack_contract.CUDA_VARIANTS[version]
    major = str(variant.cuda_major)
    payload = {
        f"ggml-cuda-{major}.dll": b"backend",
        f"cudart64_{major}.dll": b"cudart",
        f"cublas64_{major}.dll": b"cublas",
        f"cublasLt64_{major}.dll": b"cublas-lt",
    }
    files = []
    for name, contents in sorted(payload.items()):
        digest = hashlib.sha256(contents).hexdigest()
        if corrupt_hash and name.startswith("ggml-cuda"):
            digest = "0" * 64
        files.append({"name": name, "sha256": digest, "size": len(contents)})
    manifest = {
        "contract_version": 2,
        "llama_cpp_tag": "b-test",
        "llama_cpp_commit": "1" * 40,
        "platform": "windows",
        "arch": "x64",
        "backend": "cuda",
        "cuda_version": version,
        "cuda_major": variant.cuda_major,
        "backend_library": f"ggml-cuda-{major}.dll",
        "compatibility": {
            "minimum_compute_capability": variant.minimum_compute_capability,
            "minimum_driver_family": variant.minimum_driver_family,
        },
        "device_code": {
            "inspector": {
                "name": "NVIDIA cuobjdump",
                "sha256": cuda_pack_contract.CUOBJDUMP_SHA256,
                "version": "cuobjdump release 13.3, V13.3.29",
            },
            "ptx_architectures": sorted(variant.ptx_architectures),
            "sass_architectures": sorted(variant.sass_architectures),
        },
        "source_assets": [
            {
                "name": f"llama-b-test-bin-win-cuda-{version}-x64.zip",
                "sha256": "3" * 64,
                "url": "https://github.com/ggml-org/llama.cpp/releases/"
                f"download/b-test/llama-b-test-bin-win-cuda-{version}-x64.zip",
            },
            {
                "name": f"cudart-llama-bin-win-cuda-{version}-x64.zip",
                "sha256": "4" * 64,
                "url": "https://github.com/ggml-org/llama.cpp/releases/"
                f"download/b-test/cudart-llama-bin-win-cuda-{version}-x64.zip",
            },
        ],
        "files": files,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in payload.items():
            add_bytes(archive, name, contents)
        add_bytes(
            archive,
            "cuda-pack.json",
            (json.dumps(manifest) + "\n").encode(),
        )
        if extra_name is not None:
            add_bytes(archive, extra_name, b"extra")


class VerifyCudaPackTest(unittest.TestCase):
    def test_verifies_payload_hashes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "cuda13.tar.gz"
            write_test_pack(pack, "13.3")
            manifest = subject.verify_pack(
                pack, expected_tag="b-test", expected_commit="1" * 40
            )
            self.assertEqual(manifest["cuda_major"], 13)

    def test_rejects_corrupt_payload_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "cuda12.tar.gz"
            write_test_pack(pack, "12.4", corrupt_hash=True)
            with self.assertRaisesRegex(subject.VerificationError, "SHA-256"):
                subject.verify_pack(
                    pack, expected_tag="b-test", expected_commit="1" * 40
                )

    def test_rejects_non_top_level_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "cuda12.tar.gz"
            write_test_pack(pack, "12.4", extra_name="../escape.dll")
            with self.assertRaisesRegex(subject.VerificationError, "top-level"):
                subject.verify_pack(
                    pack, expected_tag="b-test", expected_commit="1" * 40
                )


if __name__ == "__main__":
    unittest.main()
