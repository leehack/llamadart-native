#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import cuda_pack_contract as subject  # noqa: E402


def manifest(version: str) -> dict[str, object]:
    variant = subject.CUDA_VARIANTS[version]
    return {
        "cuda_version": version,
        "cuda_major": variant.cuda_major,
        "compatibility": {
            "minimum_compute_capability": variant.minimum_compute_capability,
            "minimum_driver_family": variant.minimum_driver_family,
        },
        "device_code": {
            "ptx_architectures": sorted(variant.ptx_architectures),
            "sass_architectures": sorted(variant.sass_architectures),
        },
    }


class CudaPackContractTest(unittest.TestCase):
    def test_parses_architectures_with_family_suffixes(self) -> None:
        output = "\n".join(
            (
                "ELF file 1: kernels.sm_86.cubin",
                "ELF file 2: kernels.sm_120a.cubin",
                "ELF file 3: duplicate.sm_86.cubin",
            )
        )
        self.assertEqual(
            subject.parse_listed_architectures(output),
            frozenset({"86", "120a"}),
        )

    def test_inspects_exact_cuda_13_device_code(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, "cuobjdump release 13.3, V13.3.29", ""),
            subprocess.CompletedProcess(
                [],
                0,
                "\n".join(
                    f"ELF file: ggml.sm_{arch}.cubin"
                    for arch in ("86", "89", "120a", "121a")
                ),
                "",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                "\n".join(
                    f"PTX file: ggml.sm_{arch}.ptx"
                    for arch in ("75", "80", "90")
                ),
                "",
            ),
        ]
        with (
            mock.patch.object(subject.subprocess, "run", side_effect=results),
            mock.patch.object(
                subject, "file_sha256", return_value=subject.CUOBJDUMP_SHA256
            ),
        ):
            result = subject.inspect_device_code(
                Path("cuobjdump.exe"),
                Path("ggml-cuda-13.dll"),
                subject.CUDA_VARIANTS["13.3"],
            )
        self.assertEqual(result["ptx_architectures"], ["75", "80", "90"])
        self.assertEqual(
            result["sass_architectures"], ["120a", "121a", "86", "89"]
        )
        self.assertEqual(
            result["inspector"]["sha256"], subject.CUOBJDUMP_SHA256
        )

    def test_missing_fatbin_target_fails_closed(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, "cuobjdump release 13.3, V13.3.29", ""),
            subprocess.CompletedProcess([], 0, "ELF file: ggml.sm_86.cubin", ""),
            subprocess.CompletedProcess(
                [], 0, "PTX file: ggml.sm_75.ptx", ""
            ),
        ]
        with (
            mock.patch.object(subject.subprocess, "run", side_effect=results),
            mock.patch.object(
                subject, "file_sha256", return_value=subject.CUOBJDUMP_SHA256
            ),
        ):
            with self.assertRaisesRegex(
                subject.CudaContractError, "SASS architectures differ"
            ):
                subject.inspect_device_code(
                    Path("cuobjdump.exe"),
                    Path("ggml-cuda-13.dll"),
                    subject.CUDA_VARIANTS["13.3"],
                )

    def test_selector_covers_driver_and_architecture_boundaries(self) -> None:
        manifests = [manifest("12.4"), manifest("13.3")]
        cases = (
            (49, 610, None),
            (50, 524, None),
            (50, 525, 12),
            (70, 610, 12),
            (75, 579, 12),
            (75, 580, 13),
            (120, 610, 13),
        )
        for capability, driver, expected in cases:
            with self.subTest(capability=capability, driver=driver):
                selected = subject.select_cuda_pack(
                    manifests,
                    compute_capability=capability,
                    driver_family=driver,
                )
                actual = None if selected is None else selected["cuda_major"]
                self.assertEqual(actual, expected)

    def test_selector_rejects_tampered_architecture_metadata(self) -> None:
        tampered = manifest("13.3")
        tampered["device_code"]["ptx_architectures"] = ["75"]
        with self.assertRaisesRegex(subject.CudaContractError, "PTX metadata"):
            subject.select_cuda_pack(
                [tampered], compute_capability=75, driver_family=580
            )


if __name__ == "__main__":
    unittest.main()
