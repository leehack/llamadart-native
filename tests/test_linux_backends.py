from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build  # noqa: E402


class LinuxBackendContractsTest(unittest.TestCase):
    def test_linux_backend_cache_vars_and_reachability(self) -> None:
        self.assertEqual(
            ("full", "cpu", "vulkan", "cuda", "hip", "blas"),
            build.LINUX_BACKENDS,
        )

        def expected_vars(
            arch: str,
            *,
            vulkan: bool = False,
            cuda: bool = False,
            hip: bool = False,
            blas: bool = False,
        ) -> dict[str, str]:
            expected = {
                "GGML_VULKAN": "ON" if vulkan else "OFF",
                "GGML_OPENCL": "OFF",
                "GGML_CUDA": "ON" if cuda else "OFF",
                "GGML_HIP": "ON" if hip else "OFF",
                "GGML_BLAS": "ON" if blas else "OFF",
                "GGML_ZENDNN": "OFF",
                "GGML_CPU_KLEIDIAI": "ON" if arch == "arm64" else "OFF",
            }
            if blas:
                expected["GGML_BLAS_VENDOR"] = "OpenBLAS"
            return expected

        expected_by_arch = {
            "x64": {
                "full": expected_vars("x64", vulkan=True, cuda=True, blas=True),
                "cpu": expected_vars("x64"),
                "vulkan": expected_vars("x64", vulkan=True),
                "cuda": expected_vars("x64", cuda=True),
                "hip": expected_vars("x64", hip=True),
                "blas": expected_vars("x64", blas=True),
            },
            "arm64": {
                "full": expected_vars("arm64", vulkan=True, blas=True),
                "cpu": expected_vars("arm64"),
                "vulkan": expected_vars("arm64", vulkan=True),
                "blas": expected_vars("arm64", blas=True),
            },
        }
        for arch, expected_backends in expected_by_arch.items():
            for backend, expected in expected_backends.items():
                with self.subTest(arch=arch, backend=backend):
                    self.assertEqual(
                        expected,
                        build.linux_backend_cache_vars(arch, backend),
                    )

        for backend in ("cuda", "hip"):
            with self.subTest(backend=backend):
                with self.assertRaises(SystemExit):
                    build.linux_backend_cache_vars("arm64", backend)

        for backend in build.LINUX_BACKENDS:
            with patch.object(
                sys,
                "argv",
                ["build.py", "linux", "--arch", "x64", "--backend", backend],
            ):
                args = build.parse_args()
            self.assertEqual(backend, args.backend)
            self.assertEqual("x64", args.arch)

        with patch.object(sys, "argv", ["build.py", "linux"]):
            args = build.parse_args()
        self.assertEqual("full", args.backend)
        self.assertIsNone(args.arch)

    def test_obsolete_zendnn_scaffolding_removed_from_code(self) -> None:
        self.assertFalse(
            hasattr(build, "patch_llama_zendnn_install_target"),
            "patch_llama_zendnn_install_target must be removed",
        )
        self.assertFalse(
            hasattr(build, "restore_llama_zendnn_install_target"),
            "restore_llama_zendnn_install_target must be removed",
        )

        build_py_source = (ROOT / "tools/build.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "ggml-zendnn",
            build_py_source.lower(),
            "tools/build.py must not mutate upstream ZenDNN sources",
        )

        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        linux_presets = {
            preset["name"]: preset["cacheVariables"]
            for preset in presets["configurePresets"]
            if preset["name"] in ("linux-x64-full", "linux-arm64-full")
        }
        self.assertEqual(
            {"linux-x64-full", "linux-arm64-full"}, set(linux_presets)
        )
        for name, cache_vars in linux_presets.items():
            with self.subTest(preset=name):
                self.assertEqual("OFF", cache_vars.get("GGML_ZENDNN"))

        manifest_source = (ROOT / "scripts/generate_assets_manifest.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("zendnn", manifest_source.lower())

    def test_release_workflow_ships_linux_backend_lanes(self) -> None:
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text(
            encoding="utf-8"
        )
        linux_job = workflow.split("\n  build-linux:\n", 1)[1].split(
            "\n  build-linux-hip:\n", 1
        )[0]
        hip_job, following_jobs = workflow.split("\n  build-linux-hip:\n", 1)[
            1
        ].split("\n  build-windows:\n", 1)
        package_job = following_jobs.split("\n  package-and-release:\n", 1)[1]
        for backend in ("vulkan", "cuda", "blas"):
            self.assertIn(f"backend: {backend}", linux_job)
        self.assertIn(
            "python3 tools/build.py linux --arch x64 --backend hip", hip_job
        )
        self.assertIn("libggml-hip.so", hip_job)
        self.assertIn("build-linux-hip", package_job.splitlines()[0])

    def test_backend_documentation_contracts(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        strategy = (ROOT / "docs/platform_backend_strategy.md").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("zendnn", readme.lower(), "README.md must not claim ZenDNN")
        self.assertNotIn(
            "zendnn",
            strategy.lower(),
            "docs/platform_backend_strategy.md must not claim ZenDNN",
        )

        self.assertIn("Linux x64: Vulkan + CUDA + BLAS + CPU", readme)
        self.assertIn("HIP/ROCm", readme)

        self.assertIn("Vulkan + CUDA + BLAS + CPU", strategy)
        self.assertIn("HIP/ROCm", strategy)
        self.assertIn("rocblas-dev", strategy)
        self.assertIn("hipblas-dev", strategy)


if __name__ == "__main__":
    unittest.main()
