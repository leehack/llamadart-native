import json
import unittest
from pathlib import Path

from tools import build


ROOT = Path(__file__).resolve().parents[2]


class WindowsBuildConfigTest(unittest.TestCase):
    def test_arm64_vulkan_uses_fast_cross_preset(self) -> None:
        self.assertEqual(
            build.windows_preset("arm64", "vulkan"),
            "windows-arm64-vulkan-fast",
        )

    def test_other_windows_builds_keep_full_presets(self) -> None:
        for arch, backend in (
            ("arm64", "blas"),
            ("x64", "vulkan"),
            ("x64", "cuda"),
        ):
            with self.subTest(arch=arch, backend=backend):
                self.assertEqual(
                    build.windows_preset(arch, backend),
                    f"windows-{arch}-full",
                )

    def test_fast_preset_keeps_arm64_release_contract(self) -> None:
        document = json.loads((ROOT / "CMakePresets.json").read_text())
        preset = next(
            preset
            for preset in document["configurePresets"]
            if preset["name"] == "windows-arm64-vulkan-fast"
        )

        self.assertEqual(preset["generator"], "Ninja")
        self.assertEqual(preset["architecture"]["value"], "arm64")
        self.assertEqual(
            preset["cacheVariables"]["CMAKE_TOOLCHAIN_FILE"],
            "${sourceDir}/cmake/windows-arm64-clang-toolchain.cmake",
        )
        self.assertEqual(
            preset["cacheVariables"][
                "LLAMADART_WINDOWS_ARM64_VULKAN_FAST_COMPILE"
            ],
            "ON",
        )
        self.assertEqual(preset["cacheVariables"]["GGML_VULKAN"], "ON")
        self.assertEqual(preset["cacheVariables"]["GGML_CPU_KLEIDIAI"], "ON")


if __name__ == "__main__":
    unittest.main()
