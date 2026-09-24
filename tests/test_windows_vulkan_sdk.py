from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WindowsVulkanSdkTest(unittest.TestCase):
    def test_arm64_uses_native_arm64_sdk_and_checks_import_library(self) -> None:
        action = (ROOT / ".github/actions/install-vulkan-sdk-windows/action.yml").read_text(encoding="utf-8")
        self.assertIn("/warm/vulkansdk-windows-ARM64-$ver.exe", action)
        self.assertIn("/windows/vulkansdk-windows-X64-$ver.exe", action)
        self.assertIn("29c3e7b9ff9c9d38455708bed7e744f879c770c15f3fedae55c409817c375515", action)
        self.assertIn("34a921d951858274ca8e470e9d0a3b7624db41216b3908ccea9f73c8a1b7500e", action)
        self.assertIn("Get-FileHash $installer -Algorithm SHA256", action)
        self.assertIn('dumpbin /headers $lib', action)

    def test_release_and_validation_lanes_use_the_action(self) -> None:
        release = (ROOT / ".github/workflows/native_release.yml").read_text(encoding="utf-8")
        windows_job = release.split("\n  build-windows:\n", 1)[1].split("\n  smoke-release-contract:\n", 1)[0]
        self.assertIn("uses: ./.github/actions/install-vulkan-sdk-windows\n        with:\n          arch: ${{ matrix.arch }}", windows_job)
        self.assertNotIn("sdk.lunarg.com", release)
        validate = (ROOT / ".github/workflows/validate_wrapper.yml").read_text(encoding="utf-8")
        lane = validate.split("\n  windows-arm64-vulkan:\n", 1)[1].split("\n  linux-artifact-contract:\n", 1)[0]
        self.assertIn("uses: ./.github/actions/install-vulkan-sdk-windows\n        with:\n          arch: arm64", lane)
        self.assertIn("python tools/build.py windows --arch arm64 --backend vulkan", lane)


if __name__ == "__main__":
    unittest.main()
