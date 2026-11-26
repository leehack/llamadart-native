from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build  # noqa: E402


class AndroidOpenClOptionsTest(unittest.TestCase):
    def test_android_builds_do_not_load_the_unshipped_adreno_kernel_library(self) -> None:
        args = build.android_base_cmake_args("arm64-v8a", Path("/ndk"))
        self.assertIn("-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF", args)


if __name__ == "__main__":
    unittest.main()
