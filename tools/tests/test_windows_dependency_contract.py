#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import build as subject  # noqa: E402


class WindowsDependencyContractTest(unittest.TestCase):
    def test_psapi_is_treated_as_a_windows_system_dependency(self) -> None:
        self.assertTrue(subject.is_windows_system_dependency("PSAPI.DLL"))

    def test_api_set_is_treated_as_a_windows_system_dependency(self) -> None:
        self.assertTrue(
            subject.is_windows_system_dependency(
                "api-ms-win-core-synch-l1-2-0.dll"
            )
        )

    def test_arm64_openmp_runtime_remains_bundleable(self) -> None:
        self.assertFalse(
            subject.is_windows_system_dependency("libomp140.aarch64.dll")
        )


if __name__ == "__main__":
    unittest.main()
