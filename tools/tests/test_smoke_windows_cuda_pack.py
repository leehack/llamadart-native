#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import smoke_windows_cuda_pack as subject  # noqa: E402


class SmokeWindowsCudaPackTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "non-Windows contract")
    def test_rejects_non_windows_host(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must run on Windows"):
            subject.smoke(Path("unused"), "ggml-cuda-13.dll")


if __name__ == "__main__":
    unittest.main()
