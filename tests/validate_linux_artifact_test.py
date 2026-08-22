#!/usr/bin/env python3
"""Regression tests for Linux runtime archive validation diagnostics."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_linux_artifact.py"


class ValidateLinuxArtifactTest(unittest.TestCase):
    def test_missing_explicit_tool_reports_a_clean_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_tool = Path(directory) / "missing-readelf"
            result = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(Path(directory) / "runtime.tar.gz"),
                    "--tool",
                    str(missing_tool),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"ERROR: ELF inspection tool does not exist or is not executable: "
            f"{missing_tool}",
        )
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
