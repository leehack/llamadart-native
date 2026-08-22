#!/usr/bin/env python3
"""Regression tests for Linux runtime archive validation diagnostics."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from tools.validate_linux_artifact import resolve_tool, validate_archive


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_linux_artifact.py"


class ValidateLinuxArtifactTest(unittest.TestCase):
    def test_version_suffixed_llvm_objdump_uses_objdump_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "llvm-objdump-17"
            tool.touch(mode=0o755)

            resolved, mode = resolve_tool(str(tool))

        self.assertEqual(resolved, str(tool))
        self.assertEqual(mode, "objdump")

    def test_auto_discovery_accepts_llvm_objdump(self) -> None:
        with mock.patch(
            "tools.validate_linux_artifact.shutil.which",
            side_effect=lambda candidate: (
                "/opt/llvm/bin/llvm-objdump"
                if candidate == "llvm-objdump"
                else None
            ),
        ):
            resolved, mode = resolve_tool(None)

        self.assertEqual(resolved, "/opt/llvm/bin/llvm-objdump")
        self.assertEqual(mode, "objdump")

    def test_multi_hop_symlink_cycles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "runtime.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, target in (
                    ("libllamadart.so", "libllama.so"),
                    ("libllama.so", "libllamadart.so"),
                    ("libmtmd.so", "libmtmd.so.0"),
                    ("libmtmd.so.0", "libmtmd.so"),
                ):
                    member = tarfile.TarInfo(name)
                    member.type = tarfile.SYMTYPE
                    member.linkname = target
                    archive.addfile(member)

            errors = validate_archive(archive_path, "/usr/bin/true", "readelf")

        self.assertIn(
            "libllamadart.so: symlink chain contains a cycle: "
            "libllamadart.so -> libllama.so -> libllamadart.so",
            errors,
        )
        self.assertIn(
            "libmtmd.so: symlink chain contains a cycle: "
            "libmtmd.so -> libmtmd.so.0 -> libmtmd.so",
            errors,
        )

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
