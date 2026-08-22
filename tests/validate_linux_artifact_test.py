#!/usr/bin/env python3
"""Regression tests for Linux runtime archive validation diagnostics."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from tools.validate_linux_artifact import (
    extract_archive,
    extract_member_safely,
    resolve_tool,
    validate_archive,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_linux_artifact.py"


class ValidateLinuxArtifactTest(unittest.TestCase):
    def test_supported_python_uses_the_data_extraction_filter(self) -> None:
        archive = mock.Mock(spec=tarfile.TarFile)
        member = tarfile.TarInfo("libexample.so")
        destination = Path("extract")

        with mock.patch(
            "tools.validate_linux_artifact.TARFILE_EXTRACT_SUPPORTS_FILTER",
            True,
        ):
            extract_member_safely(archive, member, destination)

        archive.extract.assert_called_once_with(
            member, destination, filter="data"
        )

    def test_supported_python_recreates_validated_symlinks_manually(self) -> None:
        archive = mock.Mock(spec=tarfile.TarFile)
        member = tarfile.TarInfo("libexample.so")
        member.type = tarfile.SYMTYPE
        member.linkname = "libexample.so.1"

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with mock.patch(
                "tools.validate_linux_artifact.TARFILE_EXTRACT_SUPPORTS_FILTER",
                True,
            ):
                extract_member_safely(archive, member, destination)

            extracted = destination / member.name
            self.assertTrue(extracted.is_symlink())
            self.assertEqual(extracted.readlink(), Path(member.linkname))
        archive.extract.assert_not_called()

    def test_symlink_creation_oserror_is_a_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "runtime.tar.gz"
            member = tarfile.TarInfo("libexample.so")
            member.type = tarfile.SYMTYPE
            member.linkname = "libexample.so.1"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(member)

            with mock.patch.object(
                Path,
                "symlink_to",
                side_effect=PermissionError("symlink creation denied"),
            ):
                errors = validate_archive(archive_path, "/usr/bin/true", "readelf")

        self.assertEqual(
            errors,
            ["Archive extraction failed: symlink creation denied"],
        )

    def test_payload_extraction_oserror_is_a_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "runtime.tar.gz"
            member = tarfile.TarInfo("libexample.so")
            member.size = 1
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(member, BytesIO(b"x"))

            with mock.patch.object(
                tarfile.TarFile,
                "extract",
                side_effect=OSError("payload creation denied\n" + "x" * 600),
            ):
                errors = validate_archive(archive_path, "/usr/bin/true", "readelf")

        self.assertEqual(len(errors), 1)
        self.assertTrue(
            errors[0].startswith("Archive extraction failed: payload creation denied ")
        )
        self.assertTrue(errors[0].endswith("..."))
        self.assertNotIn("\n", errors[0])
        self.assertLessEqual(len(errors[0]), len("Archive extraction failed: ") + 512)

    def test_legacy_fallback_discards_archive_metadata_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "runtime.tar.gz"
            destination = root / "extract"
            destination.mkdir()
            payload = b"ELF fixture"
            member = tarfile.TarInfo("libexample.so")
            member.size = len(payload)
            member.mode = 0o7777
            member.uid = 0
            member.gid = 0
            member.uname = "root"
            member.gname = "root"
            member.mtime = 1
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(member, BytesIO(payload))

            with mock.patch(
                "tools.validate_linux_artifact.TARFILE_EXTRACT_SUPPORTS_FILTER",
                False,
            ):
                extract_archive(archive_path, destination)

            extracted = destination / member.name
            self.assertEqual(extracted.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(extracted.stat().st_mode), 0o600)
            self.assertNotEqual(extracted.stat().st_mtime, member.mtime)

    def test_member_path_traversal_is_rejected_before_extraction(self) -> None:
        for member_name in ("../escape", r"..\escape", r"C:\escape"):
            with self.subTest(member_name=member_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    archive_path = root / "runtime.tar.gz"
                    destination = root / "extract"
                    destination.mkdir()
                    member = tarfile.TarInfo(member_name)
                    member.size = 1
                    with tarfile.open(archive_path, "w:gz") as archive:
                        archive.addfile(member, BytesIO(b"x"))

                    with self.assertRaisesRegex(
                        ValueError, "Archive member must be a flat runtime filename"
                    ):
                        extract_archive(archive_path, destination)

                    self.assertEqual(list(destination.iterdir()), [])

    def test_symlink_target_traversal_is_rejected_before_extraction(self) -> None:
        for linkname in ("../escape", r"..\escape", r"C:\escape"):
            with self.subTest(linkname=linkname):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    archive_path = root / "runtime.tar.gz"
                    destination = root / "extract"
                    destination.mkdir()
                    member = tarfile.TarInfo("libexample.so")
                    member.type = tarfile.SYMTYPE
                    member.linkname = linkname
                    with tarfile.open(archive_path, "w:gz") as archive:
                        archive.addfile(member)

                    with self.assertRaisesRegex(
                        ValueError, "Archive member must be a flat runtime filename"
                    ):
                        extract_archive(archive_path, destination)

                    self.assertEqual(list(destination.iterdir()), [])

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

    def test_missing_auto_discovery_lists_every_candidate(self) -> None:
        with mock.patch(
            "tools.validate_linux_artifact.shutil.which", return_value=None
        ):
            with self.assertRaisesRegex(
                ValueError,
                "readelf, llvm-readelf, objdump, and llvm-objdump",
            ):
                resolve_tool(None)

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
