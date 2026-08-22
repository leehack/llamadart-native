#!/usr/bin/env python3
"""Regression tests for Linux runtime archive symlink handling."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "tools/package_linux_artifact.py"


class PackageLinuxArtifactTest(unittest.TestCase):
    def run_packager(
        self, input_dir: Path, output: Path, pattern: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PACKAGER),
                "--input-dir",
                str(input_dir),
                "--output",
                str(output),
                "--pattern",
                pattern,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_selected_symlink_includes_its_complete_target_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "libexample.so.1.2.3").write_bytes(b"ELF fixture")
            (input_dir / "libexample.so.1").symlink_to("libexample.so.1.2.3")
            (input_dir / "libexample.so").symlink_to("libexample.so.1")
            output = root / "runtime.tar.gz"

            result = self.run_packager(input_dir, output, "libexample.so")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with tarfile.open(output, "r:gz") as archive:
                members = {member.name: member for member in archive.getmembers()}
            self.assertEqual(
                set(members),
                {"libexample.so", "libexample.so.1", "libexample.so.1.2.3"},
            )
            self.assertTrue(members["libexample.so"].issym())
            self.assertEqual(members["libexample.so"].linkname, "libexample.so.1")
            self.assertTrue(members["libexample.so.1"].issym())
            self.assertEqual(
                members["libexample.so.1"].linkname, "libexample.so.1.2.3"
            )
            self.assertTrue(members["libexample.so.1.2.3"].isfile())

    def test_selected_broken_symlink_fails_with_target_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "libbroken.so").symlink_to("libbroken.so.1")
            output = root / "runtime.tar.gz"

            result = self.run_packager(input_dir, output, "libbroken.so")

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "Linux runtime symlink target does not exist: "
                "libbroken.so -> libbroken.so.1",
                result.stdout,
            )
            self.assertFalse(output.exists())

    def test_selected_directory_target_fails_with_type_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "runtime-directory").mkdir()
            (input_dir / "libinvalid.so").symlink_to(
                "runtime-directory", target_is_directory=True
            )
            output = root / "runtime.tar.gz"

            result = self.run_packager(input_dir, output, "libinvalid.so")

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "Linux runtime symlink target is not a regular file: "
                "libinvalid.so -> runtime-directory",
                result.stdout,
            )
            self.assertFalse(output.exists())

    def test_selected_parent_directory_symlink_fails_as_non_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "libescape.so").symlink_to("..", target_is_directory=True)
            output = root / "runtime.tar.gz"

            result = self.run_packager(input_dir, output, "libescape.so")

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "Linux runtime symlink libescape.so must target a sibling file, "
                "not ..",
                result.stdout,
            )
            self.assertFalse(output.exists())

    def test_selected_current_directory_symlink_fails_as_non_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "libescape.so").symlink_to(".", target_is_directory=True)
            output = root / "runtime.tar.gz"

            result = self.run_packager(input_dir, output, "libescape.so")

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "Linux runtime symlink libescape.so must target a sibling file, "
                "not .",
                result.stdout,
            )
            self.assertFalse(output.exists())

    def test_patterns_with_path_components_fail_before_archive_creation(self) -> None:
        unsafe_patterns = (
            "../*.so",
            "./*.so",
            "subdir/*.so",
            r"..\*.so",
            r".\*.so",
            r"subdir\*.so",
            "/tmp/*.so",
            r"C:\tmp\*.so",
            "",
            ".",
            "..",
        )
        for pattern in unsafe_patterns:
            with self.subTest(pattern=pattern):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    input_dir = root / "input"
                    input_dir.mkdir()
                    (root / "outside.so").write_bytes(b"outside")
                    output = root / "runtime.tar.gz"

                    result = self.run_packager(input_dir, output, pattern)

                    self.assertEqual(
                        result.returncode,
                        1,
                        result.stdout + result.stderr,
                    )
                    self.assertIn(
                        "Linux runtime pattern must be a filename glob without "
                        "path components",
                        result.stdout,
                    )
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
