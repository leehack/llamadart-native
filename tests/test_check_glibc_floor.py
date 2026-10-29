from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_glibc_floor as floor  # noqa: E402

GNU_READELF = """\
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND fmodf@GLIBC_2.38 (2)
     4: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND sqrtf@GLIBC_2.17 (4)
     5: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@GLIBC_2.34 (3)
    10: 00000000000006b0    92 FUNC    GLOBAL DEFAULT   11 g
"""
LLVM_READELF = "     4: 0000000000000000     0 FUNC    WEAK   DEFAULT   UND __cxa_finalize@GLIBC_2.17\n"


class ImportsAboveFloorTest(unittest.TestCase):
    def test_floor_is_the_ubuntu_24_04_build_maximum(self) -> None:
        self.assertEqual((2, 38), floor.GLIBC_FLOOR)

    def test_imports_at_or_below_the_floor_pass(self) -> None:
        self.assertEqual({}, floor.imports_above_floor(GNU_READELF))
        self.assertEqual({}, floor.imports_above_floor(LLVM_READELF))

    def test_imports_above_the_floor_are_reported_by_symbol(self) -> None:
        text = GNU_READELF + "    6: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND sqrtf@GLIBC_2.43 (5)\n"
        self.assertEqual({"sqrtf": (2, 43)}, floor.imports_above_floor(text))

    def test_versions_compare_numerically_not_lexically(self) -> None:
        text = "     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND f@GLIBC_2.4 (2)\n"
        self.assertEqual({}, floor.imports_above_floor(text))
        text = text.replace("2.4", "2.100")
        self.assertEqual({"f": (2, 100)}, floor.imports_above_floor(text))

    def test_defined_symbols_and_private_versions_are_not_imports(self) -> None:
        text = (
            "     1: 00000000000006b0    92 FUNC    GLOBAL DEFAULT   11 g@GLIBC_2.99\n"
            "     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND h@GLIBC_PRIVATE\n"
        )
        self.assertEqual({}, floor.imports_above_floor(text))


class CommandLineTest(unittest.TestCase):
    def run_tool(self, readelf_output: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libx.so").write_bytes(b"")
            fake = root / "readelf"
            fake.write_text(f"#!/bin/sh\ncat <<'EOF'\n{readelf_output}EOF\n")
            fake.chmod(0o755)
            return subprocess.run(
                [sys.executable, str(ROOT / "tools/check_glibc_floor.py"), "--readelf", str(fake), str(root)],
                capture_output=True,
                text=True,
            )

    def test_exit_status_follows_the_floor(self) -> None:
        passing = self.run_tool(GNU_READELF)
        self.assertEqual(0, passing.returncode, passing.stderr)
        failing = self.run_tool(GNU_READELF + "     6: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND sqrtf@GLIBC_2.43 (5)\n")
        self.assertEqual(1, failing.returncode)
        self.assertIn("sqrtf@GLIBC_2.43 is above the GLIBC_2.38 floor", failing.stderr)

    def test_a_directory_without_libraries_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/check_glibc_floor.py"), directory], capture_output=True, text=True
            )
        self.assertEqual(1, result.returncode)


class WorkflowWiringTest(unittest.TestCase):
    def test_release_and_pr_lanes_run_the_floor_check(self) -> None:
        release = (ROOT / ".github/workflows/native_release.yml").read_text().split("\n  build-linux:\n")[1]
        release = release.split("\n  build-linux-hip:\n")[0]
        pr = (ROOT / ".github/workflows/validate_wrapper.yml").read_text().split("\n  linux-artifact-contract:\n")[1]
        pr = pr.split("\n  wrapper-contract:\n")[0]
        self.assertIn('python3 tools/check_glibc_floor.py "bin/linux/${{ matrix.arch }}"', release)
        self.assertIn('python3 tools/check_glibc_floor.py "bin/linux/${{ matrix.arch }}"', pr)
        self.assertLess(release.index("ldd-cuda.txt"), release.index("check_glibc_floor.py"))


if __name__ == "__main__":
    unittest.main()
