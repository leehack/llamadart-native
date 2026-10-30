from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_glibc_floor as floor  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))

from workflow_text import job_block, job_keys, step_keys, step_running  # noqa: E402

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

    def test_objdump_listings_are_read_like_readelf_listings(self) -> None:
        llvm = "0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.43) sqrtf\n"
        gnu = "0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.43  sqrtf\n"
        weak = "0000000000000000  w   DF *UND*\t0000000000000000 (GLIBC_2.43) sqrtf\n"
        for listing in (llvm, gnu, weak):
            with self.subTest(listing=listing):
                self.assertEqual({"sqrtf": (2, 43)}, floor.imports_above_floor(listing))
        below = "0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.2.5) sqrtf\n"
        libstdcxx = "0000000000000000      DF *UND*\t0000000000000000 (GLIBCXX_3.4) _Zx\n"
        defined = "00000000000006b0 g    DF .text\t0000000000000010 (GLIBC_2.99) g\n"
        self.assertEqual({}, floor.imports_above_floor(below + libstdcxx + defined))

    def test_defined_symbols_and_private_versions_are_not_imports(self) -> None:
        text = (
            "     1: 00000000000006b0    92 FUNC    GLOBAL DEFAULT   11 g@GLIBC_2.99\n"
            "     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND h@GLIBC_PRIVATE\n"
        )
        self.assertEqual({}, floor.imports_above_floor(text))


ABOVE_FLOOR = "     6: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND sqrtf@GLIBC_2.43 (5)\n"


class CommandLineTest(unittest.TestCase):
    def run_tool(self, listings: dict[str, str], *options: str, fail: str = "") -> subprocess.CompletedProcess[str]:
        """Run against libraries named by ``listings``; readelf answers per library."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = root / "libs"
            libraries.mkdir()
            for name, listing in listings.items():
                (libraries / name).write_bytes(b"")
                (root / f"{name}.txt").write_text(listing)
            fake = root / "readelf"
            fake.write_text(
                "#!/bin/sh\n"
                "for last; do :; done\n"
                'name="$(basename "$last")"\n'
                f'[ "$name" = "{fail}" ] && exit 3\n'
                f'cat "{root}/$name.txt"\n'
            )
            fake.chmod(0o755)
            return subprocess.run(
                [sys.executable, str(ROOT / "tools/check_glibc_floor.py"), "--readelf", str(fake), *options, str(libraries)],
                capture_output=True,
                text=True,
            )

    def test_exit_status_follows_the_floor(self) -> None:
        passing = self.run_tool({"libx.so": GNU_READELF})
        self.assertEqual(0, passing.returncode, passing.stderr)
        failing = self.run_tool({"libx.so": GNU_READELF + ABOVE_FLOOR})
        self.assertEqual(1, failing.returncode)
        self.assertIn("sqrtf@GLIBC_2.43 is above the GLIBC_2.38 floor", failing.stderr)

    def test_every_library_is_checked(self) -> None:
        result = self.run_tool({"liba.so": GNU_READELF, "libb.so.0": GNU_READELF, "libz.so": GNU_READELF + ABOVE_FLOOR})
        self.assertEqual(1, result.returncode)
        self.assertIn("libz.so: sqrtf@GLIBC_2.43", result.stderr)
        self.assertNotIn("liba.so", result.stderr)
        passing = self.run_tool({"liba.so": GNU_READELF, "libb.so.0": GNU_READELF, "libz.so": GNU_READELF})
        self.assertIn("PASS 3 libraries", passing.stdout)

    def test_every_failure_is_reported_before_the_exit(self) -> None:
        result = self.run_tool({"liba.so": ABOVE_FLOOR, "libz.so": ABOVE_FLOOR})
        self.assertEqual(1, result.returncode)
        self.assertIn("liba.so: sqrtf", result.stderr)
        self.assertIn("libz.so: sqrtf", result.stderr)

    def test_a_readelf_failure_fails_closed_for_any_library(self) -> None:
        for failing_library in ("liba.so", "libz.so"):
            with self.subTest(library=failing_library):
                result = self.run_tool({"liba.so": GNU_READELF, "libz.so": GNU_READELF}, fail=failing_library)
                self.assertEqual(1, result.returncode, result.stdout)
                self.assertIn("readelf failed on", result.stderr)
                self.assertIn(failing_library, result.stderr)
                self.assertNotIn("PASS", result.stdout)

    def test_a_missing_readelf_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libx.so"
            library.write_bytes(b"")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/check_glibc_floor.py"), "--readelf", f"{directory}/none", str(library)],
                capture_output=True,
                text=True,
            )
        self.assertEqual(1, result.returncode)
        self.assertIn("readelf failed", result.stderr)

    def test_the_floor_option_replaces_the_default(self) -> None:
        raised = self.run_tool({"libx.so": GNU_READELF + ABOVE_FLOOR}, "--floor", "2.43")
        self.assertEqual(0, raised.returncode, raised.stderr)
        self.assertIn("above GLIBC_2.43", raised.stdout)
        lowered = self.run_tool({"libx.so": GNU_READELF}, "--floor", "2.30")
        self.assertEqual(1, lowered.returncode)
        self.assertIn("__libc_start_main@GLIBC_2.34 is above the GLIBC_2.30 floor", lowered.stderr)
        self.assertIn("fmodf@GLIBC_2.38 is above the GLIBC_2.30 floor", lowered.stderr)

    def test_a_directory_without_libraries_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/check_glibc_floor.py"), directory], capture_output=True, text=True
            )
        self.assertEqual(1, result.returncode)


FLOOR_STEPS = {
    ("native_release.yml", "build-linux"): 'tools/check_glibc_floor.py "bin/linux/${{ matrix.arch }}"',
    ("validate_wrapper.yml", "linux-artifact-contract"): 'tools/check_glibc_floor.py "bin/linux/${{ matrix.arch }}"',
    ("validate_linux_release_rows.yml", "build-linux"): 'tools/check_glibc_floor.py "bin/linux/${{ matrix.arch }}"',
    ("validate_linux_release_rows.yml", "build-linux-hip"): "tools/check_glibc_floor.py bin/linux/x64",
}


class WorkflowWiringTest(unittest.TestCase):
    def test_each_lane_runs_the_floor_check_as_a_failing_step(self) -> None:
        for (workflow, job), command in FLOOR_STEPS.items():
            with self.subTest(workflow=workflow, job=job):
                block = job_block(workflow, job)
                step = step_running(block, "check_glibc_floor.py")
                self.assertIn(f"run: python3 {command}\n", step)
                self.assertNotIn("if", step_keys(step))
                self.assertNotIn("continue-on-error", step_keys(step))
                self.assertNotIn("continue-on-error", job_keys(block))

    def test_the_release_checks_the_floor_after_the_cuda_link_check(self) -> None:
        block = job_block("native_release.yml", "build-linux")
        self.assertLess(block.index("ldd-cuda.txt"), block.index("check_glibc_floor.py"))

    def test_the_release_floor_checks_the_merged_bundle_with_the_hip_backend(self) -> None:
        block = job_block("native_release.yml", "package-and-release")
        step = step_running(block, 'python3 tools/validate_linux_artifact.py "$archive"')
        self.assertIn("native_linux_(x64|arm64)(_.+)?$", step)
        self.assertIn("for archive in release_assets/llamadart-native-linux-*.tar.gz; do", step)
        self.assertNotIn("if", step_keys(step))
        self.assertNotIn("continue-on-error", step_keys(step))


if __name__ == "__main__":
    unittest.main()
