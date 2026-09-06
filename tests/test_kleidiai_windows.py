from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class KleidiAIWindowsAssemblyTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("clang-cl"), "ClangCL is required")
    def test_suppression_removes_clang_linemarkers_without_changing_tokens(self) -> None:
        outputs = {}
        for option in ("/E", "/EP"):
            result = subprocess.run(
                [
                    "clang-cl", "--target=aarch64-pc-windows-msvc",
                    "/nologo", "/TC", option, "-",
                ],
                input="#define VALUE 42\nVALUE\n",
                text=True,
                capture_output=True,
                check=True,
            )
            outputs[option] = result.stdout
        self.assertIn('# 1 "<stdin>"', outputs["/E"])
        self.assertNotIn("#", outputs["/EP"])
        tokens = "\n".join(
            line for line in outputs["/E"].splitlines()
            if not line.startswith("#")
        ).strip()
        self.assertEqual(tokens, outputs["/EP"].strip())

    def test_preprocessing_metadata_is_scoped_to_clangcl_marmasm(self) -> None:
        for case in (
            "enabled",
            "not_msvc",
            "not_clang",
            "not_visual_studio",
            "missing_target",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as build:
                result = subprocess.run(
                    [
                        "cmake",
                        "-S", str(ROOT / "tests/fixtures/kleidiai_windows"),
                        "-B", build,
                        f"-DHELPER={ROOT / 'cmake/kleidiai_windows.cmake'}",
                        f"-DCASE={case}",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                output = result.stdout + result.stderr
                self.assertNotEqual(0, result.returncode)
                self.assertIn("METADATA_CONTRACT_PASSED", output, output)


if __name__ == "__main__":
    unittest.main()
