from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build import ANDROID_ARM64_CPU_VARIANTS, android_arm64_cpu_variant_libs  # noqa: E402


WORKFLOW = ROOT / ".github/workflows/native_release.yml"
BUILD_PY = ROOT / "tools/build.py"

LIST_COMMAND = "python3 tools/build.py list --android-arm64-cpu-variant-libs"

SINGLED_OUT_VARIANT_LIBS = frozenset(
    {
        "libggml-cpu-android_armv8.2_2.so",
        "libggml-cpu-android_armv8.0_1.so",
    }
)


class CpuVariantListingTests(unittest.TestCase):
    def _run_list(self, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/build.py"), "list", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        return result.stdout

    def test_option_prints_one_library_per_declared_variant(self) -> None:
        lines = self._run_list("--android-arm64-cpu-variant-libs").splitlines()
        expected = [f"libggml-cpu-{name}.so" for name, _arch, _features in ANDROID_ARM64_CPU_VARIANTS]
        self.assertEqual(expected, lines)
        self.assertGreater(len(lines), 1)
        self.assertEqual(len(set(lines)), len(lines))

    def test_bare_list_still_prints_presets(self) -> None:
        stdout = self._run_list()
        self.assertIn("android: abi=arm64-v8a", stdout)
        self.assertNotIn("libggml-cpu-", stdout)

    def test_android_builder_names_its_outputs_through_the_helper(self) -> None:
        body = BUILD_PY.read_text().split("def build_android_arm64_abi(")[1].split("\ndef ")[0]
        self.assertIn("android_arm64_cpu_variant_libs()", body)
        self.assertNotIn("libggml-cpu-", body)


class NativeReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()

    def test_validation_step_reads_the_variant_list_from_build_py(self) -> None:
        self.assertIn(f'variant_list="$({LIST_COMMAND})"', self.workflow)
        self.assertIn('mapfile -t expected_cpu_variants <<< "$variant_list"', self.workflow)

    def test_empty_variant_list_fails_the_step(self) -> None:
        self.assertIn('if [ -z "$variant_list" ]; then', self.workflow)
        self.assertIn('echo "tools/build.py listed no Android arm64 CPU variants" >&2', self.workflow)

    def test_every_listed_variant_library_must_exist(self) -> None:
        self.assertIn('for lib in "${expected_cpu_variants[@]}"; do', self.workflow)
        self.assertIn('test -f "$OUT_DIR/$lib" || { echo "Missing $OUT_DIR/$lib"; exit 1; }', self.workflow)

    def test_workflow_does_not_restate_the_variant_list(self) -> None:
        mentioned = set(re.findall(r"libggml-cpu-android_[\w.]+\.so", self.workflow))
        self.assertLessEqual(mentioned, SINGLED_OUT_VARIANT_LIBS)

    def test_singled_out_variant_libraries_are_really_built(self) -> None:
        self.assertLessEqual(SINGLED_OUT_VARIANT_LIBS, set(android_arm64_cpu_variant_libs()))

    def test_no_cuda_version_literal_survives(self) -> None:
        self.assertEqual([], re.findall(r"release \d+\\?\.\d+", self.workflow))
        self.assertEqual([], re.findall(r"CUDA\\v\d", self.workflow))

    def test_nvcc_checks_derive_their_pattern_from_the_env_var(self) -> None:
        self.assertIn(
            'cuda_major_minor="$(printf \'%s\' "${CUDA_REDIST_VERSION}" | cut -d. -f1,2)"',
            self.workflow,
        )
        self.assertIn(
            'grep -Eq "release ${cuda_major_minor//./\\\\.}([^0-9.]|$)" nvcc-version.txt',
            self.workflow,
        )
        self.assertIn(
            "$cudaMajorMinor = (($env:CUDA_REDIST_VERSION -split '\\.') | Select-Object -First 2) -join '.'",
            self.workflow,
        )
        self.assertIn(
            "$cudaVersionPattern = 'release ' + [regex]::Escape($cudaMajorMinor) + '([^0-9.]|$)'",
            self.workflow,
        )
        self.assertIn("-notmatch $cudaVersionPattern", self.workflow)

    def test_windows_cuda_root_derives_from_the_env_var(self) -> None:
        self.assertIn(
            'Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\\CUDA\\v$cudaMajorMinor"',
            self.workflow,
        )


if __name__ == "__main__":
    unittest.main()
