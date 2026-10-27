from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build  # noqa: E402
import validate_linux_arm64_cpu_isa as isa  # noqa: E402


def symbol(name: str, start: int, size: int) -> str:
    return f" 1: {start:016x} {size} FUNC GLOBAL DEFAULT 14 {name}"


def disassembly(*lines: str) -> str:
    return "test.so: file format elf64-littleaarch64\nDisassembly of section .text:\n" + "\n".join(lines)


SCORE = symbol("ggml_backend_score", 0x400, 4)
KAI = symbol("kai_kernel_matmul_dotprod", 0x100, 8)
GGML = symbol("ggml_vec_dot_f16", 0x200, 16)
FMLA_8H = "200: 4e420c20 fmla v0.8h, v1.8h, v2.8h"
SDOT = "204: 4e829420 sdot v0.4s, v1.16b, v2.16b"
RET = "20c: d65f03c0 ret"


class VariantListTest(unittest.TestCase):
    def test_list_prints_the_variant_libraries_beside_the_baseline(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/build.py"), "list", "--linux-arm64-cpu-variant-libs"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            ["libggml-cpu-linux_armv8.2_2.so", "libggml-cpu-linux_armv8.6_1.so"],
            result.stdout.splitlines(),
        )

    def test_variants_declare_scored_features_matching_their_march(self) -> None:
        for name, march, features in build.LINUX_ARM64_CPU_VARIANTS:
            with self.subTest(variant=name):
                declared, allowed = isa.allowed_families(march, features)
                self.assertIn("fp16", declared)
                self.assertFalse({"sve", "sve2", "sme"} & set(march.split("+")))
                self.assertLessEqual(declared, allowed)


class BuildLinuxTest(unittest.TestCase):
    def build(self, arch: str, backend: str, host: str = "arm64") -> tuple[list[list[str]], list[tuple[Path, Path]]]:
        commands: list[list[str]] = []
        copies: list[tuple[Path, Path]] = []
        args = argparse.Namespace(arch=arch, backend=backend, clean=False, jobs=3)
        with (
            mock.patch.object(build.platform, "system", return_value="Linux"),
            mock.patch.object(build, "detect_linux_arch", return_value=host),
            mock.patch.object(build, "ensure_submodule"),
            mock.patch.object(build, "ensure_vulkan_header_submodules"),
            mock.patch.object(build, "clean_build_dir"),
            mock.patch.object(build, "configure_and_build", return_value=Path("/build/primary")),
            mock.patch.object(build, "copy_runtime_libraries"),
            mock.patch.object(build, "run", side_effect=commands.append),
            mock.patch.object(build, "copy_output", side_effect=lambda src, dst: copies.append((src, dst))),
            mock.patch.object(build.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch.object(Path, "is_file", return_value=True),
        ):
            build.build_linux(args)
        return commands, copies

    def test_core_lanes_build_each_variant_in_its_own_configure(self) -> None:
        for backend in build.LINUX_ARM64_CPU_VARIANT_BACKENDS:
            with self.subTest(backend=backend):
                commands, copies = self.build("arm64", backend, host="x64")
                configures = [command for command in commands if "--preset" in command]
                self.assertEqual(len(build.LINUX_ARM64_CPU_VARIANTS), len(configures))
                for configure, (name, march, features) in zip(configures, build.LINUX_ARM64_CPU_VARIANTS):
                    self.assertIn(str(build.BUILD_ROOT / f"linux-arm64-cpu-{name}"), configure)
                    self.assertIn(f"-DGGML_CPU_ARM_ARCH={march}", configure)
                    self.assertIn(f"-DLLAMADART_CPU_REQUIRED_FEATURES={';'.join(features)}", configure)
                    self.assertIn("-DGGML_VULKAN=OFF", configure)
                    self.assertIn("-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc", configure)
                self.assertEqual(
                    [build.BIN_DIR / "linux/arm64" / lib for lib in build.linux_arm64_cpu_variant_libs()],
                    [dst for _src, dst in copies],
                )

    def test_single_module_lanes_and_x64_build_no_variant(self) -> None:
        for arch, backend in (("arm64", "blas"), ("x64", "cpu"), ("x64", "full")):
            with self.subTest(arch=arch, backend=backend):
                commands, copies = self.build(arch, backend, host=arch)
                self.assertEqual([], commands)
                self.assertEqual([], copies)


class InstructionFamilyTest(unittest.TestCase):
    def test_base_half_conversions_are_armv8_0(self) -> None:
        for instruction in ("fcvtl v0.4s, v1.4h", "fcvtn v0.4h, v1.4s", "fcvt s0, h1", "ldr h0, [x0]", "dup v0.8h, w0"):
            with self.subTest(instruction=instruction):
                self.assertIsNone(isa.instruction_family(instruction))

    def test_extension_families(self) -> None:
        cases = {
            "fmla v0.8h, v1.8h, v2.8h": "fp16",
            "fadd h0, h1, h2": "fp16",
            "fcvtzs w0, h0": "fp16",
            "scvtf h0, w0": "fp16",
            "fmla v0.8h, v1.8h, v2.h[3]": "fp16",
            "fmlal v0.4s, v1.4h, v2.4h": "fhm",
            "sdot v0.4s, v1.16b, v2.16b": "dotprod",
            "smmla v0.4s, v1.16b, v2.16b": "i8mm",
            "bfcvtn v0.4h, v1.4s": "bf16",
            "sqrdmlah v0.4s, v1.4s, v2.4s": "rdm",
            "stlur w0, [x1, #4]": "rcpc2",
        }
        for instruction, family in cases.items():
            with self.subTest(instruction=instruction):
                self.assertEqual(family, isa.instruction_family(instruction))

    def test_march_version_implies_families(self) -> None:
        self.assertEqual((set(), set()), isa.allowed_families("armv8-a", ()))
        declared, allowed = isa.allowed_families(
            "armv8.6-a+dotprod+fp16+i8mm",
            ("GGML_USE_DOTPROD", "GGML_USE_FP16_VECTOR_ARITHMETIC", "GGML_USE_MATMUL_INT8"),
        )
        self.assertEqual({"dotprod", "fp16", "i8mm"}, declared)
        self.assertEqual({"dotprod", "fp16", "i8mm", "bf16", "rdm", "rcpc2", "fhm"}, allowed)
        with self.assertRaises(ValueError):
            isa.allowed_families("native", ())


class ValidateTest(unittest.TestCase):
    def validate(self, variant: str, symbols: list[str], *lines: str) -> str:
        return isa.validate(variant, "\n".join(symbols), disassembly(*lines), frozenset())

    def test_baseline_allows_extensions_only_inside_kleidiai(self) -> None:
        result = self.validate("baseline", [KAI, GGML], "100: 4e829420 sdot v0.4s, v1.16b, v2.16b", RET)
        self.assertIn("dotprod=0/1", result)

    def test_baseline_rejects_fp16_arithmetic_outside_kleidiai(self) -> None:
        with self.assertRaisesRegex(ValueError, r"fmla.*\(fp16\)"):
            self.validate("baseline", [KAI, GGML], FMLA_8H, RET)

    def test_baseline_rejects_unsized_code_using_an_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "dotprod"):
            self.validate("baseline", [KAI], "300: 4e829420 sdot v0.4s, v1.16b, v2.16b")

    def test_variant_requires_its_declared_isa_and_score_export(self) -> None:
        result = self.validate("linux_armv8.2_2", [KAI, GGML, SCORE], FMLA_8H, SDOT, RET)
        self.assertIn("fp16=1/0, dotprod=1/0", result)
        with self.assertRaisesRegex(ValueError, "lacks declared ISA outside KleidiAI: fp16"):
            self.validate("linux_armv8.2_2", [KAI, GGML, SCORE], SDOT, RET)
        with self.assertRaisesRegex(ValueError, "ggml_backend_score"):
            self.validate("linux_armv8.2_2", [KAI, GGML], FMLA_8H, SDOT, RET)

    def test_variant_rejects_isa_beyond_its_march(self) -> None:
        with self.assertRaisesRegex(ValueError, r"smmla.*\(i8mm\)"):
            self.validate(
                "linux_armv8.2_2", [KAI, GGML, SCORE], FMLA_8H, SDOT, "208: 4e82a420 smmla v0.4s, v1.16b, v2.16b"
            )

    def test_sve_outside_audited_functions_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "SVE/SME escaped"):
            self.validate("baseline", [KAI, GGML], "200: 2518e3e0 ptrue p0.b", RET)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Linux arm64 CPU variant"):
            isa.variant_spec("linux_armv9.2_1")


class WorkflowWiringTest(unittest.TestCase):
    def test_pr_and_release_lanes_check_the_packaged_arm64_variants(self) -> None:
        check = 'run: tools/check_linux_arm64_cpu_variants.sh "${RUNNER_TEMP}/llamadart-clean-arm64"'
        pr_job = (ROOT / ".github/workflows/validate_wrapper.yml").read_text().split("\n  linux-artifact-contract:\n")[1]
        pr_job = pr_job.split("\n  wrapper-contract:\n")[0]
        release_job = (ROOT / ".github/workflows/native_release.yml").read_text().split("\n  build-linux:\n")[1]
        release_job = release_job.split("\n  build-linux-hip:\n")[0]
        for job in (pr_job, release_job):
            self.assertIn(check, job)
            self.assertIn("qemu-user llvm", job)
        self.assertIn("if: ${{ matrix.arch == 'arm64' && matrix.include_core }}", release_job)
        script = (ROOT / "tools/check_linux_arm64_cpu_variants.sh").read_text()
        self.assertIn("tools/build.py\" list --linux-arm64-cpu-variant-libs", script)
        for name, _march, _features in build.LINUX_ARM64_CPU_VARIANTS:
            self.assertIn(f"libggml-cpu-{name}.so", script)


if __name__ == "__main__":
    unittest.main()
