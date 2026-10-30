from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
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
LDADD = "300: b8200020 ldadd w0, w0, [x1]"
LDAPR = "304: f8bfc000 ldapr x0, [x0]"
STLUR = "308: 99000020 stlur w0, [x1]"
CRC32W = "30c: 1ac24820 crc32w w0, w1, w2"
OUTLINE_GUARD = (
    "2f0: 90000010 adrp x16, 0x1000",
    "2f4: 39400210 ldrb w16, [x16]",
    "2f8: 34000050 cbz w16, 0x300",
)

DOTPROD = "GGML_USE_DOTPROD"
FP16 = "GGML_USE_FP16_VECTOR_ARITHMETIC"
I8MM = "GGML_USE_MATMUL_INT8"
EXPECTED_VARIANTS = (
    ("linux_armv8.2_2", "armv8.2-a+dotprod+fp16", (DOTPROD, FP16)),
    ("linux_armv8.6_1", "armv8.6-a+dotprod+fp16+i8mm", (DOTPROD, FP16, I8MM)),
)


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

    def test_variants_are_the_audited_set(self) -> None:
        self.assertEqual(EXPECTED_VARIANTS, build.LINUX_ARM64_CPU_VARIANTS)
        self.assertEqual(("full", "cpu", "vulkan"), build.LINUX_ARM64_CPU_VARIANT_BACKENDS)

    def test_scored_features_match_the_march_extensions(self) -> None:
        extension_features = {"dotprod": DOTPROD, "fp16": FP16, "i8mm": I8MM}
        for name, march, features in build.LINUX_ARM64_CPU_VARIANTS:
            with self.subTest(variant=name):
                extensions = set(march.split("+")[1:])
                self.assertFalse({"sve", "sve2", "sme"} & extensions)
                self.assertEqual({extension_features[extension] for extension in extensions}, set(features))
                declared, _allowed = isa.allowed_families(march, features)
                self.assertEqual(extensions, declared)

    def test_cmake_scores_only_variants_with_the_required_feature_defines(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text()
        function = cmake.split("function(llamadart_attach_arm_cpu_backend_score", 1)[1].split("\nendfunction()", 1)[0]
        self.assertIn(
            'if (NOT ANDROID AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND LLAMADART_CPU_REQUIRED_FEATURES))',
            function,
        )
        definitions = function.split("target_compile_definitions(${score_target} PRIVATE", 1)[1].split(")", 1)[0]
        self.assertIn("${LLAMADART_CPU_REQUIRED_FEATURES}", definitions.split())


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
        for backend in ("full", "cpu", "vulkan"):
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

    def test_base_instructions_with_extension_lookalike_names(self) -> None:
        for instruction in ("ldr x0, [x1]", "ldar w0, [x1]", "ldaxr w0, [x1]", "stlr w0, [x1]", "stclr_x", "adrp x16, 0x1000"):
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
            "ldapur w0, [x1, #4]": "rcpc2",
            "ldapr x0, [x1]": "rcpc",
            "ldaprb w0, [x1]": "rcpc",
            "ldaprh w0, [x1]": "rcpc",
            "ldadd w0, w1, [x2]": "lse",
            "ldaddal x0, x1, [x2]": "lse",
            "swpalb w0, w1, [x2]": "lse",
            "casal w0, w1, [x2]": "lse",
            "casp x0, x1, x2, x3, [x4]": "lse",
            "stsetl w0, [x1]": "lse",
            "ldumaxh w0, w1, [x2]": "lse",
            "crc32w w0, w1, w2": "crc",
            "crc32cx w0, w1, x2": "crc",
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
        self.assertEqual({"dotprod", "fp16", "i8mm", "bf16", "rdm", "lse", "rcpc", "rcpc2", "fhm"}, allowed)

    def test_each_march_version_adds_only_its_own_families(self) -> None:
        expected = {
            "armv8-a": set(),
            "armv8.1-a": {"rdm", "lse"},
            "armv8.2-a": {"rdm", "lse"},
            "armv8.3-a": {"rdm", "lse", "rcpc"},
            "armv8.4-a": {"rdm", "lse", "rcpc", "rcpc2", "dotprod"},
            "armv8.6-a": {"rdm", "lse", "rcpc", "rcpc2", "dotprod", "bf16", "i8mm"},
        }
        for march, families in expected.items():
            with self.subTest(march=march):
                self.assertEqual(families, isa.allowed_families(march, ())[1])
        self.assertEqual({"rdm", "lse", "fp16"}, isa.allowed_families("armv8.2-a+fp16", (FP16,))[1])
        self.assertEqual({"rdm", "lse", "rcpc", "rcpc2", "dotprod", "fp16", "fhm"},
                         isa.allowed_families("armv8.4-a+fp16", (FP16,))[1])
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

    def test_baseline_allows_lse_only_in_the_outline_atomic_helper(self) -> None:
        result = self.validate("baseline", [KAI, GGML], *OUTLINE_GUARD, LDADD, RET)
        self.assertIn("lse=1/0", result)
        for guard_lines in (OUTLINE_GUARD[1:], OUTLINE_GUARD[:2], (OUTLINE_GUARD[0], OUTLINE_GUARD[2])):
            with self.subTest(guard=guard_lines), self.assertRaisesRegex(ValueError, r"ldadd.*\(lse\)"):
                self.validate("baseline", [KAI, GGML], *guard_lines, LDADD, RET)
        with self.assertRaisesRegex(ValueError, r"ldadd.*\(lse\)"):
            self.validate("baseline", [KAI, GGML], *OUTLINE_GUARD, "2fc: d503201f nop", LDADD, RET)

    def test_outline_atomic_exemption_covers_only_the_lse_family(self) -> None:
        for line, family in ((LDAPR, "rcpc"), (STLUR, "rcpc2"), (CRC32W, "crc"), (SDOT, "dotprod"), (FMLA_8H, "fp16")):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, rf"\({family}\)"):
                self.validate("baseline", [KAI, GGML], *OUTLINE_GUARD, line, RET)

    def test_outline_atomic_guard_must_use_x16_and_cbz(self) -> None:
        adrp, ldrb, cbz = OUTLINE_GUARD
        mutants = {
            "adrp x17": (adrp.replace("x16", "x17"), ldrb, cbz),
            "adrp x15": (adrp.replace("x16", "x15"), ldrb, cbz),
            "ldrb dest": (adrp, ldrb.replace("w16, [x16]", "w15, [x16]"), cbz),
            "ldrb base": (adrp, ldrb.replace("[x16]", "[x15]"), cbz),
            "cbz reg": (adrp, ldrb, cbz.replace("w16", "w15")),
            "cbnz": (adrp, ldrb, cbz.replace("cbz", "cbnz")),
        }
        for name, guard_lines in mutants.items():
            with self.subTest(mutant=name), self.assertRaisesRegex(ValueError, r"ldadd.*\(lse\)"):
                self.validate("baseline", [KAI, GGML], *guard_lines, LDADD, RET)

    def test_outline_atomic_guard_is_recognised_anywhere_in_the_stream(self) -> None:
        bti = "2e8: d503245f bti c"
        nop = "2e0: d503201f nop"
        self.assertIn("lse=1/0", self.validate("baseline", [KAI, GGML], nop, bti, *OUTLINE_GUARD, LDADD, RET))
        second = ("400: 90000010 adrp x16, 0x1000", "404: 39400210 ldrb w16, [x16]", "408: 34000050 cbz w16, 0x410", "40c: b8200020 ldadd w0, w0, [x1]")
        self.assertIn("lse=2/0", self.validate("baseline", [KAI, GGML], *OUTLINE_GUARD, LDADD, RET, bti, *second))
        for lead in ((), (OUTLINE_GUARD[2],), (OUTLINE_GUARD[1], OUTLINE_GUARD[2])):
            with self.subTest(lead=len(lead)), self.assertRaisesRegex(ValueError, r"ldadd.*\(lse\)"):
                self.validate("baseline", [KAI, GGML], *lead, LDADD, RET)

    def test_outline_atomic_guard_must_test_the_lse_flag(self) -> None:
        flag = 0x1000
        guarded = (*OUTLINE_GUARD, LDADD)
        self.assertEqual([], isa.audit(GGML, disassembly(*guarded), set(), flag)[2])
        offset_guard = (OUTLINE_GUARD[0], "2f4: 39408610 ldrb w16, [x16, #0x21]", OUTLINE_GUARD[2])
        self.assertEqual([], isa.audit(GGML, disassembly(*offset_guard, LDADD), set(), 0x1021)[2])
        for wrong in (0x2000, 0x1021):
            with self.subTest(flag=wrong):
                failures = isa.audit(GGML, disassembly(*guarded), set(), wrong)[2]
                self.assertEqual(1, len(failures))
                self.assertIn("__aarch64_have_lse_atomics", failures[0])

    def test_outline_atomic_guards_must_agree_when_the_flag_symbol_is_stripped(self) -> None:
        other = ("30c: 90000010 adrp x16, 0x2000", OUTLINE_GUARD[1], OUTLINE_GUARD[2])
        failures = isa.audit(GGML, disassembly(*OUTLINE_GUARD, LDADD, *other, LDADD), set())[2]
        self.assertEqual(["outline-atomic guards test different flags: 0x1000, 0x2000"], failures)

    def test_lse_flag_address_reads_the_symtab_object(self) -> None:
        symtab = "\n".join(
            [
                "    33: 0000000000000510    44 FUNC    LOCAL  DEFAULT    11 init_have_lse_atomics",
                "    63: 0000000000020021     1 OBJECT  LOCAL  DEFAULT    21 __aarch64_have_lse_atomics",
            ]
        )
        self.assertEqual(0x20021, isa.lse_flag_address(symtab))
        self.assertIsNone(isa.lse_flag_address(symtab.splitlines()[0]))

    def test_baseline_rejects_each_armv8_3_to_8_6_family_the_qemu_smoke_would_trap_on(self) -> None:
        for line, family in ((LDAPR, "rcpc"), (STLUR, "rcpc2"), (CRC32W, "crc")):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, rf"\({family}\)"):
                self.validate("baseline", [KAI, GGML], line, RET)

    def test_variants_allow_exactly_the_lse_and_rcpc_families_of_their_march(self) -> None:
        guarded = (*OUTLINE_GUARD, LDADD)
        self.assertIn("lse=1/0", self.validate("linux_armv8.2_2", [KAI, GGML, SCORE], FMLA_8H, SDOT, LDADD, RET))
        self.assertIn("lse=1/0", self.validate("linux_armv8.2_2", [KAI, GGML, SCORE], FMLA_8H, SDOT, *guarded, RET))
        for line, family in ((LDAPR, "rcpc"), (STLUR, "rcpc2"), (CRC32W, "crc")):
            with self.subTest(variant="linux_armv8.2_2", family=family), self.assertRaisesRegex(
                ValueError, rf"\({family}\)"
            ):
                self.validate("linux_armv8.2_2", [KAI, GGML, SCORE], FMLA_8H, SDOT, line, RET)
        result = self.validate(
            "linux_armv8.6_1",
            [KAI, GGML, SCORE],
            FMLA_8H,
            SDOT,
            "208: 4e82a420 smmla v0.4s, v1.16b, v2.16b",
            LDADD,
            LDAPR,
            STLUR,
            RET,
        )
        self.assertIn("lse=1/0, rcpc=1/0, rcpc2=1/0", result)
        with self.assertRaisesRegex(ValueError, r"\(crc\)"):
            self.validate(
                "linux_armv8.6_1",
                [KAI, GGML, SCORE],
                FMLA_8H,
                SDOT,
                "208: 4e82a420 smmla v0.4s, v1.16b, v2.16b",
                CRC32W,
                RET,
            )

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


class CommandLineFlagWiringTest(unittest.TestCase):
    """main() must read the flag from .symtab and carry it down to the audit."""

    def run_audit(self, symtab: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libggml-cpu.so"
            library.write_bytes(b"")
            (root / "dynsym.txt").write_text(GGML + "\n")
            (root / "symtab.txt").write_text(symtab + "\n")
            (root / "disassembly.txt").write_text(disassembly(*OUTLINE_GUARD, LDADD, RET) + "\n")
            readelf = root / "readelf"
            readelf.write_text(
                "#!/bin/sh\n"
                f'if [ "$1" = "--syms" ]; then cat "{root}/symtab.txt"; else cat "{root}/dynsym.txt"; fi\n'
            )
            objdump = root / "objdump"
            objdump.write_text(f'#!/bin/sh\ncat "{root}/disassembly.txt"\n')
            readelf.chmod(0o755)
            objdump.chmod(0o755)
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/validate_linux_arm64_cpu_isa.py"),
                    "--objdump", str(objdump),
                    "--readelf", str(readelf),
                    "--llama-source", directory,
                    "--kleidiai-source", directory,
                    "--variant", "baseline",
                    str(library),
                ],
                capture_output=True,
                text=True,
            )

    @staticmethod
    def flag_symbol(address: int) -> str:
        return f"  7: {address:016x} 1 OBJECT LOCAL HIDDEN 21 {isa.LSE_FLAG_SYMBOL}"

    def test_a_guard_testing_another_address_fails(self) -> None:
        result = self.run_audit(self.flag_symbol(0x2000))
        self.assertEqual(1, result.returncode, result.stdout)
        self.assertIn(f"guard tests 0x1000 not {isa.LSE_FLAG_SYMBOL} at 0x2000", result.stderr)

    def test_a_guard_testing_the_flag_passes(self) -> None:
        result = self.run_audit(self.flag_symbol(0x1000))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("PASS baseline", result.stdout)

    def test_a_stripped_library_falls_back_to_guard_agreement(self) -> None:
        result = self.run_audit("  1: 0000000000000000 0 NOTYPE LOCAL DEFAULT UND")
        self.assertEqual(0, result.returncode, result.stderr)


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

    def test_qemu_smoke_covers_every_selection_outcome(self) -> None:
        script = (ROOT / "tools/check_linux_arm64_cpu_variants.sh").read_text()
        cpus = script.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0].splitlines()
        self.assertEqual(
            [
                "cortex-a53 libggml-cpu.so",
                "cortex-a72 libggml-cpu.so",
                "neoverse-n1 libggml-cpu-linux_armv8.2_2.so",
                "neoverse-v1 libggml-cpu-linux_armv8.6_1.so",
                "neoverse-n2 libggml-cpu-linux_armv8.6_1.so",
            ],
            cpus,
        )


if __name__ == "__main__":
    unittest.main()
