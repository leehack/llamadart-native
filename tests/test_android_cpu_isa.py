import importlib.util
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "android_cpu_isa", Path(__file__).resolve().parents[1] / "tools/validate_android_cpu_isa.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def symbol(name="audited", start=0x100, size=8):
    return f" 1: {start:016x} {size} FUNC GLOBAL DEFAULT 14 {name}"


def disassembly(body):
    return "test.so: file format elf64-littleaarch64\nDisassembly of section .text:\n" + body


class AndroidCpuIsaTest(unittest.TestCase):
    def check(self, symbols, body, allowed=frozenset({"audited"})):
        return audit.validate_disassembly(symbols, disassembly(body), allowed)

    def test_exact_function_range_contains_nested_assembly_labels(self):
        result = self.check(symbol(), "100: 04bf5020 rdvl x0, #1\n104 <local_label>:\n104: d65f03c0 ret")
        self.assertEqual(result, (2, 1, 1))

    def test_baseline_without_kleidiai_remains_strictly_valid(self):
        self.assertEqual(self.check(symbol("baseline"), "100: d65f03c0 ret", frozenset()), (1, 0, 0))

    def test_sve_in_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.check(symbol() + "\n" + symbol("baseline", 0x200),
                       "100: 04bf5020 rdvl x0, #1\n200: 2518e3e0 ptrue p0.b")

    def test_kai_prefix_is_not_an_exception(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.check(symbol() + "\n" + symbol("kai_new_unreviewed", 0x200),
                       "100: 04bf5020 rdvl x0, #1\n200: 2518e3e0 ptrue p0.b")

    def test_range_end_is_exclusive(self):
        with self.assertRaisesRegex(ValueError, "no sized function"):
            self.check(symbol(), "108: 04bf5020 rdvl x0, #1")

    def test_ambiguous_overlapping_function_fails(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.check(symbol() + "\n" + symbol("baseline"), "100: 04bf5020 rdvl x0, #1")

    def test_missing_or_zero_sized_optimized_function_fails(self):
        for symbols in (symbol("baseline"), symbol("audited", size=0), ""):
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                self.check(symbols, "100: 04bf5020 rdvl x0, #1")

    def test_optimized_kernel_cannot_disappear(self):
        with self.assertRaisesRegex(ValueError, "lost its scalable"):
            self.check(symbol(), "100: d65f03c0 ret")

    def test_empty_wrong_arch_unknown_or_malformed_disassembly_fails(self):
        for body in ("", "100: 00000000 <unknown>", "100: 00000000 .word 0", "100: rdvl x0, #1"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.check(symbol(), body)
        with self.assertRaisesRegex(ValueError, "AArch64"):
            audit.validate_disassembly(symbol(), "file format elf64-x86-64")

    def test_scalable_forms_beyond_original_six_mnemonics(self):
        for instruction in ("smstart sm", "smstop", "rdsvl x0, #1", "cntd x0", "incw x0",
                            "addsvl x0, x0, #1", "fmopa za0.s, p0/m, p1/m, z0.s, z1.s",
                            "ldr z0, [x0]", "str p0, [x0]", "zero {za}"):
            with self.subTest(instruction=instruction):
                self.assertTrue(audit.is_scalable(0, instruction))
        self.assertTrue(audit.is_scalable(0x04000000, "future_sve_alias x0"))
        self.assertFalse(audit.is_scalable(0xD65F03C0, "ret"))

    def test_duplicate_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.check(symbol(), "100: 04bf5020 rdvl x0, #1\n100: d65f03c0 ret")

    def test_dispatch_or_probe_source_mutation_is_rejected(self):
        with mock.patch.object(audit, "tree_digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Unaudited"):
                audit.validate_sources(Path("llama"), Path("kleidiai"))

    def test_tree_hash_binds_names_contents_and_new_callers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "probe.cpp"
            first.write_text("guarded")
            original = audit.tree_digest(root)
            first.write_text("bypassed")
            self.assertNotEqual(original, audit.tree_digest(root))
            first.write_text("guarded")
            first.rename(root / "caller.cpp")
            self.assertNotEqual(original, audit.tree_digest(root))
            (root / "probe.cpp").write_text("guarded")
            self.assertNotEqual(original, audit.tree_digest(root))

    def test_missing_source_tree_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty"):
                audit.tree_digest(directory)

    def test_cli_legacy_baseline_does_not_require_kleidiai_source(self):
        args = ["audit", "--objdump", "objdump", "--readelf", "readelf",
                "--llama-source", "absent", "--kleidiai-source", "absent", "test.so"]
        outputs = [symbol("baseline"), disassembly("100: d65f03c0 ret")]
        with mock.patch("sys.argv", args), mock.patch.object(audit.subprocess, "check_output", side_effect=outputs), \
                mock.patch.object(audit, "validate_sources") as validate_sources, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(), 0)
            validate_sources.assert_not_called()

    def test_cli_scalable_artifact_requires_dispatch_source_audit(self):
        args = ["audit", "--objdump", "objdump", "--readelf", "readelf",
                "--llama-source", "absent", "--kleidiai-source", "absent", "test.so"]
        outputs = [symbol(), disassembly("100: 04bf5020 rdvl x0, #1")]
        with mock.patch("sys.argv", args), mock.patch.object(audit.subprocess, "check_output", side_effect=outputs), \
                mock.patch.object(audit, "validate_sources", side_effect=ValueError("unaudited")) as validate_sources, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(), 1)
            validate_sources.assert_called_once()


if __name__ == "__main__":
    unittest.main()
