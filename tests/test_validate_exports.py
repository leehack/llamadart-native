import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import validate_exports

NM_OUTPUT = """\
                 U ___cxa_finalize
                 U _dlsym
0000000000004a10 T _llama_dart_exit_free
0000000000004b20 T _llama_dart_decode
"""

# Dynamic symbols only; static definitions must not satisfy export checks.
READELF_OUTPUT = """\

Symbol table '.dynsym' contains 3 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@GLIBC_2.17 (2)
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND dlsym@GLIBC_2.34 (3)
     3: 0000000000004b20    64 FUNC    GLOBAL DEFAULT   12 llama_dart_decode

"""

READELF_STATIC = """\
Symbol table '.symtab' contains 3 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 000000000000f900     8 FUNC    LOCAL  DEFAULT   11 __cxa_atexit
     2: 000000000000f910     8 FUNC    GLOBAL DEFAULT   11 llama_dart_encode
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND static_only_reference
"""

READELF_IMPORT = (
    '     4: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND '
    '__cxa_atexit@GLIBC_2.17 (2)\n'
)


class ValidateExportsTests(unittest.TestCase):
    def test_readelf_inspects_only_the_dynamic_symbol_table(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'libllamadart.so'
            library.touch()
            args = SimpleNamespace(library=library, format='readelf',
                                   tool='llvm-readelf')
            with patch.object(validate_exports.subprocess, 'run') as run:
                run.return_value.stdout = READELF_OUTPUT
                self.assertEqual(validate_exports.run_tool(args), READELF_OUTPUT)
                self.assertEqual(run.call_args.args[0],
                                 ['llvm-readelf', '--dyn-syms', '--wide', str(library)])

    def test_readelf_requires_visible_dynamic_exports(self):
        listing = READELF_OUTPUT + (
            ' 4: 0000000000005000 64 FUNC GLOBAL HIDDEN 12 hidden_api\n'
            ' 5: 0000000000005100 64 FUNC LOCAL DEFAULT 12 local_api\n'
            ' 6: 0000000000005200 64 FUNC WEAK DEFAULT 12 weak_api@@V1\n'
            ' 7: 0000000000005300 64 FUNC GLOBAL PROTECTED 12 protected_api\n'
            ' 8: 0000000000005400 64 FUNC GLOBAL INTERNAL 12 internal_api\n'
        ) + READELF_STATIC
        self.assertEqual(
            validate_exports.exported_symbols_from_readelf(listing),
            {'llama_dart_decode', 'weak_api', 'protected_api'})

    @unittest.skipIf(sys.platform == 'win32', 'the stand-in readelf is a shell script')
    def test_private_or_static_definition_fails_export_gate(self):
        for listing in (
                READELF_OUTPUT + READELF_STATIC,
                READELF_OUTPUT +
                ' 4: 0000000000005000 64 FUNC GLOBAL HIDDEN 12 llama_dart_encode\n',
                READELF_OUTPUT +
                ' 4: 0000000000005000 64 FUNC LOCAL DEFAULT 12 llama_dart_encode\n'):
            with self.subTest(listing=listing):
                result = self.run_tool(listing, tool_format='readelf',
                                       symbol='llama_dart_encode')
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('Missing exports', result.stderr)

    def test_nm_exports_and_imports_are_told_apart(self):
        self.assertEqual(
            validate_exports.exported_symbols_from_nm(NM_OUTPUT),
            {'llama_dart_exit_free', 'llama_dart_decode'})
        self.assertEqual(
            validate_exports.imported_symbols_from_nm(NM_OUTPUT),
            {'__cxa_finalize', 'dlsym'})

    def test_readelf_imports_are_the_undefined_symbols(self):
        self.assertEqual(
            validate_exports.imported_symbols_from_readelf(READELF_OUTPUT + READELF_STATIC),
            {'__cxa_finalize', 'dlsym'})
        self.assertEqual(
            validate_exports.imported_symbols_from_readelf(
                READELF_OUTPUT + READELF_IMPORT),
            {'__cxa_finalize', 'dlsym', '__cxa_atexit'})

    def run_tool(self, nm_output, *arguments, library_bytes=b'',
                 tool_format='nm', symbol='llama_dart_decode'):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'libllamadart.dylib'
            library.write_bytes(library_bytes)
            listing = Path(temp) / 'nm.txt'
            listing.write_text(nm_output)
            tool = Path(temp) / 'nm'
            tool.write_text(f'#!/bin/sh\ncat "{listing}"\n')
            tool.chmod(0o755)
            return subprocess.run(
                [sys.executable, str(ROOT / 'tools/validate_exports.py'),
                 '--format', tool_format, '--tool', str(tool),
                 '--symbol', symbol, *arguments, str(library)],
                capture_output=True, text=True)

    @unittest.skipIf(sys.platform == 'win32', 'the stand-in nm is a shell script')
    def test_forbidden_import_in_readelf_output(self):
        # The hidden definition in the symbol table is not an import.
        absent = self.run_tool(
            READELF_OUTPUT + READELF_STATIC, '--forbid-import', '__cxa_atexit',
            tool_format='readelf')
        self.assertEqual(absent.returncode, 0, absent.stderr)

        imported = self.run_tool(
            READELF_OUTPUT + READELF_IMPORT + READELF_STATIC, '--forbid-import', '__cxa_atexit',
            tool_format='readelf')
        self.assertEqual(imported.returncode, 1)
        self.assertIn('__cxa_atexit', imported.stderr)

    @unittest.skipIf(sys.platform == 'win32', 'the stand-in nm is a shell script')
    def test_imports_only_skips_the_required_exports(self):
        # A llama.cpp library exports none of the wrapper's functions.
        missing = self.run_tool(
            READELF_OUTPUT, '--forbid-import', '__cxa_atexit',
            tool_format='readelf', symbol='llama_dart_encode')
        self.assertEqual(missing.returncode, 1)
        self.assertIn('llama_dart_encode', missing.stderr)

        skipped = self.run_tool(
            READELF_OUTPUT, '--forbid-import', '__cxa_atexit', '--imports-only',
            tool_format='readelf', symbol='llama_dart_encode')
        self.assertEqual(skipped.returncode, 0, skipped.stderr)

        imported = self.run_tool(
            READELF_OUTPUT + READELF_IMPORT, '--forbid-import', '__cxa_atexit',
            '--imports-only', tool_format='readelf')
        self.assertEqual(imported.returncode, 1)
        self.assertIn('__cxa_atexit', imported.stderr)

        unchecked = self.run_tool(
            READELF_OUTPUT, '--imports-only', tool_format='readelf')
        self.assertEqual(unchecked.returncode, 2)
        self.assertIn('--forbid-import', unchecked.stderr)

    @unittest.skipIf(sys.platform == 'win32', 'the stand-in nm is a shell script')
    def test_forbidden_import_fails_only_when_imported(self):
        absent = self.run_tool(NM_OUTPUT, '--forbid-import', '__cxa_atexit')
        self.assertEqual(absent.returncode, 0, absent.stderr)

        imported = self.run_tool(
            NM_OUTPUT + '                 U ___cxa_atexit\n',
            '--forbid-import', '__cxa_atexit')
        self.assertEqual(imported.returncode, 1)
        self.assertIn('__cxa_atexit', imported.stderr)

        # A definition of the same name inside the library is not an import.
        defined = self.run_tool(
            NM_OUTPUT + '0000000000002ff8 T ___cxa_atexit\n',
            '--forbid-import', '__cxa_atexit')
        self.assertEqual(defined.returncode, 0, defined.stderr)

    @unittest.skipIf(sys.platform == 'win32', 'the stand-in nm is a shell script')
    def test_forbidden_import_refuses_a_universal_file(self):
        # nm would list one slice of it only.
        for magic in (b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf',
                      b'\xbe\xba\xfe\xca', b'\xbf\xba\xfe\xca'):
            with self.subTest(magic=magic):
                universal = self.run_tool(
                    NM_OUTPUT, '--forbid-import', '__cxa_atexit',
                    library_bytes=magic + b'\0' * 28)
                self.assertEqual(universal.returncode, 2)
                self.assertIn('single-architecture', universal.stderr)
        thin = self.run_tool(
            NM_OUTPUT, '--forbid-import', '__cxa_atexit',
            library_bytes=b'\xcf\xfa\xed\xfe' + b'\0' * 28)
        self.assertEqual(thin.returncode, 0, thin.stderr)
        # Without the import check a universal file is accepted as before.
        unchecked = self.run_tool(
            NM_OUTPUT, library_bytes=b'\xca\xfe\xba\xbe' + b'\0' * 28)
        self.assertEqual(unchecked.returncode, 0, unchecked.stderr)


if __name__ == '__main__':
    unittest.main()
