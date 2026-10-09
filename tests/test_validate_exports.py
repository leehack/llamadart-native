import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import validate_exports

NM_OUTPUT = """\
                 U ___cxa_finalize
                 U _dlsym
0000000000004a10 T _llama_dart_exit_free
0000000000004b20 T _llama_dart_decode
"""

# readelf -Ws of an ELF library that imports two functions and has a hidden
# definition of __cxa_atexit in its full symbol table.
READELF_OUTPUT = """\

Symbol table '.dynsym' contains 3 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@GLIBC_2.17 (2)
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND dlsym@GLIBC_2.34 (3)
     3: 0000000000004b20    64 FUNC    GLOBAL DEFAULT   12 llama_dart_decode

Symbol table '.symtab' contains 1 entry:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 000000000000f900     8 FUNC    LOCAL  DEFAULT   11 __cxa_atexit
"""

READELF_IMPORT = (
    '     4: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND '
    '__cxa_atexit@GLIBC_2.17 (2)\n'
)


class ValidateExportsTests(unittest.TestCase):
    def test_nm_exports_and_imports_are_told_apart(self):
        self.assertEqual(
            validate_exports.exported_symbols_from_nm(NM_OUTPUT),
            {'llama_dart_exit_free', 'llama_dart_decode'})
        self.assertEqual(
            validate_exports.imported_symbols_from_nm(NM_OUTPUT),
            {'__cxa_finalize', 'dlsym'})

    def test_readelf_imports_are_the_undefined_symbols(self):
        self.assertEqual(
            validate_exports.imported_symbols_from_readelf(READELF_OUTPUT),
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
            READELF_OUTPUT, '--forbid-import', '__cxa_atexit',
            tool_format='readelf')
        self.assertEqual(absent.returncode, 0, absent.stderr)

        imported = self.run_tool(
            READELF_OUTPUT + READELF_IMPORT, '--forbid-import', '__cxa_atexit',
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
