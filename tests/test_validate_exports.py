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


class ValidateExportsTests(unittest.TestCase):
    def test_nm_exports_and_imports_are_told_apart(self):
        self.assertEqual(
            validate_exports.exported_symbols_from_nm(NM_OUTPUT),
            {'llama_dart_exit_free', 'llama_dart_decode'})
        self.assertEqual(
            validate_exports.imported_symbols_from_nm(NM_OUTPUT),
            {'__cxa_finalize', 'dlsym'})

    def run_tool(self, nm_output, *arguments):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'libllamadart.dylib'
            library.write_bytes(b'')
            listing = Path(temp) / 'nm.txt'
            listing.write_text(nm_output)
            tool = Path(temp) / 'nm'
            tool.write_text(f'#!/bin/sh\ncat "{listing}"\n')
            tool.chmod(0o755)
            return subprocess.run(
                [sys.executable, str(ROOT / 'tools/validate_exports.py'),
                 '--format', 'nm', '--tool', str(tool),
                 '--symbol', 'llama_dart_decode', *arguments, str(library)],
                capture_output=True, text=True)

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


if __name__ == '__main__':
    unittest.main()
