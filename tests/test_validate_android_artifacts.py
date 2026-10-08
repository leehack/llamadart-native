import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import validate_android_artifacts as validator

RUNTIME = set(validator.EXCEPTION_RUNTIME_SYMBOLS)

# readelf output for one library: the LOAD headers and its dynamic symbols.
SEGMENTS = """\
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x001000 0x001000 R E 0x4000
"""


def dynamic_symbols(defined=(), imported=()):
    lines = [
        "Symbol table '.dynsym' contains 9 entries:",
        '   Num:    Value          Size Type    Bind   Vis       Ndx Name',
        '     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND ',
    ]
    for index, name in enumerate(sorted(defined), start=1):
        lines.append(f'   {index:3}: 000000000007e7e4    92 FUNC    GLOBAL DEFAULT    14 {name}')
    for index, name in enumerate(sorted(imported), start=100):
        lines.append(f'   {index:3}: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND {name}@LIBC')
    return '\n'.join(lines) + '\n'


def shared_runtime():
    return {
        'libggml-base.so': (RUNTIME, set()),
        'libggml-cpu.so': (set(), {'__cxa_throw'}),
        'libllama.so': (set(), RUNTIME),
        'libllamadart.so': (set(), {'__cxa_begin_catch', '_ZTISt9exception', '__cxa_throw'}),
    }


class ExceptionRuntimeTests(unittest.TestCase):
    def test_one_library_defines_the_runtime_and_the_others_import_it(self):
        self.assertEqual([], validator.exception_runtime_errors(shared_runtime()))

    def test_a_second_definition_is_rejected(self):
        symbols = shared_runtime()
        symbols['libllama.so'] = ({'__cxa_throw', '_ZTISt9exception'}, RUNTIME - {'__cxa_throw', '_ZTISt9exception'})
        errors = validator.exception_runtime_errors(symbols)
        self.assertTrue(any(e.startswith('__cxa_throw: defined by libggml-base.so, libllama.so') for e in errors), errors)
        self.assertTrue(any(e.startswith('_ZTISt9exception: defined by') for e in errors), errors)

    def test_a_wrapper_with_a_private_runtime_is_rejected(self):
        symbols = shared_runtime()
        symbols['libllamadart.so'] = (set(), set())
        errors = validator.exception_runtime_errors(symbols)
        self.assertEqual(2, len(errors), errors)
        self.assertTrue(all(e.startswith('libllamadart.so: does not import') for e in errors), errors)

    def test_a_thrower_with_a_private_runtime_is_rejected(self):
        symbols = shared_runtime()
        symbols['libllama.so'] = (set(), {'__cxa_begin_catch'})
        errors = validator.exception_runtime_errors(symbols)
        self.assertTrue(any(e.startswith('libllama.so: does not import __cxa_throw') for e in errors), errors)
        self.assertTrue(any(e.startswith('libllama.so: does not import _ZTISt13runtime_error') for e in errors), errors)

    def test_a_runtime_outside_the_bundle_is_rejected(self):
        symbols = shared_runtime()
        del symbols['libggml-base.so']
        errors = validator.exception_runtime_errors(symbols)
        self.assertTrue(any('defined by no library of the bundle' in e for e in errors), errors)

    def test_a_backend_module_alone_is_accepted(self):
        self.assertEqual([], validator.exception_runtime_errors({'libggml-opencl.so': (set(), set())}))

    def run_tool(self, libraries):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp) / 'out'
            out_dir.mkdir()
            listings = Path(temp) / 'listings'
            listings.mkdir()
            for name, (defined, imported) in libraries.items():
                (out_dir / name).write_bytes(b'')
                (listings / f'{name}.syms').write_text(dynamic_symbols(defined, imported))
            readelf = Path(temp) / 'llvm-readelf'
            readelf.write_text(
                '#!/bin/sh\n'
                'for last; do :; done\n'
                'name=$(basename "$last")\n'
                f'if [ "$1" = "--dyn-syms" ]; then cat "{listings}/$name.syms"; '
                f'else cat "{Path(temp) / "segments"}"; fi\n')
            readelf.chmod(0o755)
            (Path(temp) / 'segments').write_text(SEGMENTS)
            return subprocess.run(
                [sys.executable, str(ROOT / 'tools/validate_android_artifacts.py'),
                 '--readelf', str(readelf), str(out_dir)],
                capture_output=True, text=True)

    def test_tool_accepts_the_shared_runtime_and_names_its_library(self):
        result = self.run_tool(shared_runtime())
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('__cxa_throw: defined by libggml-base.so', result.stdout)

    def test_tool_rejects_a_second_definition(self):
        libraries = shared_runtime()
        libraries['libmtmd.so'] = ({'_ZTISt13runtime_error'}, set())
        result = self.run_tool(libraries)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('_ZTISt13runtime_error: defined by libggml-base.so, libmtmd.so', result.stderr)


if __name__ == '__main__':
    unittest.main()
