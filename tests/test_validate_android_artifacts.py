import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
import validate_android_artifacts as validator
import llama_cpp_patches as patches
import build
from workflow_text import job_block, matrix_rows, step_running

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

    def run_tool(self, libraries, omitted_marker=None, omitted_library=None, flags=(),
                 library_contents=None):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp) / 'out'
            out_dir.mkdir()
            listings = Path(temp) / 'listings'
            listings.mkdir()
            for name, (defined, imported) in libraries.items():
                markers = [marker['text'] for entry in patches.load_series()
                           for marker in entry.get('android_markers', [])
                           if marker['library'] in (name, 'libggml-cpu.so' if name.startswith('libggml-cpu') else name)
                           and not (marker['text'] == omitted_marker and omitted_library in (None, name))]
                contents = '\0'.join(markers).encode()
                contents += (library_contents or {}).get(name, b'')
                (out_dir / name).write_bytes(contents)
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
                 '--readelf', str(readelf), *flags, str(out_dir)],
                capture_output=True, text=True)

    def test_tool_accepts_the_shared_runtime_and_names_its_library(self):
        result = self.run_tool(shared_runtime())
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('__cxa_throw: defined by libggml-base.so', result.stdout)

    def test_tool_requires_the_f16_dot_repair_marker(self):
        result = self.run_tool(shared_runtime(),
                               omitted_marker='llamadart F16 dot F32 accumulation')
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn('libggml-cpu.so: built without 0005-arm-f16-dot-f32-accumulation.patch',
                      result.stderr)

    def arm64_bundle(self):
        libraries = shared_runtime()
        del libraries['libggml-cpu.so']
        for name in build.android_arm64_cpu_variant_libs():
            libraries[name] = (set(), {'__cxa_throw'})
        return libraries

    def test_tool_accepts_every_shipped_arm64_cpu_variant_with_its_markers(self):
        result = self.run_tool(self.arm64_bundle())
        self.assertEqual(0, result.returncode, result.stderr)

    def test_tool_rejects_an_arm64_cpu_variant_without_a_marker(self):
        for variant in build.android_arm64_cpu_variant_libs():
            result = self.run_tool(
                self.arm64_bundle(), omitted_marker='llamadart F16 matmul route',
                omitted_library=variant)
            self.assertEqual(1, result.returncode, variant)
            self.assertIn(
                f'{variant}: built without 0008-arm-fp16-unhinted-f16-matmul-route.patch',
                result.stderr)

    def test_tool_rejects_a_bundle_missing_an_arm64_cpu_variant(self):
        libraries = self.arm64_bundle()
        missing = build.android_arm64_cpu_variant_libs()[-1]
        del libraries[missing]
        result = self.run_tool(libraries)
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn(f'{missing}: not in the bundle', result.stderr)

    def test_tool_rejects_a_bundle_with_no_cpu_module(self):
        libraries = shared_runtime()
        del libraries['libggml-cpu.so']
        result = self.run_tool(libraries)
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn('libggml-cpu.so: not in', result.stderr)

    def test_requires_a_cpu_even_with_no_marker_requirements(self):
        self.assertEqual(
            ['libggml-cpu.so: not in the bundle'],
            validator.cpu_variant_errors([Path('libggml-opencl.so')]))

    def test_opencl_packaging_skips_markers_on_unshipped_cpu_variants(self):
        libraries = self.arm64_bundle()
        libraries['libggml-opencl.so'] = (set(), set())
        result = self.run_tool(
            libraries, omitted_marker='llamadart F16 matmul route',
            flags=('--backend-only',))
        self.assertEqual(0, result.returncode, result.stderr)

    def test_backend_only_validation_keeps_non_cpu_marker_checks(self):
        libraries = {'libggml-vulkan.so': (set(), set())}
        result = self.run_tool(
            libraries, omitted_marker='has no vkGetBufferDeviceAddress',
            flags=('--backend-only',))
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn('libggml-vulkan.so: built without', result.stderr)

    def test_vulkan_packaging_rejects_integer_dot_shaders(self):
        libraries = {'libggml-vulkan.so': (set(), set())}
        flags = ('--backend-only',)
        result = self.run_tool(libraries, flags=flags)
        self.assertEqual(0, result.returncode, result.stderr)

        result = self.run_tool(
            libraries, flags=flags,
            library_contents={'libggml-vulkan.so': b'\0mul_mat_vec_q4_0_q8_1_f32\0'},
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn('libggml-vulkan.so: contains an integer-dot shader', result.stderr)

    def test_tool_rejects_a_second_definition(self):
        libraries = shared_runtime()
        libraries['libmtmd.so'] = ({'_ZTISt13runtime_error'}, set())
        result = self.run_tool(libraries)
        self.assertNotEqual(0, result.returncode)
        self.assertIn('_ZTISt13runtime_error: defined by libggml-base.so, libmtmd.so', result.stderr)

    def test_release_workflow_passes_backend_only_only_to_opencl_lanes(self):
        block = job_block('native_release.yml', 'build-android')
        rows = matrix_rows(block)
        backend_only_lanes = [
            (row['android_abi'], row['backend'])
            for row in rows
            if row['include_core'] == 'false'
        ]
        core_lanes = [
            (row['android_abi'], row['backend'])
            for row in rows
            if row['include_core'] == 'true'
        ]
        self.assertEqual(
            [('arm64-v8a', 'opencl'), ('x86_64', 'opencl')],
            backend_only_lanes,
        )
        self.assertEqual(
            [('arm64-v8a', 'vulkan'), ('x86_64', 'vulkan')],
            core_lanes,
        )
        self.assertEqual(len(rows), len(backend_only_lanes) + len(core_lanes))

        step = step_running(block, 'tools/validate_android_artifacts.py')
        self.assertRegex(
            step,
            r'if \[ "\$\{\{ matrix\.include_core \}\}" = "false" \]; then\s+'
            r'ARTIFACT_ARGS\+\=\(--backend-only\)\s+fi',
        )
        self.assertIn(
            'python3 tools/validate_android_artifacts.py --readelf "$READELF" "${ARTIFACT_ARGS[@]}" "$OUT_DIR"',
            step,
        )


if __name__ == '__main__':
    unittest.main()
