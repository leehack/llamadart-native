import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build
import validate_windows_bundle
import windows_runtime

SECTION_RVA = 0x1000
SECTION_OFFSET = 0x400
MACHINES = {'x64': 0x8664, 'arm64': 0xAA64}


def make_pe(path, *, arch='x64', linker=(14, 44), imports=None, delay_imports=None,
            exports=(), file_version=None):
    """Writes a PE32+ DLL whose one section holds only the requested tables."""
    section = bytearray()

    def add(data, align=8):
        section.extend(b'\0' * (-len(section) % align))
        rva = SECTION_RVA + len(section)
        section.extend(data)
        return rva

    def thunks(symbols):
        names = [add(struct.pack('<H', 0) + symbol.encode() + b'\0', 2) for symbol in symbols]
        return add(b''.join(struct.pack('<Q', rva) for rva in names) + bytes(8))

    directories = [(0, 0)] * 16
    descriptors = b''
    for dll, symbols in (imports or {}).items():
        table = thunks(symbols)
        descriptors += struct.pack('<IIIII', table, 0, 0, add(dll.encode() + b'\0', 2), table)
    if descriptors:
        directories[1] = (add(descriptors + bytes(20), 4), len(descriptors) + 20)
    descriptors = b''
    for dll, symbols in (delay_imports or {}).items():
        table = thunks(symbols)
        name = add(dll.encode() + b'\0', 2)
        descriptors += struct.pack('<IIIIIIII', 1, name, add(bytes(8)), table, table, 0, 0, 0)
    if descriptors:
        directories[13] = (add(descriptors + bytes(32), 4), len(descriptors) + 32)
    if exports:
        names = [add(symbol.encode() + b'\0', 2) for symbol in exports]
        functions = add(b''.join(struct.pack('<I', SECTION_RVA) for _ in exports), 4)
        name_table = add(b''.join(struct.pack('<I', rva) for rva in names), 4)
        ordinals = add(b''.join(struct.pack('<H', index) for index in range(len(exports))), 2)
        directory = struct.pack('<IIHHIIIIIII', 0, 0, 0, 0, names[0], 1, len(exports), len(exports),
                                functions, name_table, ordinals)
        directories[0] = (add(directory, 4), 40)
    if file_version:
        major, minor, build_number, revision = file_version
        blob = add(b'VS_VERSION_INFO\0' + struct.pack('<IIII', 0xFEEF04BD, 0x10000,
                                                      major << 16 | minor, build_number << 16 | revision), 4)
        root = SECTION_RVA + len(section) + (-len(section) % 4)
        level = struct.pack('<IIHHHH', 0, 0, 0, 0, 0, 1)
        tree = (level + struct.pack('<II', 16, 0x80000000 | 24)
                + level + struct.pack('<II', 1, 0x80000000 | 48)
                + level + struct.pack('<II', 0x409, 72)
                + struct.pack('<IIII', blob, 36, 0, 0))
        directories[2] = (add(tree, 4), len(tree))
        assert directories[2][0] == root

    header = bytearray(SECTION_OFFSET)
    header[0:2] = b'MZ'
    struct.pack_into('<I', header, 0x3C, 0x40)
    header[0x40:0x44] = b'PE\0\0'
    struct.pack_into('<HHIIIHH', header, 0x44, MACHINES[arch], 1, 0, 0, 0, 240, 0x2022)
    optional = 0x58
    struct.pack_into('<HBB', header, optional, 0x20B, *linker)
    struct.pack_into('<Q', header, optional + 24, 0x180000000)
    struct.pack_into('<I', header, optional + 108, 16)
    for index, (rva, size) in enumerate(directories):
        struct.pack_into('<II', header, optional + 112 + 8 * index, rva, size)
    struct.pack_into('<8sIIII', header, optional + 240, b'.rdata', len(section), SECTION_RVA,
                     len(section), SECTION_OFFSET)
    Path(path).write_bytes(bytes(header) + bytes(section))


CRT = {'vcruntime140.dll': ['memcpy'], 'msvcp140.dll': ['_Thrd_yield']}
RUNTIME = (14, 44, 35211, 0)


def complete_bundle(directory, *, arch='x64', runtime=RUNTIME, linker=(14, 44)):
    make_pe(directory / 'ggml-base.dll', arch=arch, linker=linker, exports=['ggml_init'],
            imports={'KERNEL32.dll': ['GetLastError'], **CRT, 'vcomp140.dll': ['_vcomp_fork']})
    make_pe(directory / 'llama.dll', arch=arch, linker=linker,
            imports={'ggml-base.dll': ['ggml_init'], **CRT, 'api-ms-win-crt-heap-l1-1-0.dll': ['free']})
    make_pe(directory / 'msvcp140.dll', arch=arch, file_version=runtime, exports=['_Thrd_yield'],
            imports={'VCRUNTIME140.dll': ['memcpy']})
    make_pe(directory / 'vcruntime140.dll', arch=arch, file_version=runtime, exports=['memcpy'])
    make_pe(directory / 'vcomp140.dll', arch=arch, exports=['_vcomp_fork'],
            imports={'vcruntime140.dll': ['memcpy']})


def errors_for(directory, arch='x64'):
    images = {path.name.lower(): windows_runtime.read_pe(path) for path in sorted(directory.glob('*.dll'))}
    return validate_windows_bundle.bundle_errors(images, arch)


class PeReaderTests(unittest.TestCase):
    def test_reads_every_table_the_closure_check_needs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.dll'
            make_pe(path, arch='arm64', linker=(14, 51), imports={'MSVCP140.dll': ['?id@?$codecvt@']},
                    delay_imports={'nvcuda.dll': ['cuInit']}, exports=['a', 'b'],
                    file_version=(14, 51, 36231, 1))
            image = windows_runtime.read_pe(path)
        self.assertEqual(image.machine, 'arm64')
        self.assertEqual(image.linker_version, (14, 51))
        self.assertEqual(image.imports, {'msvcp140.dll': frozenset({'?id@?$codecvt@'})})
        self.assertEqual(image.delay_imports, {'nvcuda.dll': frozenset({'cuInit'})})
        self.assertEqual(set(image.all_imports()), {'msvcp140.dll', 'nvcuda.dll'})
        self.assertEqual(image.exports, frozenset({'a', 'b'}))
        self.assertEqual(image.file_version, (14, 51, 36231, 1))

    def test_rejects_a_non_pe_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'text.dll'
            path.write_bytes(b'not a portable executable')
            with self.assertRaises(windows_runtime.PeError):
                windows_runtime.read_pe(path)


class BundleClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)

    def test_complete_bundle_passes(self):
        complete_bundle(self.bundle)
        self.assertEqual(errors_for(self.bundle), [])

    def test_bundle_relying_on_the_installed_runtime_fails(self):
        complete_bundle(self.bundle)
        (self.bundle / 'msvcp140.dll').unlink()
        (self.bundle / 'vcruntime140.dll').unlink()
        errors = errors_for(self.bundle)
        self.assertIn('llama.dll: imports msvcp140.dll, which neither the bundle nor Windows provides', errors)
        self.assertIn('vcomp140.dll: imports vcruntime140.dll, which neither the bundle nor Windows provides',
                      errors)

    def test_shipping_a_windows_library_fails(self):
        complete_bundle(self.bundle)
        make_pe(self.bundle / 'psapi.dll')
        self.assertIn('psapi.dll: Windows provides this library; the bundle must not ship it',
                      errors_for(self.bundle))

    def test_foreign_architecture_fails(self):
        complete_bundle(self.bundle, arch='arm64')
        self.assertIn('llama.dll: arm64 image in a x64 bundle', errors_for(self.bundle))

    def test_missing_export_fails(self):
        complete_bundle(self.bundle)
        make_pe(self.bundle / 'ggml.dll', imports={'ggml-base.dll': ['ggml_removed']})
        self.assertIn('ggml.dll: ggml-base.dll does not export ggml_removed', errors_for(self.bundle))

    def test_runtime_older_than_an_msvc_linker_fails(self):
        complete_bundle(self.bundle, linker=(14, 51))
        self.assertIn('ggml-base.dll: linked by MSVC 14.51, newer than the bundled Visual C++ runtime '
                      '14.44.35211.0', errors_for(self.bundle))

    def test_lld_link_version_is_not_compared(self):
        complete_bundle(self.bundle, arch='arm64', linker=(14, 0))
        self.assertEqual(errors_for(self.bundle, 'arm64'), [])

    def test_runtime_dlls_from_different_redistributables_fail(self):
        complete_bundle(self.bundle)
        make_pe(self.bundle / 'vcruntime140.dll', file_version=(14, 50, 35710, 0), exports=['memcpy'])
        errors = errors_for(self.bundle)
        self.assertEqual(len(errors), 1)
        self.assertIn('come from different redistributables', errors[0])


class RedistLookupTests(unittest.TestCase):
    def test_only_the_desktop_redistributable_folders_are_used(self):
        with tempfile.TemporaryDirectory() as temp:
            redist = Path(temp)
            for relative in ('x64/Microsoft.VC145.CRT/msvcp140.dll',
                             'x64/Microsoft.VC145.OpenMP/vcomp140.dll',
                             'arm64/Microsoft.VC145.OpenMP.LLVM/libomp140.aarch64.dll',
                             'debug_nonredist/x64/Microsoft.VC145.DebugCRT/vcruntime140.dll',
                             'onecore/x64/Microsoft.VC145.CRT/vcruntime140.dll'):
                (redist / relative).parent.mkdir(parents=True, exist_ok=True)
                (redist / relative).write_bytes(b'')
            find = windows_runtime.find_vc_redistributable
            self.assertEqual(find('MSVCP140.dll', 'x64', redist), redist / 'x64/Microsoft.VC145.CRT/msvcp140.dll')
            self.assertEqual(find('vcomp140.dll', 'x64', redist), redist / 'x64/Microsoft.VC145.OpenMP/vcomp140.dll')
            self.assertEqual(find('libomp140.aarch64.dll', 'arm64', redist),
                             redist / 'arm64/Microsoft.VC145.OpenMP.LLVM/libomp140.aarch64.dll')
            self.assertIsNone(find('vcruntime140.dll', 'x64', redist))
            self.assertIsNone(find('msvcp140.dll', 'arm64', redist))

    def test_build_resolves_runtime_from_the_toolset_not_from_path(self):
        with tempfile.TemporaryDirectory() as temp:
            redist = Path(temp) / 'redist'
            on_path = Path(temp) / 'tools'
            (redist / 'x64/Microsoft.VC145.CRT').mkdir(parents=True)
            (redist / 'x64/Microsoft.VC145.CRT/vcruntime140.dll').write_bytes(b'')
            on_path.mkdir()
            for name in ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll'):
                (on_path / name).write_bytes(b'')
            with mock.patch.dict(os.environ, {'VCToolsRedistDir': str(redist)}):
                resolved = build.resolve_windows_dependency('VCRUNTIME140.dll', [on_path], 'x64')
                self.assertEqual(resolved, redist / 'x64/Microsoft.VC145.CRT/vcruntime140.dll')
                self.assertIsNone(build.resolve_windows_dependency('vcruntime140_1.dll', [on_path], 'x64'))
            with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
                build.resolve_windows_dependency('msvcp140.dll', [on_path], 'x64')

    def test_dependency_search_skips_the_windows_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            windows = Path(temp) / 'Windows'
            system32 = windows / 'System32'
            vendor = Path(temp) / 'vendor'
            system32.mkdir(parents=True)
            vendor.mkdir()
            environment = {'SystemRoot': str(windows), 'PATH': os.pathsep.join([str(system32), str(vendor)])}
            with mock.patch.dict(os.environ, environment, clear=True), \
                    mock.patch.object(build, 'detect_vcpkg_root', return_value=None):
                roots = build.build_windows_dependency_search_roots([vendor / 'llama.dll'])
            self.assertEqual(roots, [vendor])


if __name__ == '__main__':
    unittest.main()
