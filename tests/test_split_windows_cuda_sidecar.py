from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
import split_windows_cuda_sidecar
from test_windows_runtime import CRT, complete_bundle, make_pe

SIDECAR = ['cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll', 'ggml-cuda.dll']


def add_cuda(directory, *, cuda_imports=None):
    make_pe(directory / 'cudart64_12.dll', exports=['cudaMalloc'], imports={'KERNEL32.dll': ['GetLastError']})
    make_pe(directory / 'cublasLt64_12.dll', exports=['cublasLtCreate'], imports={'KERNEL32.dll': ['GetLastError']})
    make_pe(directory / 'cublas64_12.dll', exports=['cublasCreate_v2'],
            imports={'cublasLt64_12.dll': ['cublasLtCreate']})
    make_pe(directory / 'ggml-cuda.dll', imports={
        'ggml-base.dll': ['ggml_init'], 'cudart64_12.dll': ['cudaMalloc'], 'cublas64_12.dll': ['cublasCreate_v2'],
        'nvcuda.dll': ['cuInit'], **CRT, **(cuda_imports or {}),
    })


class SplitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.bundle = Path(temp.name) / 'windows-x64'
        self.sidecar = Path(temp.name) / 'windows-x64-cuda12'
        self.bundle.mkdir()
        complete_bundle(self.bundle)

    def listing(self, directory):
        return sorted(path.name for path in directory.glob('*.dll')) if directory.exists() else []

    def test_moves_the_cuda_backend_and_only_its_dependencies(self):
        add_cuda(self.bundle)
        self.assertEqual(split_windows_cuda_sidecar.split(self.bundle, self.sidecar), [])
        self.assertEqual(self.listing(self.sidecar), SIDECAR)
        self.assertEqual(self.listing(self.bundle),
                         ['ggml-base.dll', 'llama.dll', 'msvcp140.dll', 'vcomp140.dll', 'vcruntime140.dll'])

    def test_incomplete_merged_bundle_is_left_untouched(self):
        add_cuda(self.bundle)
        (self.bundle / 'cublasLt64_12.dll').unlink()
        errors = split_windows_cuda_sidecar.split(self.bundle, self.sidecar)
        self.assertEqual(errors, ['merged bundle: cublas64_12.dll: imports cublaslt64_12.dll, '
                                  'which neither the bundle nor Windows provides'])
        self.assertEqual(self.listing(self.sidecar), [])

    def test_unexpected_cuda_dependency_fails(self):
        add_cuda(self.bundle, cuda_imports={'nvrtc64_120_0.dll': ['nvrtcVersion']})
        make_pe(self.bundle / 'nvrtc64_120_0.dll', exports=['nvrtcVersion'])
        errors = split_windows_cuda_sidecar.split(self.bundle, self.sidecar)
        self.assertEqual(len(errors), 1)
        self.assertIn("'nvrtc64_120_0.dll'", errors[0])
        self.assertEqual(self.listing(self.sidecar), [])
        self.assertIn('ggml-cuda.dll', self.listing(self.bundle))

    def test_runtime_a_core_library_imports_stays_out_of_the_sidecar_and_fails(self):
        add_cuda(self.bundle)
        make_pe(self.bundle / 'llama.dll', imports={'ggml-base.dll': ['ggml_init'], 'cudart64_12.dll': ['cudaMalloc'],
                                                     **CRT})
        errors = split_windows_cuda_sidecar.split(self.bundle, self.sidecar)
        self.assertEqual(len(errors), 1)
        self.assertNotIn("'cudart64_12.dll'", errors[0].split(', expected')[0])
        self.assertEqual(self.listing(self.sidecar), [])

    def test_bundle_without_cuda_fails(self):
        with self.assertRaisesRegex(ValueError, 'ggml-cuda.dll is missing'):
            split_windows_cuda_sidecar.split(self.bundle, self.sidecar)

    def test_non_empty_sidecar_directory_fails(self):
        add_cuda(self.bundle)
        self.sidecar.mkdir()
        (self.sidecar / 'stale.dll').write_bytes(b'')
        self.assertEqual(split_windows_cuda_sidecar.split(self.bundle, self.sidecar),
                         [f'{self.sidecar} is not empty'])
        self.assertIn('ggml-cuda.dll', self.listing(self.bundle))


if __name__ == '__main__':
    unittest.main()
