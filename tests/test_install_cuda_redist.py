from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import install_cuda_redist


class CudaRedistComponentTests(unittest.TestCase):
    def test_nvml_headers_are_installed_and_required(self):
        # Upstream ggml-cuda.cu includes <nvml.h> in every NVIDIA build.
        self.assertIn('cuda_nvml_dev', install_cuda_redist.DEFAULT_COMPONENTS)
        self.assertIn('cuda_nvml_dev', install_cuda_redist.REQUIRED_COMPONENTS)

    def test_required_components_are_installed_by_default(self):
        self.assertLessEqual(install_cuda_redist.REQUIRED_COMPONENTS,
                             set(install_cuda_redist.DEFAULT_COMPONENTS))


if __name__ == '__main__':
    unittest.main()
