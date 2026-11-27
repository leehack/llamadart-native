from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build  # noqa: E402
from workflow_text import job_block, step_keys, step_running


class AndroidVulkanGlslcTest(unittest.TestCase):
    def test_override_replaces_ndk_glslc_and_missing_override_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ndk = Path(temp) / "ndk"
            ndk_glslc = ndk / "shader-tools/linux-x86_64/glslc"
            ndk_glslc.parent.mkdir(parents=True)
            ndk_glslc.write_text("")
            host_glslc = Path(temp) / "glslc"
            host_glslc.write_text("")

            self.assertEqual(ndk_glslc, build.android_vulkan_glslc(ndk, {}))
            self.assertEqual(
                host_glslc,
                build.android_vulkan_glslc(ndk, {"ANDROID_VULKAN_GLSLC": str(host_glslc)}),
            )
            with self.assertRaises(SystemExit):
                build.android_vulkan_glslc(ndk, {"ANDROID_VULKAN_GLSLC": str(Path(temp) / "missing")})

    def test_release_android_vulkan_lanes_use_host_glslc(self) -> None:
        workflow = (ROOT / ".github/workflows/native_release.yml").read_text(encoding="utf-8")
        android_job = workflow.split("\n  build-android:\n", 1)[1].split("\n  build-apple:\n", 1)[0]
        self.assertIn('if [ "${{ matrix.backend }}" = "vulkan" ]; then', android_job)
        self.assertIn("sudo apt-get install -y glslc", android_job)
        self.assertIn('echo "ANDROID_VULKAN_GLSLC=$(command -v glslc)" >> "$GITHUB_ENV"', android_job)

    def test_android_vulkan_builds_reject_integer_dot_support(self) -> None:
        log = "configure-build.log"
        grep = f"grep -Fx -- '-- GL_EXT_integer_dot_product not supported by glslc' {log}"

        validation_step = step_running(
            job_block("validate_wrapper.yml", "android-vulkan-shaders"),
            "Build the release Android Vulkan backend",
        )
        self.assertNotIn("if", step_keys(validation_step))
        self.assertIn(f"          python3 - <<'PY' 2>&1 | tee {log}", validation_step.splitlines())
        self.assertIn(f"          {grep}", validation_step.splitlines())

        release_step = step_running(
            job_block("native_release.yml", "build-android"),
            "tools/build.py android --abi",
        )
        self.assertNotIn("if", step_keys(release_step))
        self.assertIn(
            "          python3 tools/build.py android --abi ${{ matrix.android_abi }}"
            " --backend ${{ matrix.backend }}"
            f" 2>&1 | tee {log}\n"
            '          if [ "${{ matrix.backend }}" = "vulkan" ]; then\n'
            f"            {grep}\n"
            "          fi\n",
            release_step,
        )


if __name__ == "__main__":
    unittest.main()
