from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_release_provenance import (  # noqa: E402
    direct_dispatch_input_expressions,
    workflow_run_blocks,
)


WORKFLOW = ROOT / ".github/workflows/native_release.yml"


class ReleaseWorkflowInputTests(unittest.TestCase):
    def test_no_dispatch_expression_is_embedded_in_any_run_block(self) -> None:
        workflow = WORKFLOW.read_text()
        self.assertGreater(len(workflow_run_blocks(workflow)), 20)
        self.assertEqual((), direct_dispatch_input_expressions(workflow))

    def test_detector_covers_inline_and_block_run_forms(self) -> None:
        synthetic = """
jobs:
  inline:
    steps:
      - run: echo '${{ inputs.correlation_id }}'
  block:
    steps:
      - run: |
          echo "${{ github.event.inputs.llama_cpp_tag }}"
          echo "${{ github . event . inputs['native_release_tag'] }}"
"""
        self.assertEqual(
            (
                "${{ inputs.correlation_id }}",
                "${{ github.event.inputs.llama_cpp_tag }}",
                "${{ github . event . inputs['native_release_tag'] }}",
            ),
            direct_dispatch_input_expressions(synthetic),
        )

    def test_adversarial_dispatch_values_are_data_not_shell_source(self) -> None:
        workflow = WORKFLOW.read_text()
        validation_block = next(
            block
            for block in workflow_run_blocks(workflow)
            if "release_contract.py validate-dispatch" in block
        )
        base = {
            "INPUT_LLAMA_CPP_TAG": "v0.2.0",
            "INPUT_LLAMA_CPP_COMMIT": "1" * 40,
            "INPUT_NATIVE_RELEASE_TAG": "v0.2.0",
            "INPUT_SMOKE_POLICY": "required",
            "INPUT_CORRELATION_ID": "test-run-1",
            "INPUT_PUBLISH_RELEASE": "false",
            "GITHUB_REPOSITORY": "leehack/llamadart-native",
        }
        for variable in (
            "INPUT_LLAMA_CPP_TAG",
            "INPUT_LLAMA_CPP_COMMIT",
            "INPUT_NATIVE_RELEASE_TAG",
            "INPUT_SMOKE_POLICY",
            "INPUT_CORRELATION_ID",
            "INPUT_PUBLISH_RELEASE",
        ):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / f"executed-{variable}"
                environment = os.environ.copy()
                environment.update(base)
                environment[variable] = (
                    f'bad"; $(touch {marker}); `touch {marker}`\nsecond-line'
                )
                environment["GITHUB_OUTPUT"] = str(root / "output")
                environment["RUNNER_TEMP"] = str(root)
                result = subprocess.run(
                    ["bash", "-c", validation_block],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertFalse(marker.exists(), result.stderr)


if __name__ == "__main__":
    unittest.main()
