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
          echo "${{ github['event']['inputs']['smoke_policy'] }}"
          echo '${{ github["event"].inputs["correlation_id"] }}'
          echo "${{ github.event['inputs'].publish_release }}"
          echo "${{ toJSON(github.event.inputs) }}"
          echo "${{ toJSON(github['event']['inputs']) }}"
          echo "${{ github.event.inputs[format('llama_cpp_tag')] }}"
          echo "${{ github[format('event')]['inputs'][format('llama_cpp_tag')] }}"
"""
        self.assertEqual(
            (
                "${{ inputs.correlation_id }}",
                "${{ github.event.inputs.llama_cpp_tag }}",
                "${{ github . event . inputs['native_release_tag'] }}",
                "${{ github['event']['inputs']['smoke_policy'] }}",
                '${{ github["event"].inputs["correlation_id"] }}',
                "${{ github.event['inputs'].publish_release }}",
                "${{ toJSON(github.event.inputs) }}",
                "${{ toJSON(github['event']['inputs']) }}",
                "${{ github.event.inputs[format('llama_cpp_tag')] }}",
                "${{ github[format('event')]['inputs'][format('llama_cpp_tag')] }}",
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
            "INPUT_NATIVE_SOURCE_SHA": "2" * 40,
            "INPUT_NATIVE_RELEASE_TAG": "v0.2.0",
            "INPUT_SMOKE_POLICY": "required",
            "INPUT_CORRELATION_ID": "test-run-1",
            "INPUT_PUBLISH_RELEASE": "false",
            "APPROVAL_EVENT_NAME": "workflow_dispatch",
            "APPROVAL_ACTOR": "github-actions[bot]",
            "APPROVAL_TRIGGERING_ACTOR": "github-actions[bot]",
            "APPROVAL_REPOSITORY_OWNER": "leehack",
            "APPROVAL_RUN_ATTEMPT": "1",
            "APPROVAL_WORKFLOW_REF": "refs/heads/main",
            "APPROVAL_WORKFLOW_SHA": "2" * 40,
            "GITHUB_REPOSITORY": "leehack/llamadart-native",
        }
        for variable in (
            "INPUT_LLAMA_CPP_TAG",
            "INPUT_LLAMA_CPP_COMMIT",
            "INPUT_NATIVE_SOURCE_SHA",
            "INPUT_NATIVE_RELEASE_TAG",
            "INPUT_SMOKE_POLICY",
            "INPUT_CORRELATION_ID",
            "INPUT_PUBLISH_RELEASE",
            "APPROVAL_EVENT_NAME",
            "APPROVAL_ACTOR",
            "APPROVAL_TRIGGERING_ACTOR",
            "APPROVAL_REPOSITORY_OWNER",
            "APPROVAL_RUN_ATTEMPT",
            "APPROVAL_WORKFLOW_REF",
            "APPROVAL_WORKFLOW_SHA",
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
                # Preparation intentionally ignores actor/default-branch-only
                # publication fields after safely transporting them as quoted
                # arguments. They may therefore validate successfully here;
                # the injection invariant is that none becomes shell source.
                if variable not in {
                    "APPROVAL_ACTOR",
                    "APPROVAL_TRIGGERING_ACTOR",
                    "APPROVAL_REPOSITORY_OWNER",
                    "APPROVAL_RUN_ATTEMPT",
                    "APPROVAL_WORKFLOW_REF",
                }:
                    self.assertNotEqual(0, result.returncode)
                self.assertFalse(marker.exists(), result.stderr)

    def test_default_branch_lookup_url_encodes_ref_names(self) -> None:
        workflow = WORKFLOW.read_text()
        validation_block = next(
            block
            for block in workflow_run_blocks(workflow)
            if "release_contract.py validate-dispatch" in block
        )
        self.assertIn("urllib.parse.quote", validation_block)
        self.assertIn(
            'commits/$(api_path_segment "$default_branch")',
            validation_block,
        )


if __name__ == "__main__":
    unittest.main()
