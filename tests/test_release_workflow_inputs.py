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
    metal_floor_contract_errors,
    workflow_run_blocks,
)


WORKFLOW = ROOT / ".github/workflows/native_release.yml"
METAL_FLOOR = ROOT / ".github/workflows/metal_os_floor.yml"


class MetalFloorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text()
        self.floor = METAL_FLOOR.read_text()

    def assert_refused(self, workflow: str, floor: str, message: str) -> None:
        self.assertNotEqual(self.workflow + self.floor, workflow + floor, "the mutation applied")
        errors = metal_floor_contract_errors(workflow, floor)
        self.assertTrue(any(message in error for error in errors), errors)

    def test_the_workflows_meet_the_contract(self) -> None:
        self.assertEqual([], metal_floor_contract_errors(self.workflow, self.floor))

    def test_the_release_must_check_the_apple_markers(self) -> None:
        self.assert_refused(
            self.workflow.replace("tools/llama_cpp_patches.py check-apple", "tools/other.py"),
            self.floor, "check-apple")

    def test_the_release_must_use_the_floor_unconditionally(self) -> None:
        gate = "    uses: ./.github/workflows/metal_os_floor.yml\n"
        self.assert_refused(self.workflow.replace(gate, ""), self.floor, "metal_os_floor.yml")
        self.assert_refused(self.workflow.replace(gate, "    if: false\n" + gate),
                            self.floor, "metal-os-floor")

    def test_every_floor_job_must_run_with_bf16(self) -> None:
        for job in ("macos", "ios-simulator"):
            start = self.floor.index(f"\n  {job}:\n")
            end = self.floor.find("\n  ios-simulator:\n", start + 1)
            end = len(self.floor) if end < 0 else end
            mutated = self.floor[:start] + self.floor[start:end].replace(" --bf16", "") + (
                self.floor[end:])
            self.assert_refused(self.workflow, mutated, f"{job} must run --bf16")

    def test_the_negative_control_does_not_count_as_a_bf16_run(self) -> None:
        run = './metal_floor_check "$PWD/artifact/libllamadart.dylib" --bf16\n          if GGML'
        self.assert_refused(self.workflow, self.floor.replace(run, "if GGML"),
                            "macos must run --bf16")

    def test_a_floor_step_cannot_be_allowed_to_fail(self) -> None:
        step = "      - name: Load the Metal backend with this OS's Metal compiler\n"
        self.assert_refused(self.workflow,
                            self.floor.replace(step, step + "        continue-on-error: true\n"),
                            "continue-on-error")

    def test_a_floor_job_cannot_be_turned_off(self) -> None:
        guard = "    if: ${{ !cancelled() && (needs.build.result == 'success' || needs.build.result == 'skipped') }}\n"
        for off in ("    if: false\n", "    if: ${{ false }}\n"):
            self.assert_refused(self.workflow, self.floor.replace(guard, off, 1), "turned off")

    def test_the_floor_must_keep_its_durable_runners(self) -> None:
        self.assert_refused(self.workflow, self.floor.replace("[macos-15, macos-26]", "[macos-26]"),
                            "macos-15")
        self.assert_refused(self.workflow, self.floor.replace("[macos-15, macos-26]", "[macos-15]"),
                            "macos-26")
        self.assert_refused(self.workflow,
                            self.floor.replace("runs-on: macos-15\n", "runs-on: macos-26\n"),
                            "ios-simulator must run on macos-15")


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
