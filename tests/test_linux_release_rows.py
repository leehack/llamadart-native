from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_text import WORKFLOWS, job_block, job_keys, matrix_rows, step_keys, step_name, steps  # noqa: E402

RELEASE = "native_release.yml"
LANE = "validate_linux_release_rows.yml"
# The lane replaces these: it builds checked-out sources and never saves a cache
# the release could restore or that would evict the release's.
RELEASE_ONLY = {"Checkout llama.cpp ref", "Set up ccache", "Upload"}
LANE_ONLY = {"Select candidate upstream", "Restore release ccache", "Prime ccache"}


def workflow_env(workflow: str) -> dict[str, str]:
    block = re.search(r"^env:\n((?:  .*\n)+)", (WORKFLOWS / workflow).read_text(), re.MULTILINE)
    return dict(re.findall(r"^  ([A-Z0-9_]+): (.+)$", block[1], re.MULTILINE))


class ReleaseRowsMirrorTest(unittest.TestCase):
    def assert_same_steps(self, job: str) -> None:
        release = {step_name(step): step for step in steps(job_block(RELEASE, job))}
        lane = {step_name(step): step for step in steps(job_block(LANE, job))}
        self.assertEqual(
            [name for name in release if name not in RELEASE_ONLY],
            [name for name in lane if name not in LANE_ONLY],
        )
        for name, step in lane.items():
            if name in release:
                self.assertEqual(release[name], step, name)

    def test_build_linux_steps_match_the_release_without_its_upload(self) -> None:
        self.assert_same_steps("build-linux")

    def test_build_linux_hip_runs_the_release_steps_and_the_floor_check(self) -> None:
        release = [step_name(step) for step in steps(job_block(RELEASE, "build-linux-hip"))]
        lane = job_block(LANE, "build-linux-hip")
        names = [step_name(step) for step in steps(lane)]
        self.assertEqual(
            [name for name in release if name not in RELEASE_ONLY],
            [name for name in names if name not in LANE_ONLY | {"Check the glibc floor"}],
        )
        by_name = {step_name(step): step for step in steps(job_block(RELEASE, "build-linux-hip"))}
        for step in steps(lane):
            if step_name(step) in by_name:
                self.assertEqual(by_name[step_name(step)], step, step_name(step))
        self.assertLess(names.index("Validate Linux HIP artifacts"), names.index("Check the glibc floor"))
        self.assertLess(names.index("Check the glibc floor"), names.index("Stage upload payload"))

    def test_every_release_row_runs_pinned(self) -> None:
        release = matrix_rows(job_block(RELEASE, "build-linux"))
        lane = matrix_rows(job_block(LANE, "build-linux"))
        self.assertEqual(5, len(release))
        pinned = [{k: v for k, v in row.items() if k != "upstream"} for row in lane if row["upstream"] == "pinned"]
        self.assertEqual(release, pinned)

    def test_candidate_rows_repeat_release_rows(self) -> None:
        release = matrix_rows(job_block(RELEASE, "build-linux"))
        candidates = [row for row in matrix_rows(job_block(LANE, "build-linux")) if row["upstream"] != "pinned"]
        self.assertIn(
            {"arch": "x64", "backend": "cuda", "upstream": "v0.8.0"},
            [{key: row[key] for key in ("arch", "backend", "upstream")} for row in candidates],
        )
        for row in candidates:
            self.assertIn({k: v for k, v in row.items() if k != "upstream"}, release)

    def test_the_lane_uses_the_release_image_container_and_toolkit(self) -> None:
        for job in ("build-linux", "build-linux-hip"):
            for key in ("runs-on", "container"):
                pattern = rf"^    {key}: .+$"
                release = re.search(pattern, job_block(RELEASE, job), re.MULTILINE)
                lane = re.search(pattern, job_block(LANE, job), re.MULTILINE)
                self.assertEqual(release and release[0], lane and lane[0], (job, key))
        release_env = workflow_env(RELEASE)
        lane_env = workflow_env(LANE)
        for key in ("CUDA_REDIST_VERSION", "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER"):
            self.assertEqual(release_env[key], lane_env[key], key)

    def test_the_candidate_is_the_wrapper_lanes_candidate(self) -> None:
        name = "LLAMADART_V080_QUALIFICATION_SHA"
        self.assertEqual(workflow_env("validate_wrapper.yml")[name], workflow_env(LANE)[name])

    def test_steps_are_not_skipped_or_allowed_to_fail(self) -> None:
        allowed_conditions = {
            "Select candidate upstream": "if: matrix.upstream != 'pinned'",
            "Report ccache stats": "if: always()",
            "Validate clean Linux consumer dlopen": "if: ${{ matrix.include_core }}",
            "Audit and select the Linux arm64 CPU variants": "if: ${{ matrix.arch == 'arm64' && matrix.include_core }}",
        }
        for job in ("build-linux", "build-linux-hip"):
            block = job_block(LANE, job)
            self.assertNotIn("if", job_keys(block))
            self.assertNotIn("continue-on-error", job_keys(block))
            for step in steps(block):
                self.assertNotIn("continue-on-error", step_keys(step), step_name(step))
                if "if" in step_keys(step):
                    self.assertIn(allowed_conditions[step_name(step)], step)


class LaneIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (WORKFLOWS / LANE).read_text()

    def test_the_lane_publishes_nothing_and_saves_no_cache(self) -> None:
        for forbidden in ("upload-artifact", "actions/cache@", "setup-ccache", "gh release", "workflow_call", "RELEASE_TOKEN"):
            self.assertNotIn(forbidden, self.text)
        self.assertEqual(2, self.text.count("actions/cache/restore@"))
        self.assertRegex(self.text, r"(?m)^permissions:\n  contents: read\n")

    def test_the_lane_runs_on_a_schedule_and_on_release_input_changes(self) -> None:
        trigger = self.text.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
        self.assertIn("schedule:", trigger)
        self.assertIn("workflow_dispatch:", trigger)
        self.assertIn("pull_request:", trigger)
        for path in (
            ".github/workflows/native_release.yml",
            ".github/workflows/validate_linux_release_rows.yml",
            "tools/build.py",
            "tools/apt_retry.sh",
            "tools/install_cuda_redist.py",
            "tools/check_glibc_floor.py",
            "third_party/llama.cpp",
        ):
            self.assertIn(f"- {path}\n", trigger)

    def test_the_release_workflow_stays_dispatch_only(self) -> None:
        release = (WORKFLOWS / RELEASE).read_text()
        trigger = release.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
        self.assertNotIn("schedule:", trigger)
        self.assertNotIn("pull_request:", trigger)


if __name__ == "__main__":
    unittest.main()
