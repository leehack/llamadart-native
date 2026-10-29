from __future__ import annotations

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
PINNED_RUNNER = "ubuntu-24.04"
PINNED_JOBS = {
    "native_release.yml": ("build-linux", "build-android"),
    "validate_wrapper.yml": (
        "linux-artifact-contract",
        "kleidiai-dispatch-emulated",
        "android-vulkan-shaders",
    ),
}


def job_runner(workflow: str, job: str) -> str:
    text = (WORKFLOWS / workflow).read_text()
    block = re.search(rf"^  {re.escape(job)}:\n((?:(?:    .*)?\n)+)", text, re.MULTILINE)
    if not block:
        raise AssertionError(f"{workflow} has no job {job}")
    runner = re.search(r"^    runs-on: (.+)$", block[1], re.MULTILINE)
    if not runner:
        raise AssertionError(f"{workflow} job {job} has no runs-on")
    return runner[1].strip()


class RunnerPinTest(unittest.TestCase):
    def test_jobs_that_depend_on_the_ubuntu_24_04_image_stay_pinned(self) -> None:
        for workflow, jobs in PINNED_JOBS.items():
            for job in jobs:
                with self.subTest(workflow=workflow, job=job):
                    self.assertEqual(PINNED_RUNNER, job_runner(workflow, job))

    def test_job_runner_reads_the_named_job_only(self) -> None:
        self.assertEqual("ubuntu-latest", job_runner("validate_wrapper.yml", "android-arm64-isa"))
        with self.assertRaises(AssertionError):
            job_runner("validate_wrapper.yml", "no-such-job")


if __name__ == "__main__":
    unittest.main()
