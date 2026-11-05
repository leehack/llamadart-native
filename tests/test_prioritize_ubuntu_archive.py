from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prioritize_ubuntu_archive as mirror
from workflow_text import job_block, step_running


class MirrorPriorityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "apt-mirrors.txt"
        self.path.write_bytes(mirror.ORIGINAL)

    def test_swaps_only_priorities_and_preserves_file_metadata(self):
        self.path.chmod(0o640)
        before = self.path.stat()
        mirror.prioritize_archive(self.path)
        expected = (
            b"http://azure.archive.ubuntu.com/ubuntu/\tpriority:2\n"
            b"https://archive.ubuntu.com/ubuntu/\tpriority:1\n"
            b"https://security.ubuntu.com/ubuntu/\tpriority:3\n"
        )
        self.assertEqual(expected, self.path.read_bytes())
        after = self.path.stat()
        for field in ("st_ino", "st_uid", "st_gid", "st_mode", "st_size"):
            self.assertEqual(getattr(before, field), getattr(after, field), field)

    def test_already_preferred_is_not_written_again(self):
        self.path.write_bytes(mirror.PREFERRED)
        os.utime(self.path, ns=(1_000_000_000, 1_000_000_000))
        before = self.path.stat()
        mirror.prioritize_archive(self.path)
        self.assertEqual(mirror.PREFERRED, self.path.read_bytes())
        self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_unknown_or_duplicate_layout_is_rejected_without_writing(self):
        fixtures = (
            b"",
            mirror.ORIGINAL + mirror.ORIGINAL.splitlines(keepends=True)[0],
            mirror.ORIGINAL.replace(b"priority:2", b"priority:1"),
            mirror.ORIGINAL.replace(b"https://archive.", b"http://archive."),
            mirror.ORIGINAL.replace(b"\t", b" "),
            mirror.ORIGINAL.rstrip(b"\n"),
            mirror.ORIGINAL.replace(b"https://archive.", b"https://user:secret@archive."),
        )
        for content in fixtures:
            with self.subTest(content=content):
                self.path.write_bytes(content)
                os.utime(self.path, ns=(1_000_000_000, 1_000_000_000))
                before = self.path.stat()
                with self.assertRaisesRegex(ValueError, "Unexpected APT mirrorlist"):
                    mirror.prioritize_archive(self.path)
                self.assertEqual(content, self.path.read_bytes())
                self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_symlink_is_rejected_without_touching_target(self):
        link = self.path.with_name("link")
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            mirror.prioritize_archive(link)
        self.assertEqual(mirror.ORIGINAL, self.path.read_bytes())

    def test_fifo_is_rejected_without_blocking(self):
        fifo = self.path.with_name("fifo")
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular file"):
            mirror.prioritize_archive(fifo)

    def test_cli_reports_unknown_layout_without_echoing_contents(self):
        self.path.write_bytes(b"unknown private content")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/prioritize_ubuntu_archive.py"), str(self.path)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(1, result.returncode)
        self.assertIn("Unexpected APT mirrorlist", result.stderr)
        self.assertNotIn("private content", result.stderr)
        self.assertEqual(b"unknown private content", self.path.read_bytes())

    def test_workflow_runs_helper_before_only_affected_dependency_acquisitions(self):
        command = "sudo python3 tools/prioritize_ubuntu_archive.py"
        workflow = (ROOT / ".github/workflows/validate_wrapper.yml").read_text()
        self.assertEqual(2, workflow.count(command))
        for job in ("linux-artifact-contract", "kleidiai-dispatch-emulated"):
            block = job_block("validate_wrapper.yml", job)
            step = step_running(block, command)
            self.assertLess(step.index(command), step.index("sudo apt-get update"))
            if job == "linux-artifact-contract":
                self.assertIn('if [ "${{ matrix.arch }}" = "arm64" ]; then\n            ' + command + '\n          fi', step)


if __name__ == "__main__":
    unittest.main()
