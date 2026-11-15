from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import textwrap
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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

    def run_main(self, path=None, arguments=()):
        stderr = io.StringIO()
        real_prioritize = mirror.prioritize_archive
        target = self.path if path is None else path

        def only_fixture_path(actual):
            # Fail before opening anything if the CLI ignores its fixed path.
            self.assertEqual(target, actual)
            return real_prioritize(actual)

        with mock.patch.object(mirror, "MIRRORLIST_PATH", target, create=True), \
                mock.patch.object(mirror, "prioritize_archive", side_effect=only_fixture_path) as call, \
                mock.patch.object(sys, "argv", ["prioritize_ubuntu_archive.py", *arguments]), \
                contextlib.redirect_stderr(stderr):
            code = mirror.main()
        return code, stderr.getvalue(), call

    def receipt(self, output):
        lines = output.splitlines()
        self.assertEqual(2, len(lines))
        self.assertTrue(lines[1].startswith("APT mirrorlist diagnostic: "))
        return json.loads(lines[1].removeprefix("APT mirrorlist diagnostic: "))

    def test_failure_receipt_reports_known_structure_without_accepting_it(self):
        # Deliberately noncanonical structural probes, NOT failed-image fixtures.
        content = (
            b" https://security.ubuntu.com/ubuntu/ priority:3 \r\n"
            b"http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n"
            b"https://archive.ubuntu.com/ubuntu/  priority:2"
        )
        self.path.write_bytes(content)
        before = self.path.stat()
        code, output, _ = self.run_main()
        self.assertEqual(1, code)
        data = self.receipt(output)
        self.assertEqual(len(content), data["byte_count"])
        self.assertEqual(3, data["line_count"])
        self.assertEqual({"lf": 1, "crlf": 1, "bare_cr": 0}, data["newlines"])
        self.assertFalse(data["trailing_lf"])
        self.assertEqual([
            {"mirror_id": "ubuntu_security", "known_url_id": "ubuntu_security", "priority": 3, "leading_ws": "s",
             "separator_ws": "s", "trailing_ws": "s", "ending": "crlf"},
            {"mirror_id": "azure_archive", "known_url_id": "azure_archive", "priority": 1, "leading_ws": "",
             "separator_ws": "t", "trailing_ws": "", "ending": "lf"},
            {"mirror_id": "ubuntu_archive", "known_url_id": "ubuntu_archive", "priority": 2, "leading_ws": "",
             "separator_ws": "ss", "trailing_ws": "", "ending": "none"},
        ], data["lines"])
        self.assertNotIn("http", output)
        self.assertEqual(content, self.path.read_bytes())
        self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_empty_duplicate_and_missing_newline_still_refuse(self):
        for content, count in ((b"", 0), (mirror.ORIGINAL * 2, 6),
                               (mirror.ORIGINAL.rstrip(b"\n"), 3)):
            with self.subTest(count=count):
                self.path.write_bytes(content)
                code, output, _ = self.run_main()
                self.assertEqual(1, code)
                self.assertEqual(count, self.receipt(output)["line_count"])
                self.assertEqual(content, self.path.read_bytes())

    def test_unknown_malformed_and_secret_bearing_rows_stay_unclassified(self):
        fixtures = (
            b"https://user:PRIVATE_TOKEN@archive.ubuntu.com/ubuntu/\tpriority:2\n",
            b"https://archive.ubuntu.com/ubuntu/?token=PRIVATE_TOKEN\tpriority:2\n",
            b"https://archive.ubuntu.com.evil/ubuntu/\tpriority:2\n",
            b"http://archive.ubuntu.com/ubuntu/\tpriority:2\n",
            b"https://private.invalid/PRIVATE_TOKEN\tpriority:2\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:2 PRIVATE_TOKEN\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:0\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:1000\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:02\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:-1\n",
            b"https://archive.ubuntu.com/ubuntu/\tpriority:2\rPRIVATE_TOKEN\n",
            b"PRIVATE_TOKEN\x1b[31m\n",
            b"\xff\x00PRIVATE_TOKEN\n",
        )
        for content in fixtures:
            with self.subTest(content=content):
                self.path.write_bytes(content)
                before = self.path.stat()
                code, output, _ = self.run_main()
                self.assertEqual(1, code)
                expected_url_id = ("ubuntu_archive" if content.startswith(
                    b"https://archive.ubuntu.com/ubuntu/\t") else "unclassified")
                self.assertEqual([{"mirror_id": "unclassified", "known_url_id": expected_url_id}],
                                 self.receipt(output)["lines"])
                for forbidden in ("PRIVATE_TOKEN", "http", "private.invalid", "\x1b"):
                    self.assertNotIn(forbidden, output)
                self.assertEqual(content, self.path.read_bytes())
                self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def assert_token_receipt(self, content, expected_url_id):
        self.path.write_bytes(content)
        before = self.path.stat()
        with mock.patch.object(mirror.os, "write", side_effect=AssertionError("unexpected write")):
            code, output, _ = self.run_main()
        self.assertEqual(1, code)
        self.assertEqual([{"mirror_id": "unclassified", "known_url_id": expected_url_id}],
                         self.receipt(output)["lines"])
        self.assertTrue(output.isascii())
        for forbidden in ("PRIVATE_TOKEN", "http", "priority", "\x1b", "\x00"):
            self.assertNotIn(forbidden, output)
        self.assertEqual(content, self.path.read_bytes())
        self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_exact_known_url_with_unknown_metadata_is_identified_but_refused(self):
        # Public literals are classification probes, not observed image input.
        for url, expected in (
            (b"http://azure.archive.ubuntu.com/ubuntu/", "azure_archive"),
            (b"https://archive.ubuntu.com/ubuntu/", "ubuntu_archive"),
            (b"https://security.ubuntu.com/ubuntu/", "ubuntu_security"),
            (b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/", "ubuntu_ec2_us_east_1"),
        ):
            for metadata in (b"unrecognized", b"priority:2 PRIVATE_TOKEN",
                             b"PRIVATE_TOKEN?credential=PRIVATE_TOKEN\x1b\x00",
                             b"priority:0", b"priority:1000"):
                with self.subTest(url_id=expected, metadata=metadata):
                    self.assert_token_receipt(b" \t" + url + b"\t" + metadata + b"\n", expected)

    def test_only_complete_exact_url_tokens_are_identified(self):
        fixtures = (
            b"prefixhttps://archive.ubuntu.com/ubuntu/",
            b"https://archive.ubuntu.com/ubuntu/suffix",
            b"https://user:PRIVATE_TOKEN@archive.ubuntu.com/ubuntu/",
            b"https://archive.ubuntu.com/ubuntu/?PRIVATE_TOKEN",
            b"https://archive.ubuntu.com/ubuntu/#PRIVATE_TOKEN",
            b"HTTPS://archive.ubuntu.com/ubuntu/",
            b"https://ARCHIVE.ubuntu.com/ubuntu/",
            b"https://archive.ubuntu.com/ubuntu/\r",
            b"https://archive.ubuntu.com/ubuntu/\x00",
            b"https://arch\x00ive.ubuntu.com/ubuntu/",
            b"https://arch\x1bive.ubuntu.com/ubuntu/",
            b"https://archive.ubuntu.com/ubuntu/\x1b",
            b"https://archive.ubuntu.com/ubuntu/\v",
            b"https://archive.ubuntu.com/ubuntu/\f",
            b"\vhttps://archive.ubuntu.com/ubuntu/",
            b"https://archive.ubuntu.com.evil/ubuntu/",
            b"https://private.invalid/PRIVATE_TOKEN",
            b"PRIVATE_TOKEN",
        )
        for token in fixtures:
            with self.subTest(token=token):
                self.assert_token_receipt(token + b"\tPRIVATE_TOKEN\n", "unclassified")
        for leading, expected in ((b" " * 32, "ubuntu_archive"),
                                  (b" " * 33, "unclassified")):
            with self.subTest(leading=len(leading)):
                self.assert_token_receipt(leading + b"https://archive.ubuntu.com/ubuntu/"
                                          b"\tPRIVATE_TOKEN\n", expected)
        # A line ending/control is part of a token unless SP/TAB delimits it.
        self.assert_token_receipt(b"https://archive.ubuntu.com/ubuntu/\n", "unclassified")

    def test_ec2_candidate_full_row_is_classified_but_never_admitted(self):
        # Hypothesis probes from public template + observed cloud/region;
        # these are NOT captured failed-image mirrorlist bytes.
        row = b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/\tpriority:1\n"
        fixtures = (
            (row, {"priority": 1, "leading_ws": "", "separator_ws": "t",
                   "trailing_ws": "", "ending": "lf"}),
            (b" \t" + row.replace(b"\tpriority:1\n", b"  priority:999 \r\n"),
             {"priority": 999, "leading_ws": "st", "separator_ws": "ss",
              "trailing_ws": "s", "ending": "crlf"}),
            (mirror.ORIGINAL.replace(mirror.ORIGINAL.splitlines(keepends=True)[0], row),
             {"priority": 1, "leading_ws": "", "separator_ws": "t",
              "trailing_ws": "", "ending": "lf"}),
        )
        for content, fields in fixtures:
            with self.subTest(fields=fields):
                self.path.write_bytes(content)
                before = self.path.stat()
                with mock.patch.object(mirror.os, "write", side_effect=AssertionError("write")):
                    code, output, _ = self.run_main()
                self.assertEqual(1, code)
                data = self.receipt(output)
                self.assertEqual({"mirror_id": "ubuntu_ec2_us_east_1",
                                  "known_url_id": "ubuntu_ec2_us_east_1", **fields},
                                 data["lines"][0])
                self.assertNotIn("http", output)
                self.assertEqual(content, self.path.read_bytes())
                self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_ec2_candidate_url_variants_stay_unknown(self):
        for token in (
            b"prefixhttp://us-east-1.ec2.archive.ubuntu.com/ubuntu/",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/suffix",
            b"http://us-east-1.ec2.archive.ubuntu.com.evil/ubuntu/",
            b"http://us-east-1.ec2.archive.ubuntu.com@evil/ubuntu/",
            b"http://us-east-2.ec2.archive.ubuntu.com/ubuntu/",
            b"http://eu-west-1.ec2.archive.ubuntu.com/ubuntu/",
            b"https://us-east-1.ec2.archive.ubuntu.com/ubuntu/",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu//",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/%2f",
            b"http://PRIVATE_TOKEN@us-east-1.ec2.archive.ubuntu.com/ubuntu/",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/?PRIVATE_TOKEN",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/#PRIVATE_TOKEN",
            b"HTTP://us-east-1.ec2.archive.ubuntu.com/ubuntu/",
            b"http://US-EAST-1.ec2.archive.ubuntu.com/ubuntu/",
            b"http://us-east-1.ec2.arch\x00ive.ubuntu.com/ubuntu/",
        ):
            with self.subTest(token=token):
                self.assert_token_receipt(token + b"\tpriority:1 PRIVATE_TOKEN\n", "unclassified")

    def test_bounded_priority_is_diagnostic_only(self):
        self.path.write_bytes(b"https://archive.ubuntu.com/ubuntu/\tpriority:999\n")
        code, output, _ = self.run_main()
        self.assertEqual(1, code)
        row = self.receipt(output)["lines"][0]
        self.assertEqual(("ubuntu_archive", 999), (row["mirror_id"], row["priority"]))

    def test_unsafe_or_oversize_inputs_provide_no_receipt(self):
        link = self.path.with_name("PRIVATE_LINK")
        link.symlink_to(self.path)
        fifo = self.path.with_name("PRIVATE_FIFO")
        os.mkfifo(fifo)
        directory = self.path.with_name("PRIVATE_DIRECTORY")
        directory.mkdir()
        missing = self.path.with_name("PRIVATE_MISSING")
        outputs = []
        for path in (link, fifo, directory, missing):
            with mock.patch.object(mirror.os, "read", side_effect=AssertionError("unsafe read")):
                code, output, _ = self.run_main(path)
            self.assertEqual(1, code)
            outputs.append(output)
        for content in (b"PRIVATE_TOKEN" * 400, b"\n" * 17, b"x" * 257):
            self.path.write_bytes(content)
            code, output, _ = self.run_main()
            self.assertEqual(1, code)
            self.assertEqual(content, self.path.read_bytes())
            outputs.append(output)
        self.assertEqual(1, len(set(outputs)))
        self.assertEqual(1, len(outputs[0].splitlines()))
        for output in outputs:
            self.assertNotIn("diagnostic:", output)
            self.assertNotIn("PRIVATE", output)

    def test_oversize_file_is_rejected_before_content_read(self):
        self.path.write_bytes(b"x" * 4097)
        with mock.patch.object(mirror.os, "read", side_effect=AssertionError("oversize read")):
            code, output, _ = self.run_main()
        self.assertEqual(1, code)
        self.assertNotIn("diagnostic:", output)

    def test_read_failure_does_not_echo_exception_or_write(self):
        self.path.write_bytes(b"unknown")
        before = self.path.stat()
        with mock.patch.object(mirror.os, "read", side_effect=OSError("PRIVATE_TOKEN https://private.invalid")), \
                mock.patch.object(mirror.os, "write", side_effect=AssertionError("unexpected write")):
            code, output, _ = self.run_main()
        self.assertEqual(1, code)
        self.assertNotIn("PRIVATE", output)
        self.assertNotIn("http", output)
        self.assertNotIn("diagnostic:", output)
        self.assertEqual(b"unknown", self.path.read_bytes())
        self.assertEqual(before.st_mtime_ns, self.path.stat().st_mtime_ns)

    def test_read_request_is_bounded_and_partial_or_growing_input_is_not_reported(self):
        self.path.write_bytes(b"unknown")
        real_read = mirror.os.read
        with mock.patch.object(mirror.os, "read", wraps=real_read) as read:
            code, _, _ = self.run_main()
        self.assertEqual(1, code)
        self.assertEqual(1, read.call_count)
        self.assertLessEqual(read.call_args.args[1], 4097)
        for incomplete in (b"x", b"x" * 4097):
            with mock.patch.object(mirror.os, "read", return_value=incomplete), \
                    mock.patch.object(mirror.os, "write", side_effect=AssertionError("write")):
                code, output, _ = self.run_main()
            self.assertEqual(1, code)
            self.assertNotIn("diagnostic:", output)
        self.assertEqual(b"unknown", self.path.read_bytes())

    def test_file_change_during_read_provides_no_receipt_or_priority_write(self):
        self.path.write_bytes(b"unknown")
        real_read = mirror.os.read

        def grow_after_read(descriptor, limit):
            snapshot = real_read(descriptor, limit)
            self.path.write_bytes(b"PRIVATE_TOKEN" * 400)
            return snapshot

        with mock.patch.object(mirror.os, "read", side_effect=grow_after_read), \
                mock.patch.object(mirror.os, "write", side_effect=AssertionError("priority write")):
            code, output, _ = self.run_main()
        self.assertEqual(1, code)
        self.assertNotIn("diagnostic:", output)
        self.assertNotIn("PRIVATE", output)
        self.assertEqual(b"PRIVATE_TOKEN" * 400, self.path.read_bytes())

    def test_maximum_recognized_structure_has_bounded_output(self):
        line = (b" " * 32 + b"https://archive.ubuntu.com/ubuntu/" + b"\t" * 32 +
                b"priority:999" + b" " * 32 + b"\r\n")
        self.path.write_bytes(line * 16)
        code, output, _ = self.run_main()
        self.assertEqual(1, code)
        data = self.receipt(output)
        self.assertEqual(16, len(data["lines"]))
        self.assertTrue(all(row["mirror_id"] == "ubuntu_archive" for row in data["lines"]))
        self.assertLess(len(output.encode("ascii")), 8400)
        self.assertEqual(line * 16, self.path.read_bytes())

    def test_cli_never_prints_an_os_error_message(self):
        stderr = io.StringIO()
        with mock.patch.object(mirror, "prioritize_archive",
                               side_effect=OSError("PRIVATE_TOKEN https://private.invalid")), \
                mock.patch.object(sys, "argv", ["prioritize_ubuntu_archive.py"]), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(1, mirror.main())
        self.assertNotIn("PRIVATE", stderr.getvalue())
        self.assertNotIn("http", stderr.getvalue())
        self.assertNotIn("diagnostic:", stderr.getvalue())

    def test_cli_cannot_select_an_arbitrary_path(self):
        code, output, call = self.run_main(arguments=("PRIVATE_TOKEN",))
        self.assertEqual(1, code)
        call.assert_not_called()
        self.assertNotIn("PRIVATE", output)
        self.assertNotIn("diagnostic:", output)

    def test_success_has_no_diagnostic(self):
        for content in (mirror.ORIGINAL, mirror.PREFERRED):
            self.path.write_bytes(content)
            code, output, _ = self.run_main()
            self.assertEqual(0, code)
            self.assertEqual("", output)
            self.assertEqual(mirror.PREFERRED, self.path.read_bytes())

    def test_workflow_refusal_stops_before_any_apt_command(self):
        fixtures = (
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/\tpriority:1\n",
            b"http://us-east-1.ec2.archive.ubuntu.com/ubuntu/\tPRIVATE_TOKEN\n",
            b"https://archive.ubuntu.com/ubuntu/\tPRIVATE_TOKEN\n",
            b"https://user:PRIVATE_TOKEN@archive.ubuntu.com/ubuntu/ priority:2\n",
        )
        for job, content in ((job, content)
                             for job in ("linux-artifact-contract", "kleidiai-dispatch-emulated")
                             for content in fixtures):
            with self.subTest(job=job, content=content):
                self.path.write_bytes(content)
                marker = self.path.with_name("apt-called")
                marker.unlink(missing_ok=True)
                step = step_running(job_block("validate_wrapper.yml", job),
                                    "sudo python3 tools/prioritize_ubuntu_archive.py")
                run = textwrap.dedent(step.split("        run: |\n", 1)[1])
                run = run.replace("${{ matrix.arch }}", "arm64")
                # Only the test harness overrides the fixed path. Never invoke
                # real sudo/apt: a fake apt records any accidental fall-through.
                probe = (
                    "import sys; from pathlib import Path; "
                    f"sys.path.insert(0, {str(ROOT / 'tools')!r}); "
                    "import prioritize_ubuntu_archive as m; "
                    f"m.MIRRORLIST_PATH=Path({str(self.path)!r}); "
                    "sys.exit(m.main())"
                )
                apt_probe = f"from pathlib import Path; Path({str(marker)!r}).write_text('called')"
                sudo = (
                    'sudo() { if [ "$1" = python3 ]; then '
                    f'{shlex.quote(sys.executable)} -c {shlex.quote(probe)}; '
                    'elif [ "$1" = apt-get ]; then '
                    f'{shlex.quote(sys.executable)} -c {shlex.quote(apt_probe)}; '
                    'else return 98; fi; }\n'
                )
                result = subprocess.run(["bash", "-eo", "pipefail", "-c", sudo + run],
                                        env={**os.environ, "PINNED": "false"},
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(1, result.returncode)
                self.assertFalse(marker.exists())
                self.assertNotIn("PRIVATE", result.stdout + result.stderr)
                self.assertEqual(content, self.path.read_bytes())

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
