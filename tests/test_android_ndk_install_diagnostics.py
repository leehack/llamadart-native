from __future__ import annotations

import errno
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import android_ndk_install_diagnostics as diagnostic
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_release_provenance import workflow_job


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'sdk'
        self.root.mkdir()
        self.operation = self.root / '.temp/PackageOperation01'
        self.operation.mkdir(parents=True)
        self.metadata = self.root.joinpath(*diagnostic.METADATA_PARTS)
        self.metadata.parent.mkdir(parents=True)
        self.metadata.write_text(f'# installer metadata\npath={self.operation}\nclass=BasicInstaller\n')
        self.archive = self.operation / diagnostic.ARCHIVE_NAME
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('secret-name-token-query', b'private payload not for logs')
        self.payload = buffer.getvalue()
        self.archive.write_bytes(self.payload)

    def capture(self):
        return diagnostic.capture(self.root, 'failure')

    def test_real_zip_is_hashed_and_checked_without_uploading_content(self):
        report = self.capture()
        self.assertEqual(report['status'], 'captured')
        evidence = report['archive']
        self.assertEqual(evidence['size'], len(self.payload))
        self.assertEqual(evidence['sha1'], hashlib.sha1(self.payload).hexdigest())
        self.assertEqual(evidence['sha256'], hashlib.sha256(self.payload).hexdigest())
        self.assertFalse(evidence['size_matches'])
        self.assertFalse(evidence['sha1_matches'])
        self.assertEqual(evidence['zip_integrity'], {'status': 'pass', 'exit_code': 0})
        encoded = json.dumps(report)
        self.assertNotIn('private payload', encoded)
        self.assertNotIn('secret-name', encoded)
        self.assertNotIn(str(self.root), encoded)
        self.assertLess(len(encoded), 4096)
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_truncated_zip_captures_digest_and_integrity_failure(self):
        self.archive.write_bytes(self.payload[:25])
        evidence = self.capture()['archive']
        self.assertEqual(evidence['sha1'], hashlib.sha1(self.payload[:25]).hexdigest())
        self.assertEqual(evidence['zip_integrity']['status'], 'fail')

    def test_matching_trusted_bytes_are_reported_separately_from_zip_check(self):
        with mock.patch.object(diagnostic, 'EXPECTED_SIZE', len(self.payload)), mock.patch.object(diagnostic, 'EXPECTED_SHA1', hashlib.sha1(self.payload).hexdigest()):
            evidence = self.capture()['archive']
        self.assertTrue(evidence['size_matches'])
        self.assertTrue(evidence['sha1_matches'])
        self.assertEqual(evidence['zip_integrity']['status'], 'pass')

    def test_normal_failure_retains_archive_for_following_capture(self):
        # sdkmanager exited nonzero; collection must run before another install.
        first = self.capture()
        self.assertEqual(self.capture(), first)
        self.assertTrue(self.metadata.exists())
        self.assertEqual(self.archive.read_bytes(), self.payload)

    def test_nonfailure_skips_without_touching_sdk(self):
        for outcome in ('success', 'skipped', 'cancelled'):
            with self.subTest(outcome=outcome), mock.patch.object(diagnostic.os, 'open', side_effect=AssertionError('unexpected read')):
                report = diagnostic.capture(self.root, outcome)
                self.assertEqual(report['status'], 'skipped')

    def test_missing_sdk_metadata_operation_and_archive_are_explicit(self):
        missing = Path(self.temp.name) / 'missing'
        self.assertEqual(diagnostic.capture(missing, 'failure')['status'], 'unavailable')
        self.metadata.unlink()
        self.assertEqual(self.capture()['reason'], 'installer-metadata-unavailable')
        self.metadata.write_text(f'path={self.operation}\n')
        self.archive.unlink()
        self.assertEqual(self.capture()['reason'], 'retained-archive-unavailable')
        self.operation.rmdir()
        self.assertEqual(self.capture()['status'], 'unavailable')

    def test_corrupt_or_ambiguous_metadata_does_not_echo_values(self):
        for value in ('', 'class=only', 'path=relative', f'path={self.operation}\npath=/private/secret', f'path={self.operation}\npa\\u0074h=/private/secret', 'path=\\uZZZZ', 'path=unterminated\\'):
            with self.subTest(value=value):
                self.metadata.write_text(value)
                report = self.capture()
                self.assertEqual(report['status'], 'rejected')
                self.assertNotIn('/private/secret', json.dumps(report))

    def test_java_properties_escapes_are_supported(self):
        self.metadata.write_text('path=' + str(self.operation).replace('/', '\\/') + '\n')
        self.assertEqual(self.capture()['status'], 'captured')

    def test_outside_or_traversing_operation_is_rejected_before_open(self):
        for value in (str(Path(self.temp.name)), str(self.root / '.temp/PackageOperation01/../PackageOperation01'), str(self.root / '.temp/not-an-operation'), str(self.root / '.temp/PackageOperation00'), str(self.root / '.temp/PackageOperation01/subdirectory'), 'https://example.test/token?secret=value'):
            with self.subTest(value=value):
                self.metadata.write_text('path=' + value + '\n')
                report = self.capture()
                self.assertEqual(report['status'], 'rejected')
                self.assertEqual(report['reason'], 'installer-path-outside-sdk-temp')
                self.assertNotIn('secret=value', json.dumps(report))

    def test_symlinks_in_every_evidence_component_are_rejected(self):
        for target in (self.root / 'ndk', self.metadata.parent, self.metadata, self.root / '.temp', self.operation, self.archive):
            with self.subTest(target=target):
                moved = target.with_name(target.name + '-actual')
                target.rename(moved)
                target.symlink_to(moved, target_is_directory=moved.is_dir())
                try:
                    self.assertEqual(self.capture()['status'], 'rejected')
                finally:
                    target.unlink()
                    moved.rename(target)

    def test_nonregular_evidence_is_rejected_without_blocking(self):
        self.archive.unlink()
        os.mkfifo(self.archive)
        self.assertEqual(self.capture()['reason'], 'evidence-not-regular-file')
        self.metadata.unlink()
        self.metadata.mkdir()
        self.assertEqual(self.capture()['status'], 'rejected')

    def test_large_metadata_and_archive_do_not_trigger_unbounded_reads(self):
        self.metadata.write_bytes(b'x' * (diagnostic.MAX_METADATA_BYTES + 1))
        self.assertEqual(self.capture()['reason'], 'installer-metadata-too-large')
        self.metadata.write_text(f'path={self.operation}\n')
        with mock.patch.object(diagnostic, 'EXPECTED_SIZE', 1), mock.patch.object(diagnostic, '_zip_integrity', side_effect=AssertionError('unexpected ZIP read')):
            evidence = self.capture()['archive']
        self.assertEqual(evidence['hash_status'], 'skipped-oversize')
        self.assertNotIn('sha1', evidence)

    def test_hash_deadline_is_explicit(self):
        with mock.patch.object(diagnostic.time, 'monotonic', side_effect=[0, 1000]):
            self.assertEqual(self.capture()['reason'], 'archive-read-bound-exceeded')

    def test_io_errors_are_sanitized(self):
        with mock.patch.object(diagnostic.os, 'read', side_effect=OSError(errno.EIO, 'https://secret.test/token')):
            report = self.capture()
        self.assertEqual(report['status'], 'error')
        self.assertEqual(report['errno'], errno.EIO)
        self.assertNotIn('secret.test', json.dumps(report))

    def test_zip_probe_has_no_payload_output_or_inherited_unzip_options(self):
        with mock.patch.object(diagnostic.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            self.capture()
        args, kwargs = run.call_args
        self.assertEqual(args[0][:2], ['/usr/bin/unzip', '-t'])
        self.assertEqual(kwargs['stdout'], subprocess.DEVNULL)
        self.assertEqual(kwargs['stderr'], subprocess.DEVNULL)
        self.assertEqual(kwargs['timeout'], diagnostic.ZIP_TIMEOUT_SECONDS)
        self.assertEqual(kwargs['env'], {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})

    def test_zip_timeout_and_missing_tool_are_explicit(self):
        for error, expected in ((subprocess.TimeoutExpired('unzip', 60), 'timeout'), (FileNotFoundError('private path'), 'unavailable')):
            with self.subTest(expected=expected), mock.patch.object(diagnostic.subprocess, 'run', side_effect=error):
                self.assertEqual(self.capture()['archive']['zip_integrity']['status'], expected)

    def test_cli_writes_only_a_sanitized_json_receipt(self):
        output = Path(self.temp.name) / 'out/receipt.json'
        run = subprocess.run([sys.executable, str(ROOT / 'tools/android_ndk_install_diagnostics.py'), '--sdk-root', str(self.root), '--setup-outcome', 'failure', '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), json.loads(output.read_text()))
        self.assertNotIn('private payload', run.stdout + run.stderr)
        self.assertEqual(sorted(output.parent.iterdir()), [output])


class WorkflowTests(unittest.TestCase):
    def test_both_validation_paths_capture_immediately_after_failed_setup(self):
        workflow = (ROOT / '.github/workflows/validate_wrapper.yml').read_text()
        for name in ('android-arm64-isa', 'android-vulkan-shaders'):
            job = workflow_job(workflow, name)
            self.assertLess(job.index('id: android-sdk'), job.index('Capture failed Android NDK installation'))
            self.assertLess(job.index('Capture failed Android NDK installation'), job.index('- name: Install build dependencies'))
            self.assertEqual(job.count("always() && steps.android-sdk.outcome == 'failure'"), 2)
            self.assertIn('python3 tools/android_ndk_install_diagnostics.py', job)
            self.assertIn('packages: platform-tools ndk;' + diagnostic.NDK_VERSION, job)
            self.assertIn('path: ${{ runner.temp }}/android-ndk-install-diagnostics.json', job)
            self.assertIn('timeout-minutes: 2', job)
            self.assertNotIn('continue-on-error', job)
            self.assertNotIn('sdkmanager --install', job)

    def test_release_stages_helper_before_historical_source_checkout(self):
        workflow = (ROOT / '.github/workflows/native_release.yml').read_text()
        self.assertIn('ANDROID_NDK_VERSION: ' + diagnostic.NDK_VERSION, workflow)
        job = workflow_job(workflow, 'build-android')
        self.assertLess(job.index('Stage Android setup diagnostic helper'), job.index('Checkout exact release source'))
        self.assertLess(job.index('Capture failed Android NDK installation'), job.index('Select pinned Android NDK'))
        self.assertIn('python3 "$RUNNER_TEMP/android_ndk_install_diagnostics.py"', job)
        self.assertEqual(job.count("always() && steps.android-sdk.outcome == 'failure'"), 2)
        self.assertIn('path: ${{ runner.temp }}/android-ndk-install-diagnostics.json', job)
        self.assertNotIn('continue-on-error', job)


if __name__ == '__main__':
    unittest.main()
