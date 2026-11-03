"""No cloud writes: real bundle/ZIP bytes and mocked read-only GitHub fixtures."""
from __future__ import annotations
import copy
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import publish_prepared_release as prepared
from release_contract import ContractError, build_release_result
from release_publication import ExistingTag, _sha256, _expected_release_assets
from verify_release_provenance import (direct_dispatch_input_expressions,
                                       publication_serialization_errors, workflow_run_blocks)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


class ReadOnlyApi:
    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = []

    def get(self, resource):
        self.calls.append(resource)
        return copy.deepcopy(self.fixture.run if '/runs/' in resource else self.fixture.artifact)

    def pages(self, resource, key):
        self.calls.append(resource)
        return copy.deepcopy(self.fixture.jobs if key == 'jobs' else [self.fixture.result_artifact])

    def list_all(self, resource):
        self.calls.append(resource)
        return copy.deepcopy([{'name': name} for name in self.fixture.tags] if '/tags?' in resource else self.fixture.releases)

    def download(self, artifact, output, maximum=prepared.MAX_ARCHIVE_BYTES):
        source = self.fixture.result_zip if artifact['id'] == 789 else self.fixture.archive
        if source.stat().st_size > maximum:
            raise ContractError('oversize fixture')
        shutil.copyfile(source, output)
        if output.stat().st_size != artifact['size_in_bytes'] or 'sha256:' + _sha256(output) != artifact['digest']:
            raise ContractError('fixture API digest differs')


class PreparedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        environment = mock.patch.dict(os.environ, {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_AUTHOR_NAME': 'fixture', 'GIT_AUTHOR_EMAIL': 'fixture@example.invalid',
            'GIT_COMMITTER_NAME': 'fixture', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid'})
        environment.start(); self.addCleanup(environment.stop)
        self.owner = self.temp / 'owner'; self.owner.mkdir()
        upstream = self.owner / 'third_party/llama.cpp'; upstream.mkdir(parents=True)
        git(upstream, 'init', '-q'); (upstream / 'source.c').write_text('fixture\n')
        git(upstream, 'add', '.'); git(upstream, 'commit', '-qm', 'upstream')
        upstream_sha = git(upstream, 'rev-parse', 'HEAD')
        git(self.owner, 'init', '-q'); git(self.owner, '-c', 'advice.addEmbeddedRepo=false', 'add', '.')
        git(self.owner, 'commit', '-qm', 'native source')
        native = git(self.owner, 'rev-parse', 'HEAD')
        self.reader = self.temp / 'reader'; self.reader.mkdir(); git(self.reader, 'init', '-q')
        self.expected = prepared.Expected('123', '456', '0' * 64, 'a' * 40, native,
                                          'v0.8.0', upstream_sha, 'v0.8.0-2', 'review/prepared-123')
        self.files = self.temp / 'files'; assets = self.files / 'release_assets'; assets.mkdir(parents=True)
        entries = []
        for name in sorted(_expected_release_assets(self.expected.native_release_tag) - {'assets.json', 'SHA256SUMS'}):
            path = assets / name; path.write_bytes(('qualified:' + name).encode())
            entries.append({'file': name, 'size': path.stat().st_size, 'sha256': _sha256(path)})
        manifest = {'tag': self.expected.native_release_tag, 'native_release_tag': self.expected.native_release_tag,
            'llama_cpp_tag': self.expected.upstream_ref, 'llama_cpp_commit': upstream_sha, 'native_commit': native,
            'correlation_id': self.expected.correlation_id, 'smoke_policy': 'required', 'smoke_conclusion': 'passed',
            'workflow_run_id': '123', 'workflow_run_url': self.expected.run_url, 'workflow_head_sha': 'a' * 40,
            'hook_contract_version': 2, 'artifacts': entries}
        (assets / 'assets.json').write_text(json.dumps(manifest))
        (assets / 'SHA256SUMS').write_text(''.join(item['sha256'] + '  ' + item['file'] + '\n' for item in entries))
        publication = self.files / 'publication'; publication.mkdir()
        (publication / 'native_commit').write_text(native + '\n')
        git(self.owner, 'update-ref', 'refs/heads/release-provenance', native)
        git(self.owner, 'bundle', 'create', str(publication / 'release-provenance.bundle'), 'refs/heads/release-provenance')
        self.archive = self.temp / 'publication.zip'
        with zipfile.ZipFile(self.archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for path in self.files.rglob('*'):
                if path.is_file(): zipped.write(path, str(path.relative_to(self.files)))
        from dataclasses import replace
        self.expected = replace(self.expected, artifact_sha256=_sha256(self.archive))
        self.run = {'id': 123, 'path': prepared.WORKFLOW_PATH, 'repository': {'id': 7, 'full_name': prepared.REPOSITORY},
            'head_repository': {'full_name': prepared.REPOSITORY}, 'event': 'workflow_dispatch', 'head_branch': 'main',
            'head_sha': 'a' * 40, 'status': 'completed', 'conclusion': 'success', 'run_attempt': 1}
        self.jobs = [dict(name=name, conclusion='success', status='completed', started_at='2026-01-01T00:00:00Z', completed_at='2026-01-01T00:01:00Z')
                     for name in (*prepared.SUCCESS_JOBS, *(prefix + ' fixture)' for prefix in prepared.MATRIX_JOB_PREFIXES))]
        self.jobs += [dict(name=name, conclusion='skipped', status='completed') for name in prepared.SKIPPED_JOBS]
        source = {'id': 123, 'head_sha': 'a' * 40, 'head_branch': 'main', 'repository_id': 7, 'head_repository_id': 7}
        self.artifact = {'id': 456, 'name': 'release-publication-123', 'expired': False, 'workflow_run': source,
            'created_at': '2026-01-01T00:00:30Z', 'digest': 'sha256:' + _sha256(self.archive), 'size_in_bytes': self.archive.stat().st_size}
        self.source_result = build_release_result(assets_dir=assets, release_metadata=None,
            native_release_tag=self.expected.native_release_tag, upstream_ref=self.expected.upstream_ref,
            upstream_commit=upstream_sha, native_commit=native, correlation_id=self.expected.correlation_id,
            smoke_policy='required', smoke_conclusion='passed', publish_release=False,
            workflow_repository=prepared.REPOSITORY, workflow_run_id='123', workflow_run_attempt='1',
            workflow_run_url=self.expected.run_url, workflow_head_sha='a' * 40, publication_artifact_id='456',
            publication_artifact_url=self.expected.run_url + '/artifacts/456', publication_artifact_digest=self.artifact['digest'])
        self.result_zip = self.temp / 'result.zip'; self.rewrite_result()
        self.result_artifact = dict(self.artifact, id=789, name='native-release-result-123',
                                    digest='sha256:' + _sha256(self.result_zip), size_in_bytes=self.result_zip.stat().st_size)
        self.api = ReadOnlyApi(self)
        self.tags = ['v0.8.0']
        self.releases = []
        self.existing_tag = None

    def rewrite_result(self):
        with zipfile.ZipFile(self.result_zip, 'w') as zipped:
            zipped.writestr('native-release-result.json', json.dumps(self.source_result))
        if hasattr(self, 'result_artifact'):
            self.result_artifact.update(digest='sha256:' + _sha256(self.result_zip), size_in_bytes=self.result_zip.stat().st_size)

    def prepare(self):
        return prepared.prepare(self.expected, self.api, self.temp / 'verified', self.reader, self.archive)

    def test_exact_bytes_source_and_policy_derived_wrapper_prerelease(self):
        report, desired = self.prepare()
        self.assertEqual(self.source_result, report['source']); self.assertTrue(desired.prerelease)
        self.assertEqual(16, len(desired.assets)); self.assertIn(self.expected.run_url, desired.body)
        self.assertIn('a' * 40, desired.body); self.assertEqual(self.expected.native_commit, git(self.reader, 'rev-parse', 'refs/remotes/prepared/release-provenance'))
        for path in (self.files / 'release_assets').iterdir():
            self.assertEqual(path.read_bytes(), (self.temp / 'verified/prepared/release_assets' / path.name).read_bytes())

    def test_untrusted_run_controls(self):
        changes = [('path', 'other.yml'), ('event', 'pull_request'), ('head_branch', 'feature'),
                   ('head_sha', 'b' * 40), ('status', 'in_progress'), ('conclusion', 'failure')]
        for key, value in changes:
            with self.subTest(key=key):
                run = copy.deepcopy(self.run); run[key] = value
                with self.assertRaises(ContractError): prepared.validate_run(self.expected, run, self.jobs)
        for key in ['repository', 'head_repository']:
            run = copy.deepcopy(self.run); run[key]['full_name'] = 'foreign/repo'
            with self.assertRaises(ContractError): prepared.validate_run(self.expected, run, self.jobs)

    def test_all_required_smoke_matrix_and_skip_gates(self):
        for name in ['smoke-release-contract', 'package-and-release', self.jobs[len(prepared.SUCCESS_JOBS)]['name']]:
            for mutation in ['missing', 'failure', 'skipped']:
                jobs = copy.deepcopy(self.jobs)
                if mutation == 'missing': jobs = [j for j in jobs if j['name'] != name]
                else: next(j for j in jobs if j['name'] == name)['conclusion'] = mutation
                with self.subTest(name=name, mutation=mutation), self.assertRaises(ContractError): prepared.validate_run(self.expected, self.run, jobs)
        jobs = copy.deepcopy(self.jobs); next(j for j in jobs if j['name'].startswith('build-android ('))['name'] = 'build-android (wrong-platform)'
        with self.assertRaises(ContractError): prepared.validate_run(self.expected, self.run, jobs)
        jobs = copy.deepcopy(self.jobs); next(j for j in jobs if j['name'] == 'publish-release')['conclusion'] = 'success'
        with self.assertRaises(ContractError): prepared.validate_run(self.expected, self.run, jobs)

    def test_artifact_source_digest_id_and_producer_controls(self):
        for field, value in [('id', 999), ('name', 'other'), ('expired', True), ('digest', None), ('created_at', '2025-12-31T00:00:00Z')]:
            artifact = copy.deepcopy(self.artifact); artifact[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                prepared.validate_artifact(self.expected, self.run, self.jobs, artifact, name='release-publication-123', job='package-and-release', artifact_id='456')
        for field, value in [('id', 999), ('head_sha', 'b' * 40), ('repository_id', 9), ('head_repository_id', 9), ('head_branch', 'other')]:
            artifact = copy.deepcopy(self.artifact); artifact['workflow_run'][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                prepared.validate_artifact(self.expected, self.run, self.jobs, artifact, name='release-publication-123', job='package-and-release')

    def test_qualified_digest_changed(self):
        self.artifact['digest'] = 'sha256:' + 'b' * 64
        with self.assertRaisesRegex(ContractError, 'approved artifact'): self.prepare()

    def test_actual_zip_byte_corruption(self):
        with self.archive.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ContractError, 'archive differs'): self.prepare()

    def test_source_result_correlation_and_run_must_match(self):
        self.source_result['correlation_id'] = 'other'; self.rewrite_result()
        with self.assertRaisesRegex(ContractError, 'source release result'): self.prepare()

    def test_bundle_ref_commit_and_gitlink(self):
        from dataclasses import replace
        prepared_dir = self.temp / 'source-only'; shutil.copytree(self.files, prepared_dir)
        (prepared_dir / 'publication/native_commit').write_text('b' * 40 + '\n')
        with self.assertRaisesRegex(ContractError, 'recorded native'): prepared.import_provenance(self.expected, prepared_dir, self.reader)
        (prepared_dir / 'publication/native_commit').write_text(self.expected.native_commit + '\n')
        with self.assertRaisesRegex(ContractError, 'gitlink'): prepared.import_provenance(replace(self.expected, upstream_commit='b' * 40), prepared_dir, self.reader)
        git(self.owner, 'update-ref', 'refs/heads/other', self.expected.native_commit)
        git(self.owner, 'bundle', 'create', str(prepared_dir / 'publication/release-provenance.bundle'), 'refs/heads/other')
        with self.assertRaisesRegex(ContractError, 'bundle ref'): prepared.import_provenance(self.expected, prepared_dir, self.reader)

    def test_noncanonical_duplicate_symlink_publication_zip(self):
        for name, kind in [('release_assets/./assets.json', 0), ('../escape', 0), ('release_assets/assets.json', stat.S_IFLNK), ('release_assets/', stat.S_IFLNK)]:
            archive = self.temp / ('unsafe-' + str(kind) + '-' + hashlib.sha256(name.encode()).hexdigest() + '.zip')
            shutil.copyfile(self.archive, archive)
            with zipfile.ZipFile(archive, 'a') as zipped:
                info = zipfile.ZipInfo(name); info.external_attr = (kind | 0o600) << 16; zipped.writestr(info, 'bad')
            with self.subTest(name=name), self.assertRaises(ContractError): prepared.extract_publication(archive, self.temp / ('unpack-' + archive.stem), self.expected)
            self.assertFalse((self.temp / 'escape').exists())

    def cli(self, approved=None, environment=None):
        args = ['prepared']
        for key in self.expected.__dataclass_fields__: args.extend(['--' + key.replace('_', '-'), getattr(self.expected, key)])
        args += ['--directory', str(self.temp / 'cli-verified'), '--archive', str(self.archive)]
        if approved is not None: args += ['--publish-approved', approved]
        env = {'GITHUB_REPOSITORY': prepared.REPOSITORY, 'GITHUB_REF': 'refs/heads/main', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
               'GITHUB_WORKFLOW_REF': prepared.REPOSITORY + '/.github/workflows/publish_prepared.yml@refs/heads/main',
               'GITHUB_RUN_ID': '999', 'GITHUB_SHA': 'c' * 40}
        env.update(environment or {})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), mock.patch.object(sys, 'argv', args), mock.patch.dict(os.environ, env), mock.patch.object(prepared, 'GitHub', return_value=self.api), mock.patch('release_publication._remote_tag', return_value=self.existing_tag), mock.patch.object(prepared.Path, 'cwd', return_value=self.reader):
            return prepared.main()

    def test_no_boolean_means_validate_only(self):
        with mock.patch.object(prepared, 'publish') as publisher:
            self.assertEqual(0, self.cli()); publisher.assert_not_called()

    def test_explicit_true_preserves_original_transaction_executor_separate(self):
        with mock.patch.object(prepared, 'publish') as publisher:
            self.assertEqual(0, self.cli('true')); publisher.assert_called_once()
        report = json.loads((self.temp / 'cli-verified/prepared-publication-report.json').read_text())
        self.assertEqual('published', report['status']); self.assertEqual(123, report['source']['workflow_run']['id'])
        self.assertEqual('999', report['publication_executor']['run_id']); self.assertTrue(publisher.call_args.args[1].prerelease)

    def test_wrong_workflow_ref_refuses_before_api_or_publisher(self):
        with mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true', {'GITHUB_REF': 'refs/heads/untrusted'})
        self.assertFalse(self.api.calls); publisher.assert_not_called()

    def test_production_api_requests_pin_github_and_get_only(self):
        with mock.patch.object(prepared.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=b'{"ok":true}', stderr=b'')) as runner:
            self.assertEqual({'ok': True}, prepared.GitHub().get('repos/' + prepared.REPOSITORY + '/actions/runs/123'))
        command = runner.call_args.args[0]
        self.assertEqual(['gh', 'api', '--hostname', 'github.com', '--method', 'GET'], command[:6])
        self.assertEqual(120, runner.call_args.kwargs['timeout'])

    def test_production_download_size_hash_and_error_output(self):
        output = self.temp / 'download.zip'
        def success(command, **kwargs):
            self.assertEqual(['gh', 'api', '--hostname', 'github.com', '--method', 'GET'], command[:6])
            self.assertTrue(command[-1].endswith('/456/zip'))
            kwargs['stdout'].write(self.archive.read_bytes())
            return subprocess.CompletedProcess(command, 0, stderr=b'')
        with mock.patch.object(prepared.subprocess, 'run', side_effect=success): prepared.GitHub().download(self.artifact, output)
        self.assertEqual(self.archive.read_bytes(), output.read_bytes()); output.unlink()
        artifact = dict(self.artifact, digest='sha256:' + 'f' * 64)
        with mock.patch.object(prepared.subprocess, 'run', side_effect=success), self.assertRaisesRegex(ContractError, 'size/digest'): prepared.GitHub().download(artifact, output)
        output.unlink()
        with mock.patch.object(prepared.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, stderr=b'SENSITIVE_REDIRECT_URL')):
            with self.assertRaises(ContractError) as error: prepared.GitHub().download(self.artifact, output)
            self.assertNotIn('SENSITIVE', str(error.exception))

    def exact_release(self, *, draft=False, complete=True):
        _, desired = self.prepare()
        self.existing_tag = ExistingTag(desired.native_commit, desired.transaction_id)
        self.tags.append(desired.tag)
        release = {'tag_name': desired.tag, 'name': desired.name, 'body': desired.body,
                   'draft': draft, 'prerelease': desired.prerelease,
                   'assets': [{'name': asset.name, 'size': asset.size, 'digest': 'sha256:' + asset.sha256}
                              for asset in desired.assets.values()] if complete else []}
        return desired, release

    def test_current_history_blocks_new_initial_old_release(self):
        for newer in ['v0.9.0', 'v0.8.0-3']:
            self.tags = ['v0.8.0', newer]
            with mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true')
            publisher.assert_not_called()
            shutil.rmtree(self.temp / 'cli-verified')

    def test_wrapper_requires_current_prior_release_lineage(self):
        self.tags = []
        with mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true')
        publisher.assert_not_called()

    def test_same_transaction_valid_history_tag_only_recovery(self):
        _, desired = self.prepare(); self.existing_tag = ExistingTag(desired.native_commit, desired.transaction_id); self.tags.append(desired.tag)
        with mock.patch.object(prepared, 'publish') as publisher:
            self.assertEqual(0, self.cli('true')); publisher.assert_called_once()
            self.assertEqual(desired.transaction_id, publisher.call_args.args[1].transaction_id)

    def test_same_transaction_valid_history_partial_draft_recovery(self):
        desired, release = self.exact_release(draft=True, complete=False); self.releases = [release]
        with mock.patch.object(prepared, 'publish') as publisher:
            self.assertEqual(0, self.cli('true')); publisher.assert_called_once()
            self.assertEqual(desired.transaction_id, publisher.call_args.args[1].transaction_id)

    def test_incomplete_old_recovery_after_newer_release_is_blocked(self):
        _, release = self.exact_release(draft=True, complete=False); self.tags.append('v0.9.0')
        for releases in [[], [release]]:
            self.releases = releases
            with mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true')
            publisher.assert_not_called(); shutil.rmtree(self.temp / 'cli-verified')

    def test_exact_published_replay_after_newer_is_read_only(self):
        desired, release = self.exact_release(); self.releases = [release]; self.tags.append('v0.9.0')
        with mock.patch.object(prepared, 'publish') as publisher:
            self.assertEqual(0, self.cli('true')); publisher.assert_not_called()
        report = json.loads((self.temp / 'cli-verified/prepared-publication-report.json').read_text())
        self.assertEqual(desired.transaction_id, report['transaction_id'])
        self.assertEqual('already-published-noop', report['history']['disposition'])
        self.assertEqual('blocked', report['history']['new_publication_policy'])
        self.assertIn('rollback', report['history']['reason'])

    def test_unmarked_or_wrong_transaction_never_bypasses_history(self):
        _, release = self.exact_release(); self.releases = [release]; self.tags.append('v0.9.0')
        for marker in [None, 'f' * 64]:
            self.existing_tag = ExistingTag(self.existing_tag.target, marker)
            with mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true')
            publisher.assert_not_called(); shutil.rmtree(self.temp / 'cli-verified')

    def test_complete_replay_still_requires_history_fetch_success(self):
        _, release = self.exact_release(); self.releases = [release]
        with mock.patch.object(self.api, 'list_all', side_effect=ContractError('history unavailable')), mock.patch.object(prepared, 'publish') as publisher, self.assertRaises(SystemExit): self.cli('true')
        publisher.assert_not_called()

    def test_production_history_pagination_reaches_newer_late_tag(self):
        _, desired = self.prepare()
        first = [{'name': 'v0.8.0'}] + [{'name': 'historical-' + str(index)} for index in range(99)]
        pages = [first, [{'name': 'v0.9.0'}]]
        def api(command, **kwargs):
            payload = (pages if '--paginate' in command else pages[:1]) if '/tags?' in command[-1] else [[]]
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload).encode(), stderr=b'')
        with mock.patch.object(prepared.subprocess, 'run', side_effect=api), mock.patch('release_publication._remote_tag', return_value=None):
            with self.assertRaisesRegex(ValueError, 'rollback'): prepared.check_current_history(self.expected, prepared.GitHub(), desired)

    def test_production_history_fetch_and_shape_fail_closed(self):
        with mock.patch.object(prepared.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, stdout=b'[[{"name":"v0.8.0"}]]', stderr=b'SENSITIVE_URL')):
            with self.assertRaisesRegex(ContractError, 'history request failed') as error: prepared.GitHub().list_all('repos/owner/tags?per_page=100')
            self.assertNotIn('SENSITIVE', str(error.exception))
        for payload in [{}, [], [{}], [[1]]]:
            with self.subTest(payload=payload), mock.patch.object(prepared.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(payload).encode(), stderr=b'')), self.assertRaises(ContractError): prepared.GitHub().list_all('repos/owner/tags?per_page=100')

    def test_both_entrypoints_hold_fixed_main_lock_through_publication(self):
        native = (ROOT / '.github/workflows/native_release.yml').read_text()
        replay = (ROOT / '.github/workflows/publish_prepared.yml').read_text()
        self.assertEqual([], publication_serialization_errors(native, replay))

    def test_separate_or_cancelling_workflow_lock_is_rejected(self):
        native = (ROOT / '.github/workflows/native_release.yml').read_text()
        replay = (ROOT / '.github/workflows/publish_prepared.yml').read_text()
        for name, workflow in [('native_release.yml', native), ('publish_prepared.yml', replay)]:
            for bad in [workflow.replace('group: native-release-main', 'group: native-release-other'),
                        workflow.replace('cancel-in-progress: false', 'cancel-in-progress: true')]:
                with self.subTest(name=name):
                    errors = publication_serialization_errors(bad if name == 'native_release.yml' else native,
                                                               bad if name == 'publish_prepared.yml' else replay)
                    self.assertTrue(any(name in error for error in errors))

    def test_workflow_has_no_build_or_raw_input_shell_and_defaults_closed(self):
        workflow = (ROOT / '.github/workflows/publish_prepared.yml').read_text()
        self.assertFalse(direct_dispatch_input_expressions(workflow))
        self.assertIn('default: false', workflow); self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn('cancel-in-progress: false', workflow); self.assertIn('actions: read', workflow)
        blocks = workflow_run_blocks(workflow); self.assertEqual(1, len(blocks))
        self.assertIn('--publish-approved "$INPUT_PUBLISH_APPROVED"', blocks[0]); self.assertNotIn('tools/build.py', workflow)

if __name__ == '__main__': unittest.main()
