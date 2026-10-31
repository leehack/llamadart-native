from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import build_identity
from verify_release_provenance import (
    release_source_contract_errors, workflow_job, workflow_run_blocks,
)

WORKFLOW = ROOT / '.github/workflows/native_release.yml'
ACTION = ROOT / '.github/actions/checkout-native-source/action.yml'


def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class ReleaseSourceIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        config = self.temp / 'gitconfig'
        config.touch()
        environment = mock.patch.dict(os.environ, {
            'GIT_CONFIG_GLOBAL': str(config), 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_AUTHOR_NAME': 'fixture', 'GIT_AUTHOR_EMAIL': 'test@example.invalid',
            'GIT_COMMITTER_NAME': 'fixture', 'GIT_COMMITTER_EMAIL': 'test@example.invalid',
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.source = self.temp / 'source'
        upstream = self.source / 'third_party/llama.cpp'
        (upstream / 'ggml').mkdir(parents=True)
        (upstream / 'ggml/source.c').write_text('old\n')
        git(upstream, 'init', '-q')
        self.commit(upstream, 'old upstream')
        (self.source / 'wrapper.c').write_text('wrapper\n')
        patches = self.source / 'patches/llama.cpp'
        patches.mkdir(parents=True)
        (patches / 'series.json').write_text('{"patches": []}\n')
        git(self.source, 'init', '-q')
        self.commit(self.source, 'owner pins old upstream')
        self.owner = git(self.source, 'rev-parse', 'HEAD')
        (upstream / 'ggml/source.c').write_text('new\n')
        self.commit(upstream, 'selected upstream')
        self.upstream = git(upstream, 'rev-parse', 'HEAD')
        self.workflow = WORKFLOW.read_text()

    def commit(self, repo: Path, message: str) -> None:
        git(repo, '-c', 'advice.addEmbeddedRepo=false', 'add', '-A')
        git(repo, 'commit', '-qm', message)

    def run_shell(self, block: str, **environment: str) -> subprocess.CompletedProcess:
        return subprocess.run(['bash', '-euo', 'pipefail', '-c', block],
                              cwd=self.source, env={**os.environ, **environment},
                              capture_output=True, text=True, timeout=15)

    def prepare(self) -> str:
        job = workflow_job(self.workflow, 'resolve-tag')
        blocks = [block for block in workflow_run_blocks(job)
                  if 'Release provenance commit:' in block]
        self.assertEqual(1, len(blocks), 'resolve-tag must commit the source before builds')
        output = self.temp / 'output'
        result = self.run_shell(blocks[0], LLAMA_CPP_COMMIT=self.upstream,
                                LLAMA_CPP_REF='v0.2.0', GITHUB_OUTPUT=str(output),
                                RUNNER_TEMP=str(self.temp))
        self.assertEqual(0, result.returncode, result.stderr)
        return dict(line.split('=', 1) for line in output.read_text().splitlines())['native_commit']

    def checkout(self, native: str, upstream: str | None = None) -> subprocess.CompletedProcess:
        blocks = [block for block in workflow_run_blocks(ACTION.read_text())
                  if 'git fetch' in block]
        self.assertEqual(1, len(blocks))
        return self.run_shell(blocks[0], EXPECTED_NATIVE_COMMIT=native,
                              EXPECTED_LLAMA_CPP_COMMIT=upstream or self.upstream,
                              RUNNER_TEMP=str(self.temp))

    def test_build_stamp_names_the_same_clean_commit_as_the_tag(self) -> None:
        self.assertIn('+dirty.', build_identity.revision(self.source, 'dirty'))
        native = self.prepare()
        self.assertNotEqual(self.owner, native)
        self.assertEqual(self.upstream,
                         git(self.source, 'rev-parse', f'{native}:third_party/llama.cpp'))
        git(self.source, 'checkout', '--detach', self.owner)
        result = self.checkout(native)
        self.assertEqual(0, result.returncode, result.stderr)
        identity = build_identity.identity(
            self.source, config='Release', compiler='Clang-17,Clang-17',
            target='Linux-x86_64', backends='cpu', options='0123456789abcdef')
        fields = dict(pair.split('=', 1) for pair in identity.split(';')[1:])
        self.assertEqual(native, fields['owner'])
        self.assertEqual(self.upstream, fields['llama.cpp'])
        self.assertNotIn('+dirty.', identity)
        self.assertEqual(self.upstream, build_identity.revision(self.source / 'third_party/llama.cpp'))

    def test_an_already_pinned_source_keeps_its_commit(self) -> None:
        native = self.prepare()
        self.assertEqual(native, self.prepare())
        self.assertEqual(native, build_identity.revision(self.source, 'dirty'))

    def test_import_refuses_tracked_owner_or_upstream_edits(self) -> None:
        native = self.prepare()
        for relative in ('wrapper.c', 'third_party/llama.cpp/ggml/source.c'):
            with self.subTest(path=relative):
                path = self.source / relative
                before = path.read_text()
                try:
                    path.write_text('edited\n')
                    result = self.checkout(native)
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn('tracked changes', result.stderr)
                finally:
                    path.write_text(before)

    def test_import_refuses_a_different_native_or_upstream_commit(self) -> None:
        native = self.prepare()
        for expected_native, expected_upstream in ((self.owner, self.upstream), (native, 'a' * 40)):
            with self.subTest(native=expected_native, upstream=expected_upstream):
                git(self.source, 'checkout', '--detach', self.owner)
                result = self.checkout(expected_native, expected_upstream)
                self.assertNotEqual(0, result.returncode)
                self.assertEqual(self.owner, git(self.source, 'rev-parse', 'HEAD'))


class ReleaseSourceWiringTests(unittest.TestCase):
    def test_all_builds_import_the_resolved_owner_commit(self) -> None:
        self.assertEqual([], release_source_contract_errors(WORKFLOW.read_text()))

    def test_bypassing_any_lane_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text()
        for name in ('build-android', 'build-apple', 'build-linux', 'build-linux-hip',
                     'build-windows', 'package-and-release'):
            with self.subTest(lane=name):
                original = workflow_job(workflow, name)
                for before, after in (
                    ('uses: ./.github/actions/checkout-native-source',
                     'uses: ./.github/actions/checkout-other-source'),
                    ('      - name: Checkout exact release source\n',
                     '      - name: Checkout exact release source\n        if: false\n'),
                ):
                    mutated = original.replace(before, after)
                    self.assertNotEqual(original, mutated)
                    errors = release_source_contract_errors(workflow.replace(original, mutated))
                    self.assertTrue(any(name in error for error in errors), errors)

    def test_late_provenance_and_changed_manifest_source_are_rejected(self) -> None:
        workflow = WORKFLOW.read_text()
        for before, after in (
            ('- name: Create release provenance commit', '- name: Create later provenance'),
            ('LLAMADART_NATIVE_COMMIT: ${{ needs.resolve-tag.outputs.native_commit }}',
             'LLAMADART_NATIVE_COMMIT: ${{ github.sha }}'),
        ):
            with self.subTest(mutation=before):
                mutated = workflow.replace(before, after)
                self.assertNotEqual(workflow, mutated)
                self.assertTrue(release_source_contract_errors(mutated))


if __name__ == '__main__':
    unittest.main()
