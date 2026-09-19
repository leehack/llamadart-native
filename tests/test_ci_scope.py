import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from ci_scope import changed_paths, native_required, validate_results
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_release_provenance import workflow_job

JOBS = ('android-arm64-isa', 'kleidiai-dispatch-emulated',
        'windows-arm64-kleidiai', 'linux-artifact-contract',
        'wrapper-contract', 'msvc-mtmd-link-contract')


class ScopeTests(unittest.TestCase):
    def test_known_tooling_and_docs_avoid_compilation(self):
        for path in ('docs/release_version_policy.md', 'README.md',
                     'scripts/release_publication.py', 'tests/test_release_contract.py'):
            with self.subTest(path=path):
                self.assertFalse(native_required([path]))

    def test_runtime_shared_unknown_and_platform_inputs_keep_all_lanes(self):
        for path in ('src/wrapper.c', 'tests/tts_api_test.c',
                     'tests/test_kleidiai_windows.py', 'tests/new_test.py',
                     'tools/build.py', 'tools/package_linux_artifact.py',
                     'CMakePresets.json', 'CMakeLists.txt', '.gitmodules',
                     'third_party/llama.cpp', '.github/workflows/native_release.yml',
                     '.github/workflows/validate_wrapper.yml', 'tools/ci_scope.py',
                     'new/dependency.lock', 'docs/native.c', 'AGENTS.md'):
            with self.subTest(path=path):
                self.assertTrue(native_required(['README.md', path]))
        self.assertTrue(native_required([]))

    def test_real_git_add_modify_rename_delete_and_missing_diff(self):
        with tempfile.TemporaryDirectory() as temp:
            def git(*args):
                return subprocess.check_output(['git', '-C', temp, *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init'); git('config','user.email','fixture@example.test'); git('config','user.name','Fixture')
            root=Path(temp); (root/'src').mkdir(); (root/'docs').mkdir()
            (root/'src/native.c').write_text('original\n'); (root/'README.md').write_text('original\n')
            git('add','.'); git('commit','-m','base'); base=git('rev-parse','HEAD')
            previous=Path.cwd()
            try:
                os.chdir(temp)
                (root/'README.md').write_text('updated\n'); git('add','.');git('commit','-m','docs')
                self.assertFalse(native_required(changed_paths(base)))
                (root/'docs/new.md').write_text('new\n');git('add','.');git('commit','-m','add docs')
                self.assertFalse(native_required(changed_paths(base)))
                git('mv','src/native.c','docs/native.md');git('commit','-m','rename')
                paths=changed_paths(base)
                self.assertIn('src/native.c',paths);self.assertIn('docs/native.md',paths)
                self.assertTrue(native_required(paths))
                rename=git('rev-parse','HEAD');(root/'unknown.lock').write_text('dependency\n');git('add','.');git('commit','-m','unknown')
                self.assertTrue(native_required(changed_paths(rename)))
                unknown=git('rev-parse','HEAD');git('rm','unknown.lock');git('commit','-m','delete')
                self.assertTrue(native_required(changed_paths(unknown)))
                self.assertTrue(native_required(changed_paths('0'*40)))
                with self.assertRaises(subprocess.CalledProcessError): changed_paths('missing-ref')
            finally: os.chdir(previous)

    def test_aggregate_fails_for_missing_failed_cancelled_or_wrongly_skipped_lane(self):
        for selected in ('true','false'):
            needs={'changes':{'result':'success','outputs':{'native':selected}}}
            needs.update({job:{'result':'success' if selected=='true' else 'skipped'} for job in JOBS})
            self.assertTrue(validate_results(needs))
            for job in ('changes',*JOBS):
                for bad in ('failure','cancelled','skipped' if job=='changes' or selected=='true' else 'success'):
                    changed=json.loads(json.dumps(needs));changed[job]['result']=bad
                    self.assertFalse(validate_results(changed))
            missing=dict(needs);missing.pop(JOBS[0]);self.assertFalse(validate_results(missing))
        self.assertFalse(validate_results({'changes':{'result':'success','outputs':{}}}))

    def test_actual_workflow_routes_every_compiler_and_aggregate(self):
        workflow=(ROOT/'.github/workflows/validate_wrapper.yml').read_text()
        for job in JOBS:
            body=workflow_job(workflow,job)
            self.assertIn('needs: changes',body)
            self.assertIn("if: needs.changes.outputs.native == 'true'",body)
        aggregate=workflow_job(workflow,'wrapper-validation')
        self.assertIn('if: always()',aggregate)
        self.assertIn('tools/ci_scope.py --check-results',aggregate)
        for job in ('changes',*JOBS): self.assertIn(job,aggregate)
        changes=workflow_job(workflow,'changes')
        self.assertIn('fetch-depth: 0',changes)
        self.assertIn('github.event.pull_request.base.sha || github.event.before',changes)
        self.assertIn('tools/ci_scope.py',changes)
        windows=workflow_job(workflow,'windows-arm64-kleidiai')
        self.assertIn('upstream: [pinned, post-v0.4.0]',windows)
        for filename in ('validate_wrapper.yml','validate_release_provenance.yml'):
            content=(ROOT/'.github/workflows'/filename).read_text()
            triggers=content.split('permissions:')[0]
            self.assertNotIn('paths:',triggers)
            self.assertNotIn('draft',content)
            self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}",content)
            self.assertIn('github.event.pull_request.number || github.run_id',content)


if __name__ == '__main__': unittest.main()
