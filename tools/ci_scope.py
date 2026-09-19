#!/usr/bin/env python3
"""Conservative PR/main native selection; release qualification is unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess

# Only paths with no compiled-runtime dependency may avoid native jobs.
TOOLING = frozenset({
    'scripts/auto_release_dispatch.py', 'scripts/release_contract.py',
    'scripts/release_publication.py', 'scripts/release_version_policy.py',
    'scripts/verify_historical_release_metadata.py',
    'scripts/verify_release_provenance.py', 'scripts/generate_assets_manifest.sh',
    'tests/test_assets_manifest.py', 'tests/test_auto_release_dispatch.py',
    'tests/test_historical_release_metadata.py', 'tests/test_release_contract.py',
    'tests/test_release_publication.py', 'tests/test_release_version_policy.py',
    'tests/test_release_workflow_inputs.py',
    'tests/fixtures/historical_release_metadata.json',
})


def native_required(paths):
    return not paths or any(
        path not in TOOLING and not (
            path in {'README.md', 'CONTRIBUTING.md', 'LICENSE'}
            or (path.startswith('docs/') and path.endswith('.md'))
        ) for path in paths
    )


def changed_paths(base, head='HEAD'):
    # --no-renames reports both old deletion and new addition, including moves
    # from native inputs into docs/tooling. NUL records preserve arbitrary names.
    if not base or set(base) == {'0'}:
        return []
    output = subprocess.check_output([
        'git', 'diff', '--no-renames', '--name-only', '-z', base, head, '--'
    ])
    return output.decode('utf-8', errors='surrogateescape').rstrip('\0').split('\0') if output else []


def validate_results(needs):
    scope = needs.get('changes', {})
    if scope.get('result') != 'success':
        return False
    selected = scope.get('outputs', {}).get('native')
    if selected not in ('true', 'false'):
        return False
    expected = 'success' if selected == 'true' else 'skipped'
    jobs = {'android-arm64-isa', 'kleidiai-dispatch-emulated',
            'windows-arm64-kleidiai', 'linux-artifact-contract',
            'wrapper-contract', 'msvc-mtmd-link-contract'}
    return set(needs) == jobs | {'changes'} and all(
        needs[job].get('result') == expected for job in jobs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-results', action='store_true')
    args = parser.parse_args()
    if args.check_results:
        return 0 if validate_results(json.loads(os.environ['NEEDS_JSON'])) else 1
    # An unavailable diff fails the classifier, never silently skips work.
    paths = changed_paths(os.environ.get('BASE_SHA', ''))
    value = str(native_required(paths)).lower()
    print(json.dumps({'native': value, 'paths': paths}))
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as out:
        out.write(f'native={value}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
