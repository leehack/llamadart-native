#!/usr/bin/env python3
"""Publish verified same-run preparation bytes without rebuilding them."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import zipfile

import release_publication as publication
from release_contract import ContractError, build_release_result, validate_dispatch
from release_version_policy import PolicyError, parse_native_tag, validate_history
from release_publication import (DesiredRelease, PublicationError, _expected_release_assets,
                                 _sha256, build_desired_release, publish)

REPOSITORY = 'leehack/llamadart-native'
WORKFLOW_PATH = '.github/workflows/native_release.yml'
MAX_ARCHIVE_BYTES = 4 * 1024 ** 3
SUCCESS_JOBS = ('resolve-tag', 'build-linux-hip', 'smoke-release-contract',
                'package-and-release', 'emit-release-result',
                'metal-os-floor / macos (macos-15)',
                'metal-os-floor / macos (macos-26)', 'metal-os-floor / ios-simulator')
MATRIX_COUNTS = {'build-android (': 4, 'build-apple (': 5,
                 'build-linux (': 5, 'build-windows (': 5}
MATRIX_JOB_PREFIXES = (
    'build-android (arm64-v8a, arm64, opencl,', 'build-android (arm64-v8a, arm64, vulkan,',
    'build-android (x86_64, x64, opencl,', 'build-android (x86_64, x64, vulkan,',
    'build-apple (macos-arm64,', 'build-apple (macos-x86_64,',
    'build-apple (ios-device-arm64,', 'build-apple (ios-sim-arm64,', 'build-apple (ios-sim-x86_64,',
    'build-linux (x64, cuda,', 'build-linux (x64, blas,', 'build-linux (x64, vulkan,',
    'build-linux (arm64, blas,', 'build-linux (arm64, vulkan,',
    'build-windows (x64, x64-windows, windows-latest, vulkan,',
    'build-windows (x64, x64-windows, windows-latest, blas,',
    'build-windows (x64, x64-windows, windows-2022, cuda,',
    'build-windows (arm64, arm64-windows, windows-11-vs2026-arm, vulkan,',
    'build-windows (arm64, arm64-windows, windows-11-vs2026-arm, blas,',
)
SKIPPED_JOBS = {'publish-release', 'update-llama-submodule', 'metal-os-floor / build'}


def require(condition: object, message: str) -> None:
    if not condition:
        raise ContractError(message)


@dataclass(frozen=True)
class Expected:
    run_id: str
    artifact_id: str
    artifact_sha256: str
    workflow_head_sha: str
    native_commit: str
    upstream_ref: str
    upstream_commit: str
    native_release_tag: str
    correlation_id: str

    def validate(self) -> dict[str, str]:
        for value in (self.run_id, self.artifact_id):
            require(re.fullmatch(r'[1-9][0-9]{0,19}', value), 'run/artifact IDs must be canonical positive integers')
        require(re.fullmatch(r'[0-9a-f]{64}', self.artifact_sha256), 'artifact SHA256 must be full lowercase hex')
        for value in (self.workflow_head_sha, self.native_commit, self.upstream_commit):
            require(re.fullmatch(r'[0-9a-f]{40}', value), 'expected native/head SHA must be full lowercase hex')
        return validate_dispatch(upstream_ref=self.upstream_ref,
                                 upstream_commit=self.upstream_commit,
                                 native_release_tag=self.native_release_tag,
                                 correlation_id=self.correlation_id,
                                 smoke_policy='required', publish_release=True)

    @property
    def run_url(self) -> str:
        return f'https://github.com/{REPOSITORY}/actions/runs/{self.run_id}'


class GitHub:
    """Read-only API access; archive downloads are bounded by declared size/time."""
    def get(self, resource: str) -> dict:
        result = subprocess.run(['gh', 'api', '--hostname', 'github.com',
                                 '--method', 'GET', resource],
                                capture_output=True, timeout=120, check=False)
        require(result.returncode == 0, 'GitHub provenance request failed')
        require(len(result.stdout) <= 8 * 1024 ** 2, 'GitHub provenance response too large')
        payload = json.loads(result.stdout)
        require(isinstance(payload, dict), 'GitHub provenance response must be an object')
        return payload

    def pages(self, resource: str, key: str) -> list[dict]:
        values: list[dict] = []
        for page in range(1, 11):
            payload = self.get(resource + ('&' if '?' in resource else '?') + f'per_page=100&page={page}')
            batch = payload.get(key)
            require(isinstance(batch, list), 'GitHub paginated response differs')
            values.extend(batch)
            if len(values) == payload.get('total_count'):
                return values
            require(len(batch) == 100, 'Incomplete GitHub provenance inventory')
        raise ContractError('GitHub provenance inventory exceeds supported limit')

    def list_all(self, resource: str) -> list[dict]:
        result = subprocess.run(['gh', 'api', '--hostname', 'github.com', '--method', 'GET',
                                 '--paginate', '--slurp', resource],
                                capture_output=True, timeout=120, check=False)
        require(result.returncode == 0, 'GitHub release history request failed')
        require(len(result.stdout) <= 8 * 1024 ** 2, 'GitHub release history too large')
        pages = json.loads(result.stdout)
        require(isinstance(pages, list) and bool(pages) and
                all(isinstance(page, list) and all(isinstance(item, dict) for item in page) for page in pages),
                'GitHub release history pagination shape differs')
        return [item for page in pages for item in page]

    def download(self, artifact: dict, output: Path, maximum: int = MAX_ARCHIVE_BYTES) -> None:
        size = artifact.get('size_in_bytes')
        require(type(size) is int and 0 < size <= maximum, 'artifact exceeds supported size')
        with output.open('xb') as stream:
            result = subprocess.run(['gh', 'api', '--hostname', 'github.com', '--method', 'GET',
                                     f'repos/{REPOSITORY}/actions/artifacts/{artifact["id"]}/zip'],
                                    stdout=stream, stderr=subprocess.PIPE, timeout=900, check=False)
        require(result.returncode == 0, 'GitHub artifact download failed')
        require(output.stat().st_size == size and 'sha256:' + _sha256(output) == artifact.get('digest'), 'downloaded artifact size/digest differs')


def timestamp(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'GitHub timestamp must include timezone')
    return parsed


def validate_run(expected: Expected, run: dict, jobs: list[dict]) -> None:
    require(run.get('id') == int(expected.run_id) and run.get('path') == WORKFLOW_PATH,
            'source must be the exact native release workflow run')
    require(run.get('repository', {}).get('full_name') == REPOSITORY and
            run.get('head_repository', {}).get('full_name') == REPOSITORY,
            'source run must belong to the owner repository, not a fork')
    require(run.get('event') == 'workflow_dispatch' and run.get('head_branch') == 'main' and
            run.get('head_sha') == expected.workflow_head_sha,
            'source must be the expected manually dispatched main head')
    require(run.get('status') == 'completed' and run.get('conclusion') == 'success',
            'source preparation must be completed successfully')
    require(type(run.get('run_attempt')) is int and run['run_attempt'] > 0, 'missing run attempt')
    names = [job.get('name') for job in jobs]
    require(len(names) == len(set(names)), 'ambiguous source job names')
    for job in jobs:
        require(job.get('status') == 'completed' and
                job.get('conclusion') == ('skipped' if job.get('name') in SKIPPED_JOBS else 'success'),
                'source job is failed, incomplete or unexpectedly skipped')
    require(all(names.count(name) == 1 for name in (*SUCCESS_JOBS, *SKIPPED_JOBS)),
            'source preparation is missing required build/smoke/publication gates')
    for prefix in MATRIX_JOB_PREFIXES:
        require(sum(isinstance(name, str) and name.startswith(prefix) for name in names) == 1,
                'source platform/backend matrix gate missing or ambiguous')
    for prefix, count in MATRIX_COUNTS.items():
        require(sum(isinstance(name, str) and name.startswith(prefix) for name in names) == count,
                'source build matrix coverage differs')


def validate_artifact(expected: Expected, run: dict, jobs: list[dict], artifact: dict,
                      *, name: str, job: str, artifact_id: str | None = None) -> None:
    require(artifact.get('name') == name and artifact.get('expired') is False,
            'artifact name/availability differs')
    require(type(artifact.get('id')) is int and artifact['id'] > 0, 'invalid artifact ID')
    if artifact_id is not None:
        require(artifact['id'] == int(artifact_id), 'artifact ID differs')
    source = artifact.get('workflow_run', {})
    require(source.get('id') == int(expected.run_id) and source.get('head_sha') == expected.workflow_head_sha and
            source.get('head_branch') == 'main' and
            source.get('repository_id') == run['repository']['id'] and
            source.get('head_repository_id') == run['repository']['id'],
            'artifact workflow/source repository identity differs')
    require(isinstance(artifact.get('digest'), str) and re.fullmatch(r'sha256:[0-9a-f]{64}', artifact['digest']),
            'artifact API digest missing')
    producer = next(value for value in jobs if value['name'] == job)
    require(timestamp(producer['started_at']) <= timestamp(artifact['created_at']) <= timestamp(producer['completed_at']),
            'artifact does not belong to the verified producer job')


def extract_publication(archive: Path, output: Path, expected: Expected) -> None:
    allowed = {f'release_assets/{name}' for name in _expected_release_assets(expected.native_release_tag)}
    allowed |= {'publication/native_commit', 'publication/release-provenance.bundle'}
    seen: set[str] = set()
    total = 0
    output.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = member.filename
            if member.is_dir():
                require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFDIR), 'publication directory type differs')
                require(name in ('release_assets/', 'publication/'), 'unexpected publication directory')
                continue
            require(name in allowed and name.casefold() not in seen, 'unexpected/duplicate/noncanonical publication member')
            require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG), 'publication members must be regular files')
            seen.add(name.casefold())
            total += member.file_size
            require(total <= MAX_ARCHIVE_BYTES, 'expanded publication artifact too large')
            target = output / name
            target.parent.mkdir(exist_ok=True)
            with zipped.open(member) as source, target.open('xb') as stream:
                shutil.copyfileobj(source, stream, 1024 * 1024)
    require(seen == {name.casefold() for name in allowed}, 'publication artifact inventory incomplete')


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(['git', '-C', str(repository), *arguments],
                            capture_output=True, text=True, timeout=120, check=False)
    require(result.returncode == 0, 'immutable source bundle verification failed')
    return result.stdout.strip()


def import_provenance(expected: Expected, prepared: Path, repository: Path) -> None:
    require((prepared / 'publication/native_commit').read_text() == expected.native_commit + '\n', 'recorded native commit differs')
    bundle = prepared / 'publication/release-provenance.bundle'
    require(bundle.stat().st_size <= 64 * 1024 ** 2, 'source bundle too large')
    require(git(repository, 'bundle', 'list-heads', str(bundle)).splitlines() ==
            [expected.native_commit + ' refs/heads/release-provenance'], 'source bundle ref differs')
    git(repository, 'fetch', '--no-tags', str(bundle),
        'refs/heads/release-provenance:refs/remotes/prepared/release-provenance')
    require(git(repository, 'rev-parse', 'refs/remotes/prepared/release-provenance') == expected.native_commit and
            git(repository, 'cat-file', '-t', expected.native_commit) == 'commit', 'imported native commit differs')
    require(git(repository, 'ls-tree', expected.native_commit, 'third_party/llama.cpp') ==
            '160000 commit ' + expected.upstream_commit + '\tthird_party/llama.cpp', 'imported upstream gitlink differs')


def prepare(expected: Expected, api: GitHub, directory: Path, repository: Path,
            archive: Path | None = None) -> tuple[dict, DesiredRelease]:
    contract = expected.validate()
    run = api.get(f'repos/{REPOSITORY}/actions/runs/{expected.run_id}')
    jobs = api.pages(f'repos/{REPOSITORY}/actions/runs/{expected.run_id}/jobs?filter=latest', 'jobs')
    validate_run(expected, run, jobs)
    artifact = api.get(f'repos/{REPOSITORY}/actions/artifacts/{expected.artifact_id}')
    validate_artifact(expected, run, jobs, artifact,
                      name='release-publication-' + expected.run_id, job='package-and-release',
                      artifact_id=expected.artifact_id)
    require(artifact['digest'] == 'sha256:' + expected.artifact_sha256, 'approved artifact digest differs from GitHub')
    results = [item for item in api.pages(f'repos/{REPOSITORY}/actions/runs/{expected.run_id}/artifacts', 'artifacts')
               if item.get('name') == 'native-release-result-' + expected.run_id]
    require(len(results) == 1, 'missing or ambiguous source release result artifact')
    result_artifact = results[0]
    validate_artifact(expected, run, jobs, result_artifact,
                      name='native-release-result-' + expected.run_id, job='emit-release-result')
    directory.mkdir()
    if archive is None:
        archive = directory / 'publication.zip'
        api.download(artifact, archive)
    require(archive.stat().st_size == artifact['size_in_bytes'] and _sha256(archive) == expected.artifact_sha256,
            'local publication archive differs from approved bytes')
    result_zip = directory / 'source-result.zip'
    api.download(result_artifact, result_zip, maximum=2 * 1024 ** 2)
    with zipfile.ZipFile(result_zip) as zipped:
        members = zipped.infolist()
        require(len(members) == 1 and members[0].filename == 'native-release-result.json' and
                stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG) and
                members[0].file_size <= 1024 ** 2, 'invalid source release result archive')
        source_result = json.loads(zipped.read(members[0]))
    prepared = directory / 'prepared'
    extract_publication(archive, prepared, expected)
    import_provenance(expected, prepared, repository)
    assets = prepared / 'release_assets'
    manifest = json.loads((assets / 'assets.json').read_text())
    require(manifest.get('workflow_run_id') == expected.run_id and manifest.get('workflow_run_url') == expected.run_url and
            manifest.get('workflow_head_sha') == expected.workflow_head_sha, 'manifest build workflow identity differs')
    result = build_release_result(assets_dir=assets, release_metadata=None,
        native_release_tag=expected.native_release_tag, upstream_ref=expected.upstream_ref,
        upstream_commit=expected.upstream_commit, native_commit=expected.native_commit,
        correlation_id=expected.correlation_id, smoke_policy='required', smoke_conclusion='passed',
        publish_release=False, workflow_repository=REPOSITORY, workflow_run_id=expected.run_id,
        workflow_run_attempt=str(run['run_attempt']), workflow_run_url=expected.run_url,
        workflow_head_sha=expected.workflow_head_sha, publication_artifact_id=expected.artifact_id,
        publication_artifact_url=expected.run_url + '/artifacts/' + expected.artifact_id,
        publication_artifact_digest='sha256:' + expected.artifact_sha256)
    require(source_result == result, 'GitHub source release result does not match exact prepared inputs')
    desired = build_desired_release(tag=expected.native_release_tag, native_commit=expected.native_commit,
        upstream_ref=expected.upstream_ref, upstream_commit=expected.upstream_commit,
        prerelease=contract['github_prerelease'] == 'true', assets_dir=assets,
        workflow_run_url=expected.run_url, artifact_digest='sha256:' + expected.artifact_sha256,
        correlation_id=expected.correlation_id, smoke_policy='required', smoke_conclusion='passed',
        workflow_head_sha=expected.workflow_head_sha)
    report = {'source': result, 'source_result_artifact_id': result_artifact['id'],
              'source_result_artifact_digest': result_artifact['digest'],
              'transaction_id': desired.transaction_id, 'github_prerelease': desired.prerelease}
    return report, desired


def check_current_history(expected: Expected, api: GitHub, desired: DesiredRelease) -> tuple[dict, bool]:
    """Only a fully verified published transaction can bypass new-publication ordering."""
    tags = api.list_all(f'repos/{REPOSITORY}/tags?per_page=100')
    releases = api.list_all(f'repos/{REPOSITORY}/releases?per_page=100')
    require(all(isinstance(row.get('name'), str) and row['name'] for row in tags) and
            all(isinstance(row.get('tag_name'), str) and row['tag_name'] for row in releases),
            'GitHub release history entries differ')
    tag_names = [row['name'] for row in tags]
    require(len(tag_names) == len(set(tag_names)), 'GitHub tag history is ambiguous')
    matches = [row for row in releases if row['tag_name'] == desired.tag]
    require(len(matches) <= 1, 'Multiple releases exist for the desired tag')
    tag = publication._remote_tag(REPOSITORY, desired.tag)
    require((tag is not None) == (desired.tag in tag_names), 'GitHub tag history/ref readbacks disagree')
    plan = publication.reconcile_publication(desired, tag=tag,
        release=publication._existing_release(matches[0] if matches else None))
    history = sorted(set(tag_names) | {row['tag_name'] for row in releases})
    policy = 'pass'
    reason = 'Current remote history permits this publication.'
    try:
        validate_history(parse_native_tag(expected.native_release_tag), history,
                         allow_existing_candidate=tag is not None)
    except PolicyError as error:
        if not plan.complete:
            raise
        policy = 'blocked'
        reason = str(error)
    return {'disposition': 'already-published-noop' if plan.complete else
                           ('initial-publication-permitted' if plan.create_tag else 'same-transaction-recovery-permitted'),
            'new_publication_policy': policy, 'reason': reason,
            'remote_tag_count': len(tags), 'release_count': len(releases),
            'history_sha256': hashlib.sha256(json.dumps(history).encode()).hexdigest()}, plan.complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in Expected.__dataclass_fields__:
        parser.add_argument('--' + field.replace('_', '-'), required=True)
    parser.add_argument('--publish-approved', choices=('true', 'false'), default='false')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    try:
        require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY and os.environ.get('GITHUB_REF') == 'refs/heads/main' and
                os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch' and
                os.environ.get('GITHUB_WORKFLOW_REF') == REPOSITORY + '/.github/workflows/publish_prepared.yml@refs/heads/main',
                'prepared publication is restricted to the owner main workflow')
        expected = Expected(**{key: getattr(args, key) for key in Expected.__dataclass_fields__})
        api = GitHub()
        report, desired = prepare(expected, api, args.directory, Path.cwd(), args.archive)
        report['history'], complete = check_current_history(expected, api, desired)
        report['publication_executor'] = {'run_id': os.environ.get('GITHUB_RUN_ID'),
                                        'head_sha': os.environ.get('GITHUB_SHA')}
        report['publish_approved'] = args.publish_approved == 'true'
        report['status'] = 'validated'
        output = args.directory / 'prepared-publication-report.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        if args.publish_approved == 'true':
            if not complete:
                publish(REPOSITORY, desired)
            report['status'] = 'published'
            output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'transaction_id': desired.transaction_id}))
    except (ContractError, PublicationError, OSError, ValueError, zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
