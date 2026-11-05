#!/usr/bin/env python3
"""Read-only intake and independent Windows consumer evidence verification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import publish_prepared_release as prepared
from release_contract import ContractError

HERE = Path(__file__).resolve().parent
MAX_EXPANDED = 8 * 1024 ** 3
MAX_MEMBERS = 30000
NATIVE = re.compile(r'^(?:lib)?(?:llama(?:dart)?|mtmd|ggml(?:-.*)?)(?:-windows-x64)?\.dll$', re.I)
NGRAM_PROMPT = 'Repeat and continue this sequence exactly:\n' + '\n'.join(
    ['alpha beta gamma delta epsilon zeta eta theta'] * 5)
NGRAM_FIXTURE = {'strategy': 'ngram-simple', 'sizeN': 4, 'sizeM': 8, 'minHits': 1,
                 'promptSha256': hashlib.sha256(NGRAM_PROMPT.encode()).hexdigest()}


def require(value, message):
    if not value:
        raise ContractError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def load(path):
    return loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def no_reparse(path):
    path = Path(path).absolute()
    for node in [path, *path.parents]:
        if node.exists() or node.is_symlink():
            info = node.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not
                    (getattr(info, 'st_file_attributes', 0) & 0x400), 'link/reparse path rejected')


def resolver_context(root):
    return {'resolvedExecutablePath': str(root) + '/sdk/dart.exe',
            'currentDirectoryPath': str(root), 'environment': {}}


def resolver_fixture(scratch, missing=None):
    """Plan a fresh sibling control after checking its entire resolver ancestry."""
    scratch = Path(scratch).absolute()
    mixed = scratch / 'stage'
    missing = Path(missing).absolute() if missing is not None else scratch / 'resolver-missing-v2-only'
    require(mixed.is_dir() and missing.parent == mixed.parent and missing != mixed,
            'resolver controls must be distinct siblings')
    require(not missing.exists() and not missing.is_symlink(), 'missing-cache control must be fresh')
    no_reparse(missing)
    checked = []
    for ancestor in [missing, *missing.parents]:
        for relative in ('.dart_tool/lib', '.dart_tool/llamadart/native_bundles',
                         '.dart_tool/llamadart/native_bundles_v2'):
            candidate = ancestor / relative
            no_reparse(candidate)
            require(not candidate.exists(), 'resolver ancestor contains native cache or code assets')
        for directory in (ancestor, ancestor / 'sdk', ancestor / 'bin', ancestor / 'lib'):
            no_reparse(directory)
            if directory.is_dir():
                require(not any(NATIVE.fullmatch(child.name) for child in directory.iterdir()),
                        'resolver ancestor/executable context contains native decoy')
        checked.append(str(ancestor))
    return {'mixedRoot': str(mixed), 'missingRoot': str(missing), 'checkedAncestors': checked,
            'contexts': {'mixed': resolver_context(mixed), 'missing': resolver_context(missing)}}


def canonical(name):
    p = PurePosixPath(name)
    require(bool(name) and '\\' not in name and ':' not in name and not p.is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/'))
            and not any(ord(char) < 32 for char in name)
            and all(not part.endswith((' ', '.')) and not re.fullmatch(
                r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.I)
                for part in name.split('/')), 'noncanonical archive/inventory path')
    return name


def extract(archive, output, prefix=None):
    """Validate the entire archive before extracting regular files only."""
    archive, output = Path(archive), Path(output)
    require(not output.exists(), 'fresh extraction directory required')
    no_reparse(output.parent)
    selected = []
    seen = set()
    total = 0
    zipped = zipfile.is_zipfile(archive)
    handle = zipfile.ZipFile(archive) if zipped else tarfile.open(archive, 'r:*')
    with handle:
        entries = handle.infolist() if zipped else handle.getmembers()
        require(0 < len(entries) <= MAX_MEMBERS, 'archive member count exceeds limit')
        for member in entries:
            raw = member.orig_filename if zipped else member.name
            directory = member.is_dir() if zipped else member.isdir()
            # native_release packages TARs with `tar -C bundle .`.
            if not zipped:
                if directory and raw in ('.', './'):
                    continue
                if raw.startswith('./'):
                    raw = raw[2:]
            name = canonical(raw.rstrip('/') if raw.endswith('/') else raw)
            if zipped:
                require(member.filename == raw, 'ZIP member name was normalized from its raw header')
            require(name.casefold() not in seen, 'case-insensitive duplicate archive member')
            seen.add(name.casefold())
            regular = stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG) if zipped else member.isfile()
            if directory:
                if zipped:
                    require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFDIR), 'archive directory type differs')
                continue
            require(regular, 'archive contains link or nonregular file')
            size = member.file_size if zipped else member.size
            require(type(size) is int and size >= 0, 'invalid expanded file size')
            total += size
            require(total <= MAX_EXPANDED, 'expanded archive exceeds limit')
            if prefix is None or name.startswith(prefix):
                selected.append((member, name, size))
        require(selected, 'archive selected inventory empty')
        output.mkdir()
        inventory = []
        for member, name, size in selected:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            no_reparse(target.parent)
            source = handle.open(member) if zipped else handle.extractfile(member)
            require(source is not None, 'archive member unreadable')
            with source, target.open('xb') as stream:
                remaining = size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    require(block, 'short archive member')
                    stream.write(block)
                    remaining -= len(block)
                require(not source.read(1), 'oversized archive member')
            inventory.append({'relativePath': name, 'bytes': size, 'sha256': digest(target)})
    return inventory


def manifest(root):
    root = Path(root)
    no_reparse(root)
    result = []
    for path in sorted(root.rglob('*')):
        no_reparse(path)
        if path.is_file():
            require(path.stat().st_nlink == 1, 'hardlinked evidence/source file rejected')
            result.append({'relativePath': path.relative_to(root).as_posix(),
                           'bytes': path.stat().st_size, 'sha256': digest(path)})
    require(result, 'empty file inventory')
    return result


def validate_inventory(rows):
    require(isinstance(rows, list) and rows, 'missing inventory')
    seen = set()
    for row in rows:
        name = canonical(row['relativePath'])
        require(name.casefold() not in seen, 'duplicate inventory path')
        seen.add(name.casefold())
        require(type(row['bytes']) is int and row['bytes'] >= 0 and
                re.fullmatch('[a-f0-9]{64}', row['sha256']), 'invalid inventory identity')


def validate_identity(identity, expected):
    for key in ('executorSha', 'coreSha', 'ownerRunId', 'ownerArtifactId',
                'ownerArtifactSha256', 'ownerWorkflowHeadSha', 'ownerNativeCommit',
                'runId', 'runAttempt'):
        require(identity.get(key) == expected.get(key), 'qualification identity differs: ' + key)
    for key in ('executorSha', 'coreSha', 'ownerWorkflowHeadSha', 'ownerNativeCommit'):
        require(re.fullmatch('[a-f0-9]{40}', identity[key]), 'invalid expected source SHA')
    require(re.fullmatch('[a-f0-9]{64}', identity['ownerArtifactSha256']), 'invalid expected artifact digest')
    for key in ('ownerRunId', 'ownerArtifactId', 'runId', 'runAttempt'):
        require(re.fullmatch('[1-9][0-9]{0,19}', identity[key]), 'invalid expected run/artifact identity')


def intake(args, spec):
    require(args.core_sha == spec['core_sha'] and args.owner_run_id == spec['owner_run_id']
            and args.owner_workflow_head_sha == spec['owner_workflow_head_sha'], 'fixed campaign source differs')
    expected = prepared.Expected(args.owner_run_id, args.owner_artifact_id,
        args.owner_artifact_sha256, args.owner_workflow_head_sha, args.owner_native_commit,
        spec['upstream_ref'], spec['upstream_commit'], spec['native_tag'], spec['correlation_id'])
    # This calls validation/preparation only. No publication/history/write API is invoked.
    no_reparse(args.directory.parent)
    report, desired = prepared.prepare(expected, prepared.GitHub(), args.directory, ROOT)
    asset = desired.assets[f'llamadart-native-windows-x64-{spec["native_tag"]}.tar.gz']
    inventory = extract(asset.path, args.directory / 'windows-runtime')
    dlls = sorted((row for row in inventory if row['relativePath'].lower().endswith('.dll')),
                  key=lambda row: row['relativePath'])
    require(dlls, 'candidate Windows provider inventory missing')
    write(args.directory / 'intake.json', {'report': report, 'runtime': dlls,
        'windowsArchive': {'sha256': asset.sha256, 'bytes': asset.size},
        'nativeCommit': expected.native_commit, 'workflowHeadSha': expected.workflow_head_sha,
        'diagnosticOnly': True, 'publicationInvoked': False})


def parse_case(stdout):
    lines = stdout.splitlines()
    results = [loads(line[16:]) for line in lines if line.startswith('CONSUMER_RESULT ')]
    modules = [loads(line[15:]) for line in lines if line.startswith('PASSIVE_MODULE ')]
    registry = [loads(line[18:]) for line in lines if line.startswith('REGISTRY_IDENTITY ')]
    require(len(results) == 1 and modules and len(registry) == 1, 'missing/duplicate consumer or provider record')
    return results[0], modules, registry[0]


def within(path, root):
    # Evidence comes from Windows even when the independent verifier runs elsewhere.
    p, r = path.replace('\\', '/').casefold(), root.replace('\\', '/').rstrip('/').casefold()
    return p.startswith(r + '/')


def validate_case(case, stdout, runtime):
    result, modules, registry = parse_case(stdout)
    require(json.dumps(result.get('fixture'), sort_keys=True) == json.dumps(NGRAM_FIXTURE, sort_keys=True),
            'reviewed ngram fixture metadata differs')
    mixed, missing = case['cacheRoot'], case['missingCacheRoot']
    normalized = lambda path: PurePosixPath(path.replace('\\', '/').casefold())
    require(normalized(mixed).parent == normalized(missing).parent and
            normalized(mixed) != normalized(missing), 'resolver controls are not distinct siblings')
    contexts = {'mixed': resolver_context(mixed), 'missing': resolver_context(missing)}
    require(case.get('resolverContexts') == contexts and result.get('resolverContexts') == contexts,
            'explicit resolver executable/CWD/environment context differs')
    query = result.get('query', {})
    candidate = case['name'] not in ('old-default', 'old-aot')
    require(result.get('disposed') is True and not result.get('disposeError')
            and not result.get('observerError'), 'consumer cleanup/observation incomplete')
    require(query.get('hasNextn') is True and query.get('nextnNullFalse') is True and
            query.get('probeAfterPublicOutcome') is True and
            query.get('hasLegacyQuery') is (not candidate), 'query export/null contract differs')
    providers = [row for row in modules if NATIVE.fullmatch(Path(row['modulePath'].replace('\\', '/')).name)]
    allowed = {row['sha256'] for row in runtime}
    require(providers and all(row['sha256'] in allowed for row in providers), 'loaded foreign native provider bytes')
    names = {re.sub(r'(?:-windows-x64)?\.dll$', '',
                    Path(row['modulePath'].replace('\\', '/')).name.lower()).removeprefix('lib')
             for row in providers}
    require({'llamadart', 'llama', 'ggml', 'ggml-base', 'ggml-cpu'} <= names,
            'required loaded native provider missing')
    require(query.get('wrapperSha256') in allowed and query.get('nextnOwnerSha256') == query['wrapperSha256']
            and query.get('nextnOwnerPath', '').casefold() == query.get('wrapperPath', '').casefold(),
            'nextn symbol owner differs from selected wrapper')
    known = {row['modulePath'].replace('\\', '/').casefold() for row in providers}
    for row in providers:
        for symbol in row.get('exports', []) + row.get('imports', []):
            if re.match(r'^(ggml_backend_|llama_model_load|sd_dart_llama_model_load|llama_dart_model_load)', symbol['symbol']):
                require(symbol.get('ownerPath', '').replace('\\', '/').casefold() in known, 'foreign registry/model symbol owner')
    if not case['name'].endswith('-aot'):
        require(within(case['workingDirectory'], case['cacheRoot']) and
                within(case['expectedCache'], case['cacheRoot']),
                'mixed cache is not an actual consumer working-directory ancestor')
    if case['name'] == 'old-incompatible-candidate':
        require(case['exitCode'] != 0 and result.get('completed') is False and
                'llama_dart_speculative_need_embd' in result.get('error', ''),
                'unsupported old/candidate pairing did not show actual missing-query failure')
        return
    namespace = ('.dart_tool/llamadart/native_bundles_v2/v0.8.0-3/windows-x64/extracted' if candidate
                 else '.dart_tool/llamadart/native_bundles/v0.8.0-1/windows-x64/extracted')
    expected_missing = missing + '/' + namespace if candidate else missing + '/sdk'
    require(normalized(case['expectedCache']) == normalized(mixed + '/' + namespace) and
            normalized(case['expectedMissingCache']) == normalized(expected_missing),
            'cache namespace/fallback controls differ')
    require(case['exitCode'] == 0 and result.get('completed') is True and result.get('unloaded') is True,
            'supported public generation failed')
    require(result.get('maxTokens') == 128 and result.get('threads') == 2 and result.get('gpuLayers') == 0,
            'sampling/CPU execution contract differs')
    require(all(type(result.get(key)) is int and result[key] == 128
                for key in ('baselineEvalTokens', 'ngramEvalTokens', 'reloadEvalTokens')) and
            type(result.get('draftAttempts')) is int and result['draftAttempts'] > 0,
            'actual 128-token execution/attempt evidence missing')
    baseline, ngram, reloaded = (result.get(key) for key in ('baselineOutput', 'ngramOutput', 'reloadOutput'))
    require(isinstance(baseline, str) and baseline and baseline == ngram == reloaded and
            hashlib.sha256(baseline.encode()).hexdigest() == result.get('baselineSha256') and
            hashlib.sha256(ngram.encode()).hexdigest() == result.get('ngramSha256'),
            'raw full-output parity/hash evidence differs')
    require(result.get('fullOutputParity') is True and result.get('reloadParity') is True and
            result.get('baselineBytes', 0) > 0 and result.get('baselineSha256') == result.get('ngramSha256'),
            'full output/reload parity missing')
    require(type(result.get('drafted')) is int and result['drafted'] > 0 and
            type(result.get('accepted')) is int and 0 < result['accepted'] <= result['drafted'],
            'inactive ngram or no accepted draft tokens')
    require(registry.get('primaryMatchesCodeAsset') is True and registry.get('noUnintendedDuplicateRegistry') is True,
            'actual code-asset/registry ownership mismatch')
    require(result.get('cacheSelected', '').replace('\\', '/').casefold() == case['expectedCache'].replace('\\', '/').casefold(),
            'mixed legacy/v2 cache selection differs')
    require(result.get('missingLegacySelected', '').replace('\\', '/').casefold() == case['expectedMissingCache'].replace('\\', '/').casefold(),
            'missing-legacy cache observation differs')
    if case['name'].endswith('-aot'):
        require(case.get('cleanPath') is True and case.get('overridesAbsent') is True and case.get('cwdEmpty') is True,
                'AOT environment not clean')
        before, after = case['distributionBefore'], case['distributionAfter']
        validate_inventory(before); validate_inventory(after)
        require(before == after and before == case['builtInventory'], 'whole AOT distribution changed or incomplete')
        for root in case['sourceRoots']:
            require(not within(case['distributionRoot'], root) and not within(root, case['distributionRoot'])
                    and not within(case['workingDirectory'], root) and not within(root, case['workingDirectory']),
                    'AOT distribution/CWD overlaps a source tree')
        require(not within(case['distributionRoot'], case['workingDirectory']) and
                not within(case['workingDirectory'], case['distributionRoot']), 'AOT CWD overlaps bundle')
        by_path = {row['relativePath'].casefold(): row for row in before}
        for row in providers:
            require(within(row['modulePath'], case['distributionRoot']), 'AOT provider is outside the copied whole bundle')
            relative = row['modulePath'].replace('\\', '/')[len(case['distributionRoot'].replace('\\', '/').rstrip('/')) + 1:]
            copied = by_path.get(relative.casefold())
            require(copied is not None and copied['sha256'] == row['sha256'] and copied['bytes'] == row['bytes'],
                    'AOT loaded provider lacks exact copied path/hash/length binding')


def validate_intake(receipt, expected, spec):
    require(receipt.get('nativeCommit') == expected['ownerNativeCommit'] and
            receipt.get('workflowHeadSha') == expected['ownerWorkflowHeadSha'] and
            receipt.get('publicationInvoked') is False, 'candidate intake source differs')
    source = receipt['report']['source']
    run = source['workflow_run']; artifact = source['publication_artifact']
    require(source.get('correlation_id') == spec['correlation_id'] and
            source.get('status') == 'prepared' and source.get('release') is None and
            source.get('native_commit') == expected['ownerNativeCommit'] and
            source.get('native_release_tag') == spec['native_tag'] and
            source.get('llama_cpp_commit') == spec['upstream_commit'], 'same-run source contract differs')
    require(run.get('repository') == 'leehack/llamadart-native' and
            str(run.get('id')) == expected['ownerRunId'] and
            run.get('head_sha') == expected['ownerWorkflowHeadSha'] and
            type(run.get('attempt')) is int and run['attempt'] > 0 and
            source.get('smoke') == {'policy': 'required', 'conclusion': 'passed'}, 'original workflow/smoke differs')
    require(str(artifact.get('id')) == expected['ownerArtifactId'] and
            artifact.get('digest') == 'sha256:' + expected['ownerArtifactSha256'], 'original artifact identity differs')
    archive = f'llamadart-native-windows-x64-{spec["native_tag"]}.tar.gz'
    require(source['checksums']['entries'].get(archive) == receipt['windowsArchive']['sha256'],
            'selected Windows archive checksum differs')
    validate_inventory(receipt['runtime'])


def verify(directory, expected, spec):
    directory = Path(directory)
    sealed = load(directory / 'evidence-manifest.json')
    validate_inventory(sealed)
    actual = [row for row in manifest(directory) if row['relativePath'] != 'evidence-manifest.json']
    require(sealed == actual, 'sealed evidence has missing/extra/changed files')
    data = load(directory / 'terminal.json')
    require(data.get('completed') is True and data.get('failure') is None and
            data.get('allChildrenExited') is True, 'qualification terminal incomplete/failed')
    validate_identity(data['identity'], expected)
    require(all(data.get(key) is False for key in ('finalOrdinaryRepinQualified', 'gpuQualified', 'liveMtpQualified')),
            'out-of-scope qualification claim')
    require(data.get('modelSha256') == spec['assets']['model']['sha256'] and
            data.get('oldPackageSha256') == spec['assets']['old_package']['sha256'] and
            data.get('sdkSha256') == spec['assets']['sdk']['sha256'], 'fixed asset identity differs')
    intake_report = load(directory / 'intake.json')
    validate_intake(intake_report, expected, spec)
    candidate = data['runtimeInventories']['candidate']
    require(candidate == intake_report['runtime'], 'candidate provider inventory differs from intake')
    require(data.get('sourceUnchanged') is True and data.get('diagnosticCandidateOverride') is True,
            'source/overlay contract missing')
    require([case['name'] for case in data['cases']] == spec['cases'], 'missing/extra/duplicate case inventory')
    for case in data['cases']:
        require(case.get('timedOut') is False and case.get('childExited') is True, 'child timed out or survived')
        for field in ('stdout', 'stderr'):
            row = case[field]; canonical(row['path']); path = directory / row['path']
            no_reparse(path)
            require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'raw case log differs')
        family = 'old' if case['name'] in ('old-default', 'old-aot') else 'candidate'
        validate_case(case, (directory / case['stdout']['path']).read_text(encoding='utf-8-sig'), data['runtimeInventories'][family])
    return data


def seal(directory):
    directory = Path(directory)
    require(not (directory / 'evidence-manifest.json').exists(), 'evidence already sealed; no overwrite')
    rows = manifest(directory)
    write(directory / 'evidence-manifest.json', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('intake')
    for field in ('core_sha', 'owner_run_id', 'owner_artifact_id', 'owner_artifact_sha256',
                  'owner_workflow_head_sha', 'owner_native_commit'):
        i.add_argument('--' + field.replace('_', '-'), required=True)
    i.add_argument('--directory', type=Path, required=True)
    e = sub.add_parser('extract'); e.add_argument('archive', type=Path); e.add_argument('output', type=Path); e.add_argument('--prefix')
    m = sub.add_parser('manifest'); m.add_argument('directory', type=Path); m.add_argument('output', type=Path)
    s = sub.add_parser('seal'); s.add_argument('directory', type=Path)
    r = sub.add_parser('resolver-fixture'); r.add_argument('scratch', type=Path); r.add_argument('output', type=Path)
    v = sub.add_parser('verify'); v.add_argument('directory', type=Path); v.add_argument('--expected', type=Path, required=True)
    args = p.parse_args(); spec = load(HERE / 'inputs.json')
    if args.command == 'intake': intake(args, spec)
    elif args.command == 'extract': extract(args.archive, args.output, args.prefix)
    elif args.command == 'manifest': write(args.output, manifest(args.directory))
    elif args.command == 'seal': seal(args.directory)
    elif args.command == 'resolver-fixture': write(args.output, resolver_fixture(args.scratch))
    else: verify(args.directory, load(args.expected), spec)


if __name__ == '__main__':
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError) as error:
        sys.exit(str(error))
