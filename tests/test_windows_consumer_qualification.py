"""Synthetic bytes/metadata only; no SDK, native library, model or network."""
from __future__ import annotations
import copy
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tools/qualification/windows_consumer'
spec = importlib.util.spec_from_file_location('windows_consumer_verifier', HERE / 'verify_evidence.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def zipped(self, entries, archive_name='input.zip'):
        archive = self.root / archive_name
        with zipfile.ZipFile(archive, 'w') as stream:
            for name, mode in entries:
                member = zipfile.ZipInfo(name)
                # ZipInfo normalizes backslashes on Windows and truncates NUL.
                # Preserve the adversarial header instead of writing a safe alias.
                member.filename = name
                member.orig_filename = name
                member.external_attr = mode << 16
                stream.writestr(member, b'synthetic fixture')
        with zipfile.ZipFile(archive) as stream:
            self.assertEqual([member.orig_filename for member in stream.infolist()],
                             [name for name, _ in entries])
        return archive

    def test_regular_inventory_is_extracted_and_bound(self):
        archive = self.zipped([('runtime/llama.dll', stat.S_IFREG)])
        rows = v.extract(archive, self.root / 'output')
        self.assertEqual(rows, v.manifest(self.root / 'output'))
        self.assertEqual(rows[0]['relativePath'], 'runtime/llama.dll')

    def test_extensionless_downloads_extract_to_separate_sdk_and_runtime_roots(self):
        downloads = self.root / 'downloads'
        downloads.mkdir()
        sdk = self.zipped([
            ('flutter/bin/cache/dart-sdk/bin/dart.exe', stat.S_IFREG),
            ('flutter/unselected.txt', stat.S_IFREG),
        ], 'sdk-fixture.zip')
        sdk = sdk.rename(downloads / 'sdk')
        runtime = self.tarred([('./llamadart.dll', tarfile.REGTYPE)])
        runtime = runtime.rename(downloads / 'old-runtime')
        for archive, prefix, member in (
            (sdk, 'flutter/bin/cache/dart-sdk/', 'flutter/bin/cache/dart-sdk/bin/dart.exe'),
            (runtime, None, 'llamadart.dll'),
        ):
            with self.subTest(archive=archive.name):
                original = archive.read_bytes()
                output = self.root / archive.name
                rows = v.extract(archive, output, prefix)
                self.assertEqual([member], [row['relativePath'] for row in rows])
                self.assertEqual(rows, v.manifest(output))
                for refused in (output, archive):
                    with self.assertRaisesRegex(v.ContractError, 'fresh extraction directory required'):
                        v.extract(archive, refused, prefix)
                self.assertEqual(original, archive.read_bytes())

    def test_bad_paths_types_and_case_collisions_fail_before_extraction(self):
        cases = [[(name, stat.S_IFREG)] for name in (
            '../escape', '/absolute', 'C:/escape', r'runtime\escape',
            './alias', 'a//b', 'runtime/NUL.dll', 'runtime/a. ', 'runtime/hidden\x00suffix',
        )]
        cases += [[('runtime/x', stat.S_IFLNK)],
                  [('runtime/', stat.S_IFLNK)],
                  [('runtime/X.dll', stat.S_IFREG), ('runtime/x.dll', stat.S_IFREG)]]
        for index, entries in enumerate(cases):
            with self.subTest(entries=entries):
                archive = self.zipped(entries, f'input-{index}.zip')
                output = self.root / f'output-{index}'
                with self.assertRaises(v.ContractError):
                    v.extract(archive, output)
                self.assertFalse(output.exists())

    def test_raw_backslash_header_is_rejected_when_windows_zip_reader_normalizes_it(self):
        with patch.object(zipfile.os, 'sep', '\\'):
            archive = self.zipped([(r'runtime\escape', stat.S_IFREG)])
            with zipfile.ZipFile(archive) as stream:
                member = stream.infolist()[0]
                self.assertEqual(member.orig_filename, r'runtime\escape')
                self.assertEqual(member.filename, 'runtime/escape')
            output = self.root / 'windows-bad-output'
            with self.assertRaisesRegex(v.ContractError, 'noncanonical'):
                v.extract(archive, output)
            self.assertFalse(output.exists())
            safe = self.zipped([('runtime/regular.dll', stat.S_IFREG)], 'safe.zip')
            rows = v.extract(safe, self.root / 'windows-safe-output')
            self.assertEqual(rows[0]['relativePath'], 'runtime/regular.dll')

    def test_tar_hardlink_is_rejected(self):
        archive = self.root / 'input.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            entry = tarfile.TarInfo('runtime/alias.dll')
            entry.type = tarfile.LNKTYPE
            entry.linkname = 'runtime/real.dll'
            stream.addfile(entry)
        with self.assertRaisesRegex(v.ContractError, 'nonregular'):
            v.extract(archive, self.root / 'output')

    def tarred(self, entries, archive_name='input.tar.gz'):
        archive = self.root / archive_name
        with tarfile.open(archive, 'w:gz') as stream:
            for name, kind in entries:
                member = tarfile.TarInfo(name)
                member.type = kind
                data = b'synthetic producer fixture' if kind == tarfile.REGTYPE else b''
                member.size = len(data)
                if kind in (tarfile.LNKTYPE, tarfile.SYMTYPE):
                    member.linkname = 'llamadart.dll'
                stream.addfile(member, io.BytesIO(data))
        return archive

    def test_producer_tar_dot_root_and_single_dot_prefix_are_extracted(self):
        for index, root in enumerate(('.', './')):
            with self.subTest(root=root):
                archive = self.tarred([(root, tarfile.DIRTYPE),
                    ('./ggml.dll', tarfile.REGTYPE), ('./llamadart.dll', tarfile.REGTYPE),
                    ('./include/', tarfile.DIRTYPE), ('./include/header.h', tarfile.REGTYPE)],
                    f'producer-{index}.tar.gz')
                output = self.root / f'producer-{index}'
                rows = v.extract(archive, output)
                self.assertEqual([row['relativePath'] for row in rows],
                                 ['ggml.dll', 'llamadart.dll', 'include/header.h'])
                self.assertEqual(sorted(rows, key=lambda row: row['relativePath']), v.manifest(output))

    def test_tar_normalization_preserves_strict_path_type_and_alias_guards(self):
        cases = [[(name, tarfile.REGTYPE)] for name in (
            '.', './', './../escape', '././escape', './nested/./escape',
            './nested/../escape', './nested//escape', '/absolute', './C:/escape',
            r'./runtime\escape', './runtime/NUL.dll', './runtime/a. ',
        )]
        cases += [[('./alias', kind)] for kind in (tarfile.LNKTYPE, tarfile.SYMTYPE, tarfile.FIFOTYPE)]
        cases += [[(name, tarfile.DIRTYPE)] for name in ('././', './nested/./')]
        cases += [[('llamadart.dll', tarfile.REGTYPE), ('./llamadart.dll', tarfile.REGTYPE)],
                  [('./LLAMADART.dll', tarfile.REGTYPE), ('llamadart.dll', tarfile.REGTYPE)],
                  [('./runtime', tarfile.DIRTYPE), ('runtime', tarfile.DIRTYPE)]]
        for index, entries in enumerate(cases):
            with self.subTest(entries=entries):
                archive = self.tarred(entries, f'bad-tar-{index}.tar.gz')
                output = self.root / f'bad-tar-{index}'
                with self.assertRaises(v.ContractError):
                    v.extract(archive, output)
                self.assertFalse(output.exists())

    def test_expanded_size_guard_runs_before_extraction(self):
        archive = self.zipped([('runtime/x', stat.S_IFREG)])
        with patch.object(v, 'MAX_EXPANDED', 1):
            with self.assertRaisesRegex(v.ContractError, 'expanded'):
                v.extract(archive, self.root / 'output')
        self.assertFalse((self.root / 'output').exists())

    def test_duplicate_json_and_inventory_paths_fail_closed(self):
        path = self.root / 'input.json';path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(v.ContractError, 'duplicate JSON'):
            v.load(path)
        with self.assertRaisesRegex(v.ContractError, 'duplicate inventory'):
            v.validate_inventory([{'relativePath': name, 'bytes': 1, 'sha256': 'a' * 64}
                                  for name in ['X.dll', 'x.dll']])


class ResolverFilesystem:
    """Model a complete drive without inheriting the host's root/bin links."""
    def __init__(self, path_type=PureWindowsPath):
        self.nodes = {}
        nodes = self.nodes

        class FixturePath(path_type):
            def absolute(self):
                assert self.is_absolute()
                return self

            def exists(self):
                return self in nodes

            def is_symlink(self):
                return self.exists() and stat.S_ISLNK(self.lstat().st_mode)

            def is_dir(self):
                return self.exists() and stat.S_ISDIR(self.lstat().st_mode)

            def lstat(self):
                if not self.exists():
                    raise FileNotFoundError(str(self))
                return nodes[self]

            def iterdir(self):
                return iter(node for node in nodes if node != self and node.parent == self)

        self.path = FixturePath

    def add(self, name, mode=stat.S_IFDIR, attributes=0):
        path = self.path(name)
        for node in reversed([path, *path.parents]):
            self.nodes.setdefault(node, SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0))
        self.nodes[path] = SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return path


class ResolverFixtureTopology(unittest.TestCase):
    def test_real_plan_keeps_mixed_caches_outside_the_missing_control_ancestry(self):
        tree = ResolverFilesystem()
        scratch = tree.path('C:/outer/scratch')
        mixed = scratch / 'stage'
        for namespace in ('native_bundles', 'native_bundles_v2'):
            tree.add(mixed / '.dart_tool/llamadart' / namespace / 'fixture/llama.dll', stat.S_IFREG)
        with patch.object(v, 'Path', tree.path):
            plan = v.resolver_fixture(scratch)
            missing = tree.path(plan['missingRoot'])
            self.assertEqual(mixed.parent, missing.parent)
            self.assertNotIn(mixed, missing.parents)
            self.assertEqual([str(missing), *map(str, missing.parents)], plan['checkedAncestors'])
            self.assertEqual(str(tree.path('C:/')), plan['checkedAncestors'][-1])
            self.assertEqual({}, plan['contexts']['missing']['environment'])
            self.assertFalse(missing.exists())
            with self.assertRaisesRegex(v.ContractError, 'siblings'):
                v.resolver_fixture(scratch, mixed / 'missing-legacy-cache')

    def test_linux_root_bin_symlink_reproduces_host_ancestry_rejection(self):
        tree = ResolverFilesystem(PurePosixPath)
        scratch = tree.add('/tmp/scratch/stage').parent
        tree.add('/bin', stat.S_IFLNK)
        with patch.object(v, 'Path', tree.path):
            with self.assertRaisesRegex(v.ContractError, 'link/reparse'):
                v.resolver_fixture(scratch)
            self.assertFalse((scratch / 'resolver-missing-v2-only').exists())

    def test_full_ancestor_cache_and_native_decoys_are_rejected_before_creation(self):
        for ancestor in ('C:/outer', 'C:/'):
            for relative in ('.dart_tool/lib', '.dart_tool/llamadart/native_bundles',
                             '.dart_tool/llamadart/native_bundles_v2', 'sdk', 'bin', 'lib', '.'):
                with self.subTest(ancestor=ancestor, relative=relative):
                    tree = ResolverFilesystem()
                    scratch = tree.add('C:/outer/scratch/stage').parent
                    decoy = tree.add(tree.path(ancestor) / relative)
                    if not relative.startswith('.dart_tool'):
                        tree.add(decoy / 'ggml-cpu.dll', stat.S_IFREG)
                    with patch.object(v, 'Path', tree.path):
                        with self.assertRaisesRegex(v.ContractError, 'ancestor'):
                            v.resolver_fixture(scratch)
                    self.assertFalse((scratch / 'resolver-missing-v2-only').exists())

    def test_windows_reparse_and_symlink_ancestors_are_rejected(self):
        for name in ('C:/', 'C:/outer', 'C:/outer/sdk', 'C:/bin',
                     'C:/lib', 'C:/outer/.dart_tool/llamadart/native_bundles'):
            for mode, attributes in ((stat.S_IFDIR, 0x400), (stat.S_IFLNK, 0)):
                with self.subTest(name=name, mode=mode, attributes=attributes):
                    tree = ResolverFilesystem()
                    scratch = tree.add('C:/outer/scratch/stage').parent
                    tree.add(name, mode, attributes)
                    with patch.object(v, 'Path', tree.path):
                        with self.assertRaisesRegex(v.ContractError, 'link/reparse'):
                            v.resolver_fixture(scratch)
                    self.assertFalse((scratch / 'resolver-missing-v2-only').exists())

    def test_existing_missing_tree_is_not_reused(self):
        tree = ResolverFilesystem()
        scratch = tree.add('C:/outer/scratch/stage').parent
        tree.add(scratch / 'resolver-missing-v2-only')
        with patch.object(v, 'Path', tree.path):
            with self.assertRaisesRegex(v.ContractError, 'fresh'):
                v.resolver_fixture(scratch)


def fixture(name):
    candidate = name not in ('old-default', 'old-aot')
    root = 'C:/distributed/' + name if name.endswith('-aot') else 'C:/stage/' + name
    runtime = []
    modules = []
    for i, leaf in enumerate(['llamadart.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll']):
        digest = str(i + 1) * 64
        runtime.append({'relativePath': leaf, 'bytes': 12, 'sha256': digest})
        modules.append({'modulePath': root + '/lib/' + leaf, 'bytes': 12, 'sha256': digest,
                        'imports': [], 'exports': []})
    missing_root = 'C:/resolver-missing-v2-only'
    namespace = ('.dart_tool/llamadart/native_bundles_v2/v0.8.0-3/windows-x64/extracted' if candidate
                 else '.dart_tool/llamadart/native_bundles/v0.8.0-1/windows-x64/extracted')
    expected_cache = 'C:/stage/' + namespace
    expected_missing = missing_root + '/' + namespace if candidate else missing_root + '/sdk'
    contexts = {'mixed': v.resolver_context('C:/stage'), 'missing': v.resolver_context(missing_root)}
    result = {'disposed': True, 'unloaded': True, 'completed': True, 'maxTokens': 128,
              'fixture': copy.deepcopy(v.NGRAM_FIXTURE), 'draftAttempts': 4,
              'baselineEvalTokens': 128, 'ngramEvalTokens': 128, 'reloadEvalTokens': 128,
              'resolverContexts': copy.deepcopy(contexts),
              'threads': 2, 'gpuLayers': 0, 'fullOutputParity': True, 'reloadParity': True,
              'baselineBytes': 128, 'baselineOutput': 'alpha ' * 32, 'ngramOutput': 'alpha ' * 32,
              'reloadOutput': 'alpha ' * 32,
              'baselineSha256': v.hashlib.sha256(('alpha ' * 32).encode()).hexdigest(),
              'ngramSha256': v.hashlib.sha256(('alpha ' * 32).encode()).hexdigest(),
              'drafted': 12, 'accepted': 8, 'cacheSelected': expected_cache,
              'missingLegacySelected': expected_missing,
              'query': {'hasLegacyQuery': not candidate, 'hasNextn': True, 'nextnNullFalse': True,
                        'probeAfterPublicOutcome': True, 'wrapperPath': modules[0]['modulePath'],
                        'wrapperSha256': runtime[0]['sha256'], 'nextnOwnerPath': modules[0]['modulePath'],
                        'nextnOwnerSha256': runtime[0]['sha256']}}
    case = {'name': name, 'exitCode': 0, 'expectedCache': expected_cache,
            'expectedMissingCache': expected_missing,
            'cacheRoot': 'C:/stage', 'missingCacheRoot': missing_root,
            'resolverContexts': contexts, 'workingDirectory': root}
    if name == 'old-incompatible-candidate':
        case['exitCode'] = 1;result['completed'] = False
        result['error'] = 'Failed to lookup symbol llama_dart_speculative_need_embd'
    if name.endswith('-aot'):
        rows = [{'relativePath': 'lib/' + row['relativePath'], 'bytes': row['bytes'], 'sha256': row['sha256']}
                for row in runtime]
        rows.append({'relativePath': 'bin/load_stock.exe', 'bytes': 12, 'sha256': 'f' * 64})
        case.update(cleanPath=True, overridesAbsent=True, cwdEmpty=True,
                    distributionRoot=root, workingDirectory='C:/empty/' + name,
                    sourceRoots=['C:/source/' + str(i) for i in range(6)],
                    builtInventory=copy.deepcopy(rows), distributionBefore=copy.deepcopy(rows),
                    distributionAfter=copy.deepcopy(rows))
    registry = {'primaryMatchesCodeAsset': True, 'noUnintendedDuplicateRegistry': True}
    return case, result, modules, registry, runtime


def log(result, modules, registry):
    return '\n'.join(['PASSIVE_MODULE ' + json.dumps(m) for m in modules] +
                     ['REGISTRY_IDENTITY ' + json.dumps(registry), 'CONSUMER_RESULT ' + json.dumps(result)])


class BehavioralOracle(unittest.TestCase):
    def test_all_five_named_cases_are_admitted_with_complete_synthetic_evidence(self):
        for name in v.load(HERE / 'inputs.json')['cases']:
            with self.subTest(name=name):
                case, result, modules, registry, runtime = fixture(name)
                v.validate_case(case, log(result, modules, registry), runtime)

    def test_generation_negative_controls_are_not_successes(self):
        mutations = [('drafted', 0), ('accepted', 0), ('accepted', 100), ('fullOutputParity', False),
                     ('reloadParity', False), ('ngramSha256', 'b' * 64), ('disposed', False),
                     ('unloaded', False), ('threads', 4), ('maxTokens', 16), ('gpuLayers', 1),
                     ('cacheSelected', 'C:/legacy/wrong'), ('missingLegacySelected', 'C:/v2/wrong'),
                     ('baselineEvalTokens', 127), ('ngramEvalTokens', 0), ('reloadEvalTokens', None),
                     ('draftAttempts', 0)]
        for key, value in mutations:
            with self.subTest(key=key):
                case, result, modules, registry, runtime = fixture('new-candidate')
                result[key] = value
                with self.assertRaises(v.ContractError):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_fixture_metadata_and_resolver_context_cannot_be_substituted(self):
        for key, value in [('strategy', 'ngram-map-k'), ('sizeN', 3), ('sizeM', 4),
                           ('minHits', 2), ('minHits', True), ('promptSha256', 'a' * 64)]:
            with self.subTest(key=key):
                case, result, modules, registry, runtime = fixture('new-candidate')
                result['fixture'][key] = value
                with self.assertRaisesRegex(v.ContractError, 'fixture metadata'):
                    v.validate_case(case, log(result, modules, registry), runtime)
        for change in ('descendant', 'environment', 'executable'):
            with self.subTest(change=change):
                case, result, modules, registry, runtime = fixture('new-candidate')
                if change == 'descendant':
                    case['missingCacheRoot'] = 'C:/stage/missing-legacy-cache'
                elif change == 'environment':
                    result['resolverContexts']['missing']['environment'] = {'LLAMADART_NATIVE_LIB_DIR': 'C:/foreign'}
                else:
                    result['resolverContexts']['missing']['resolvedExecutablePath'] = 'C:/foreign/dart.exe'
                with self.assertRaises(v.ContractError):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_mixed_cache_must_be_on_the_actual_child_working_directory_ancestry(self):
        for name in ('old-default', 'new-candidate', 'old-incompatible-candidate'):
            with self.subTest(name=name):
                case, result, modules, registry, runtime = fixture(name)
                v.validate_case(case, log(result, modules, registry), runtime)
                # Explicit resolver results still look correct, but this sibling
                # cache cannot affect discovery from the actual child CWD.
                case['cacheRoot'] = 'C:/stage/mixed-cache'
                case['expectedCache'] = case['cacheRoot'] + '/cache/selected'
                result['cacheSelected'] = case['expectedCache']
                case['missingCacheRoot'] = 'C:/stage/missing-only'
                case['resolverContexts'] = {key: v.resolver_context(case[field])
                                            for key, field in (('mixed', 'cacheRoot'), ('missing', 'missingCacheRoot'))}
                result['resolverContexts'] = copy.deepcopy(case['resolverContexts'])
                with self.assertRaisesRegex(v.ContractError, 'working-directory ancestor'):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_matching_return_and_expectation_cannot_waive_the_cache_namespace(self):
        for name in ('old-default', 'new-candidate'):
            with self.subTest(name=name):
                case, result, modules, registry, runtime = fixture(name)
                case['expectedMissingCache'] = case['missingCacheRoot'] + '/foreign'
                result['missingLegacySelected'] = case['expectedMissingCache']
                with self.assertRaisesRegex(v.ContractError, 'namespace/fallback'):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_missing_retained_export_and_unretired_query_fail(self):
        for key, value in [('hasNextn', False), ('hasLegacyQuery', True), ('nextnNullFalse', False),
                           ('nextnOwnerPath', 'C:/foreign/wrapper.dll'), ('nextnOwnerSha256', 'e' * 64)]:
            with self.subTest(key=key):
                case, result, modules, registry, runtime = fixture('new-candidate')
                result['query'][key] = value
                with self.assertRaises(v.ContractError):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_actual_old_incompatible_pair_is_observed_without_typed_error_requirement(self):
        case, result, modules, registry, runtime = fixture('old-incompatible-candidate')
        result['errorType'] = 'ArgumentError'
        v.validate_case(case, log(result, modules, registry), runtime)
        case['exitCode'] = 0
        with self.assertRaises(v.ContractError):
            v.validate_case(case, log(result, modules, registry), runtime)

    def test_wrong_provider_bytes_missing_provider_and_foreign_symbol_owner_fail(self):
        for kind in ('wrong-hash', 'missing-cpu', 'foreign-symbol'):
            with self.subTest(kind=kind):
                case, result, modules, registry, runtime = fixture('old-default')
                if kind == 'wrong-hash': modules[-1]['sha256'] = '0' * 64
                elif kind == 'missing-cpu': modules.pop()
                else: modules[1]['imports'] = [{'symbol': 'ggml_backend_reg_count', 'ownerPath': 'C:/foreign/ggml.dll'}]
                with self.assertRaises(v.ContractError):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_aot_requires_whole_unchanged_bundle_and_loaded_file_path_binding(self):
        for kind in ('missing-sidecar', 'after-changed', 'foreign-loaded', 'wrong-path', 'dirty-path', 'source-cwd'):
            with self.subTest(kind=kind):
                case, result, modules, registry, runtime = fixture('new-aot')
                if kind == 'missing-sidecar': case['distributionBefore'].pop()
                elif kind == 'after-changed': case['distributionAfter'][0]['sha256'] = '0' * 64
                elif kind == 'foreign-loaded': modules[-1]['modulePath'] = 'C:/foreign/ggml-cpu.dll'
                elif kind == 'wrong-path': modules[-1]['modulePath'] = case['distributionRoot'] + '/other/ggml-cpu.dll'
                elif kind == 'dirty-path': case['cleanPath'] = False
                else: case['workingDirectory'] = 'C:/source/1/cwd'
                with self.assertRaises(v.ContractError):
                    v.validate_case(case, log(result, modules, registry), runtime)

    def test_missing_and_duplicate_public_result_records_fail(self):
        case, result, modules, registry, runtime = fixture('old-default')
        valid = log(result, modules, registry)
        for malformed in (valid.replace('CONSUMER_RESULT ', 'ABSENT '), valid + '\nCONSUMER_RESULT ' + json.dumps(result)):
            with self.assertRaises(v.ContractError):
                v.validate_case(case, malformed, runtime)

    def test_independent_expected_identity_rejects_wrong_core_run_digest_and_native(self):
        fields = ('executorSha', 'coreSha', 'ownerRunId', 'ownerArtifactId', 'ownerArtifactSha256',
                  'ownerWorkflowHeadSha', 'ownerNativeCommit', 'runId', 'runAttempt')
        expected = {field: '1' for field in fields}
        for field in ('executorSha', 'coreSha', 'ownerWorkflowHeadSha', 'ownerNativeCommit'):
            expected[field] = 'a' * 40
        expected['ownerArtifactSha256'] = 'b' * 64
        v.validate_identity(expected, expected)
        for field in fields:
            actual = dict(expected);actual[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(v.ContractError):
                v.validate_identity(actual, expected)

    def test_intake_binds_original_result_artifact_and_distinct_native_commit(self):
        spec = v.load(HERE / 'inputs.json')
        expected = {'ownerNativeCommit': 'b' * 40, 'ownerWorkflowHeadSha': spec['owner_workflow_head_sha'],
                    'ownerRunId': spec['owner_run_id'], 'ownerArtifactId': '123', 'ownerArtifactSha256': 'c' * 64}
        receipt = {'nativeCommit': expected['ownerNativeCommit'], 'workflowHeadSha': expected['ownerWorkflowHeadSha'],
                   'publicationInvoked': False, 'windowsArchive': {'sha256': 'd' * 64},
                   'runtime': [{'relativePath': 'llamadart.dll', 'bytes': 1, 'sha256': 'e' * 64}],
                   'report': {'source': {'correlation_id': spec['correlation_id'], 'status': 'prepared', 'release': None,
                    'native_commit': expected['ownerNativeCommit'], 'native_release_tag': spec['native_tag'],
                    'llama_cpp_commit': spec['upstream_commit'],
                    'workflow_run': {'repository': 'leehack/llamadart-native', 'id': int(spec['owner_run_id']),
                                     'head_sha': expected['ownerWorkflowHeadSha'], 'attempt': 2},
                    'publication_artifact': {'id': 123, 'digest': 'sha256:' + 'c' * 64},
                    'smoke': {'policy': 'required', 'conclusion': 'passed'},
                    'checksums': {'entries': {'llamadart-native-windows-x64-v0.8.0-3.tar.gz': 'd' * 64}}}}}
        v.validate_intake(receipt, expected, spec)
        for field in ('ownerNativeCommit', 'ownerWorkflowHeadSha', 'ownerRunId', 'ownerArtifactId', 'ownerArtifactSha256'):
            wrong = dict(expected);wrong[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(v.ContractError):
                v.validate_intake(receipt, wrong, spec)
        receipt['report']['source']['smoke']['conclusion'] = 'skipped'
        with self.assertRaises(v.ContractError):
            v.validate_intake(receipt, expected, spec)

    def test_fixed_campaign_rejects_changed_head_before_any_api_call(self):
        from argparse import Namespace
        inputs = v.load(HERE / 'inputs.json')
        with patch.object(v.prepared, 'prepare') as prepare:
            with self.assertRaisesRegex(v.ContractError, 'campaign source'):
                v.intake(Namespace(core_sha='a' * 40), inputs)
            prepare.assert_not_called()


def complete_terminal(root):
    spec = v.load(HERE / 'inputs.json')
    expected = {'executorSha': 'a' * 40, 'coreSha': spec['core_sha'], 'ownerRunId': spec['owner_run_id'],
                'ownerArtifactId': '123', 'ownerArtifactSha256': 'b' * 64,
                'ownerWorkflowHeadSha': spec['owner_workflow_head_sha'], 'ownerNativeCommit': 'c' * 40,
                'runId': '456', 'runAttempt': '2'}
    cases = []
    for name in spec['cases']:
        case, result, modules, registry, runtime = fixture(name)
        case.update(timedOut=False, childExited=True)
        for field, text in [('stdout', log(result, modules, registry)), ('stderr', '')]:
            path = root / (name + '.' + field)
            path.write_text(text)
            case[field] = {'path': path.name, 'bytes': path.stat().st_size, 'sha256': v.digest(path)}
        cases.append(case)
    data = {'identity': expected, 'completed': True, 'failure': None, 'allChildrenExited': True, 'cases': cases,
            'modelSha256': spec['assets']['model']['sha256'], 'oldPackageSha256': spec['assets']['old_package']['sha256'],
            'sdkSha256': spec['assets']['sdk']['sha256'], 'sourceUnchanged': True, 'diagnosticCandidateOverride': True,
            'finalOrdinaryRepinQualified': False, 'gpuQualified': False, 'liveMtpQualified': False,
            'runtimeInventories': {'old': runtime, 'candidate': runtime}}
    receipt = {'nativeCommit': expected['ownerNativeCommit'], 'workflowHeadSha': expected['ownerWorkflowHeadSha'],
               'publicationInvoked': False, 'windowsArchive': {'sha256': 'd' * 64}, 'runtime': runtime,
               'report': {'source': {'correlation_id': spec['correlation_id'], 'status': 'prepared', 'release': None,
                'native_commit': expected['ownerNativeCommit'], 'native_release_tag': spec['native_tag'],
                'llama_cpp_commit': spec['upstream_commit'],
                'workflow_run': {'repository': 'leehack/llamadart-native', 'id': int(spec['owner_run_id']),
                                 'head_sha': expected['ownerWorkflowHeadSha'], 'attempt': 2},
                'publication_artifact': {'id': 123, 'digest': 'sha256:' + expected['ownerArtifactSha256']},
                'smoke': {'policy': 'required', 'conclusion': 'passed'},
                'checksums': {'entries': {'llamadart-native-windows-x64-v0.8.0-3.tar.gz': 'd' * 64}}}}}
    v.write(root / 'terminal.json', data);v.write(root / 'intake.json', receipt)
    return data, expected, spec


class SealingAndWorkflow(unittest.TestCase):
    def test_complete_terminal_is_verified_and_later_raw_log_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, expected, spec = complete_terminal(root)
            v.seal(root)
            self.assertEqual(v.verify(root, expected, spec), data)
            (root / 'new-candidate.stdout').write_text('changed after seal')
            with self.assertRaisesRegex(v.ContractError, 'sealed evidence'):
                v.verify(root, expected, spec)

    def test_terminal_requires_exact_case_inventory_and_completed_children(self):
        for kind in ('missing', 'duplicate', 'timeout', 'cleanup', 'gpu-claim', 'source-changed'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                data, expected, spec = complete_terminal(root)
                if kind == 'missing': data['cases'].pop()
                elif kind == 'duplicate': data['cases'].append(data['cases'][0])
                elif kind == 'timeout': data['cases'][0]['timedOut'] = True
                elif kind == 'cleanup': data['allChildrenExited'] = False
                elif kind == 'gpu-claim': data['gpuQualified'] = True
                else: data['sourceUnchanged'] = False
                v.write(root / 'terminal.json', data);v.seal(root)
                with self.assertRaises(v.ContractError):
                    v.verify(root, expected, spec)

    def test_seal_is_complete_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve();(root / 'failure.log').write_text('first failure')
            v.seal(root)
            self.assertEqual(v.load(root / 'evidence-manifest.json'),
                             [row for row in v.manifest(root) if row['relativePath'] != 'evidence-manifest.json'])
            with self.assertRaisesRegex(v.ContractError, 'already sealed'):
                v.seal(root)
            (root / 'failure.log').write_text('changed')
            self.assertNotEqual(v.load(root / 'evidence-manifest.json'),
                                [row for row in v.manifest(root) if row['relativePath'] != 'evidence-manifest.json'])

    def test_manual_qualification_and_model_free_source_check_never_publish(self):
        workflow = (ROOT / '.github/workflows/windows_consumer_qualification.yml').read_text()
        self.assertIn('workflow_dispatch:', workflow)
        source_check, qualification = workflow.split('  windows-consumer:', 1)
        self.assertIn('ParseFile(', source_check)
        self.assertNotIn('qualification.ps1 `', source_check)
        self.assertNotIn('INPUT_OWNER_ARTIFACT', source_check)
        self.assertIn("github.event_name == 'workflow_dispatch'", qualification)
        self.assertNotIn('contents: write', workflow)
        self.assertNotIn('secrets.', workflow)
        self.assertIn('actions: read', workflow)
        self.assertIn('runs-on: windows-2022', workflow)
        self.assertIn('overwrite: false', workflow)
        self.assertIn('github.run_attempt', workflow)
        self.assertIn('if: always()', workflow)
        source = (HERE / 'verify_evidence.py').read_text()
        import ast
        tree = ast.parse(source)
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call, ast.Attribute) and call.attr in ('publish', 'main') for call in calls))
        self.assertTrue(any(isinstance(call, ast.Attribute) and call.attr == 'prepare' for call in calls))


if __name__ == '__main__':
    unittest.main()
