from __future__ import annotations

import contextlib
import io
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import ci_ubuntu_apt as apt


class PrivateAptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.host = self.root / 'host'
        self.host.mkdir()
        self.os_release = self.host / 'os-release'
        self.os_release.write_bytes(b'ID=ubuntu\nVERSION_ID="24.04"\nVERSION_CODENAME=noble\n')
        self.keyring = self.host / 'ubuntu-archive-keyring.gpg'
        self.keyring.write_bytes((ROOT/'tools/keys/ubuntu-archive-2018.asc').read_bytes())
        self.ambient=self.host/'ambient-ubuntu-archive-keyring.gpg'
        self.ambient.write_bytes(b'UNTRUSTED_PRIVATE_TOKEN')
        self.ambient.chmod(0o666)
        self.keyring.chmod(0o644)
        self.sources = self.host / 'etc/apt/sources.list.d'
        self.sources.mkdir(parents=True)
        (self.sources / 'vendor.sources').write_bytes(b'https://user:PRIVATE_TOKEN@unknown.invalid/\n')
        (self.host / 'etc/apt/apt-mirrors.txt').write_bytes(b'PRIVATE_TOKEN unknown mirror\n')
        (self.host / 'etc/apt/apt.conf').write_bytes(b'HOST_HOOK PRIVATE_TOKEN\n')
        self.private = self.root / 'private'
        self.private.mkdir()
        self.record = self.root / 'calls.jsonl'
        self.binaries = self.root / 'bin'
        self.binaries.mkdir()
        self.fake = self.binaries / 'fake-tool'
        self.fake.write_text('#!' + sys.executable + '\n' + '''
import hashlib, json, os, pathlib, re, sys
record = pathlib.Path(os.environ['RECORD'])
args = sys.argv[1:]
name = os.environ['FAKE_TOOL']
if name == 'apt-get':
    # macOS may inject this public runtime key into a child interpreter.
    assert set(os.environ) <= {'PATH','LC_ALL','APT_CONFIG','RECORD','FAKE_TOOL','FAIL_APT','__CF_USER_TEXT_ENCODING'}
    config = pathlib.Path(os.environ['APT_CONFIG'])
    options = dict(re.findall(r'^([^ ]+) "([^"\\n]*)";$', config.read_text(), re.M))
    private = config.parent
    etc = pathlib.Path(options['Dir::Etc'])
    assert etc == private / 'etc'
    for folder in ['apt.conf.d','sources.list.d','preferences.d','trusted.gpg.d','auth.conf.d']:
        assert list((etc/folder).iterdir()) == []
    for file in ['apt.conf','preferences','trusted.gpg','auth.conf']:
        assert (etc/file).read_bytes() == b''
    text = (etc/options['Dir::Etc::sourcelist']).read_text()
    stanzas = [dict(line.split(': ',1) for line in s.splitlines()) for s in text.strip().split('\\n\\n')]
    assert [s['URIs'] for s in stanzas] == ['https://archive.ubuntu.com/ubuntu/', 'https://security.ubuntu.com/ubuntu/']
    assert [s['Suites'] for s in stanzas] == ['noble noble-updates noble-backports', 'noble-security']
    for stanza in stanzas:
        assert set(stanza) == {'Types','URIs','Suites','Components','Architectures','Signed-By','Check-Valid-Until'}
        assert stanza['Types'] == 'deb' and stanza['Architectures'] == 'amd64'
        assert stanza['Components'] == 'main universe restricted multiverse'
        assert stanza['Check-Valid-Until'] == 'yes'
        parts=stanza['Signed-By'].split()
        assert parts == [str(private/'ubuntu-archive-2018.asc'),'F6ECB3762474EDA9D21B7022871920D1991BC93C']
        assert hashlib.sha256(pathlib.Path(parts[0]).read_bytes()).hexdigest() == '2a3cc57ab6b47626b496a101c29af6dfe54d54d03d613f2326b9f2a30a15c39b'
        assert pathlib.Path(parts[0]).stat().st_mode & 0o777 == 0o644
    for k in ['Dir::State','Dir::Cache','Dir::Log']:
        assert pathlib.Path(options[k]).is_relative_to(private)
    assert options['Dir::State::status'] == '/var/lib/dpkg/status'
    assert options['Dir::Cache::pkgcache'] == options['Dir::Cache::srcpkgcache'] == ''
    assert options['APT::Update::Error-Mode'] == 'any'
    for k in ['APT::Get::Remove','APT::Get::AllowUnauthenticated','Acquire::AllowInsecureRepositories','Acquire::AllowDowngradeToInsecureRepositories','Acquire::https::AllowRedirect']:
        assert options[k] == 'false'
    for k in ['Acquire::https::Verify-Peer','Acquire::https::Verify-Host']:
        assert options[k] == 'true'
    with record.open('a') as f:
        f.write(json.dumps({'args':args,'config':str(config)})+'\\n')
    print('PRIVATE_TOKEN https://user:PRIVATE_TOKEN@unknown.invalid/')
    print('PRIVATE_TOKEN exception output',file=sys.stderr)
    sys.exit(100 if args[0] == os.environ.get('FAIL_APT') else 0)
raise AssertionError('Only fake APT is executable')
''')
        self.fake.chmod(0o755)
        self.real_run = subprocess.run
        self.query_fault = None
        self.fail_apt = ''
        self.query_calls = []
        for name, value in [('OS_RELEASE', self.os_release), ('PINNED_KEYRING', self.keyring),
                            ('TMP_ROOT', str(self.private)), ('ROOT_UID', os.getuid())]:
            patch = mock.patch.object(apt, name, value)
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(apt.os, 'geteuid', return_value=0)
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(apt.subprocess, 'run', side_effect=self.route)
        patch.start(); self.addCleanup(patch.stop)

    def route(self, args, **kwargs):
        # Absolute production commands must pass this gate. Never run real APT,
        # dpkg or sudo even if the proposal accidentally changes its arguments.
        if args[0] == '/usr/bin/apt-get':
            self.assertEqual({'PATH','LC_ALL','APT_CONFIG'}, set(kwargs['env']))
            env = {**kwargs['env'], 'RECORD': str(self.record), 'FAKE_TOOL': 'apt-get',
                   'FAIL_APT': self.fail_apt}
            self.assertEqual(subprocess.DEVNULL, kwargs['stdout'])
            self.assertEqual(subprocess.DEVNULL, kwargs['stderr'])
            result = self.real_run([str(self.fake), *args[1:]],
                **{**kwargs, 'env': env, 'check': False,
                   'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE})
            if result.returncode not in (0, 100):
                raise AssertionError(result.stderr.decode())
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, args,
                                                     result.stdout, result.stderr)
            return result
        self.assertEqual(['/usr/bin/dpkg','--print-architecture'],args)
        self.assertEqual(apt.ENV, kwargs['env'])
        self.query_calls.append(args)
        if self.query_fault:
            return self.query_fault(args)
        outputs = {
            ('/usr/bin/dpkg', '--print-architecture'): b'amd64\n',
        }
        self.assertIn(tuple(args), outputs)
        return subprocess.CompletedProcess(args, 0, outputs[tuple(args)])

    def run_main(self, lane='linux-arm64', *extra):
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['ci_ubuntu_apt.py', lane, *extra]), \
                contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
            code = apt.main()
        self.assertNotIn('PRIVATE_TOKEN', output.getvalue())
        self.assertNotIn('http', output.getvalue())
        return code, output.getvalue()

    def snapshot(self):
        return {str(p.relative_to(self.host)): (p.read_bytes(), p.stat().st_ino,
                p.stat().st_mode, p.stat().st_mtime_ns)
                for p in self.host.rglob('*') if p.is_file()}

    def calls(self):
        return [json.loads(x)['args'] for x in self.record.read_text().splitlines()] if self.record.exists() else []

    def assert_initial_queries_only(self):
        self.assertEqual([
            ['/usr/bin/dpkg', '--print-architecture'],
        ], self.query_calls)

    def assert_failure(self, result, stage, exit_status=None, check=None, mode_predicates=None):
        code, output = result
        self.assertEqual(1, code)
        lines = output.splitlines()
        self.assertEqual(2 if mode_predicates is None else 3, len(lines))
        self.assertEqual(apt.REFUSAL, lines[0])
        prefix = 'Private Ubuntu APT failure stage: '
        self.assertTrue(lines[1].startswith(prefix))
        expected = {'stage': stage}
        if check is not None:
            expected['check'] = check
        if exit_status is not None:
            expected['exit_status'] = exit_status
        self.assertEqual(expected, json.loads(lines[1][len(prefix):]))
        self.assertTrue(output.isascii())
        self.assertLessEqual(len(lines[1]), 128)
        if mode_predicates is not None:
            prefix = 'Private Ubuntu keyring mode predicates: '
            self.assertTrue(lines[2].startswith(prefix))
            self.assertEqual(mode_predicates,json.loads(lines[2][len(prefix):]))
            self.assertTrue(all(type(value) is bool for value in mode_predicates.values()))
            self.assertLessEqual(len(lines[2]),128)

    def assert_refused_without_apt(self, lane='linux-arm64', *extra,
                                  stage='prerequisites', exit_status=None, check=None, mode_predicates=None):
        before = self.snapshot()
        self.assert_failure(self.run_main(lane, *extra), stage, exit_status, check, mode_predicates)
        self.assertEqual([], self.calls())
        self.assertEqual([], list(self.private.iterdir()))
        self.assertEqual(before, self.snapshot())

    def test_private_sources_packages_options_and_host_files(self):
        expected = {
            'kleidiai': [['update'], ['install','-y','ninja-build','gcc-aarch64-linux-gnu',
                                      'g++-aarch64-linux-gnu','qemu-user']],
            'linux-arm64': [['update'], ['install','-y','ninja-build'],
                            ['install','-y','gcc-aarch64-linux-gnu','g++-aarch64-linux-gnu','qemu-user','llvm']],
        }
        for lane, commands in expected.items():
            with self.subTest(lane=lane):
                self.record.unlink(missing_ok=True)
                self.query_calls.clear()
                before = self.snapshot()
                with mock.patch.dict(os.environ, {'APT_CONFIG':'PRIVATE_TOKEN',
                        'https_proxy':'https://PRIVATE_TOKEN@unknown.invalid',
                        'DPKG_ROOT':'PRIVATE_TOKEN'}):
                    self.assertEqual((0, ''), self.run_main(lane))
                self.assertEqual(commands, self.calls())
                self.assert_initial_queries_only()
                self.assertEqual(before, self.snapshot())
                self.assertEqual([], list(self.private.iterdir()))

    def test_update_failure_stops_install_and_redacts_child_output(self):
        self.fail_apt = 'update'
        before = self.snapshot()
        self.assert_failure(self.run_main(), 'update', 100)
        self.assertEqual([['update']], self.calls())
        self.assert_initial_queries_only()
        self.assertEqual(before, self.snapshot())
        self.assertEqual([], list(self.private.iterdir()))

    def test_install_failure_stops_later_commands(self):
        self.fail_apt = 'install'
        self.assert_failure(self.run_main(), 'install', 100)
        self.assertEqual([['update'], ['install','-y','ninja-build']], self.calls())
        self.assert_initial_queries_only()
        self.assertEqual([], list(self.private.iterdir()))

    def test_unknown_lane_and_extra_arguments_refuse(self):
        self.assert_refused_without_apt('PRIVATE_TOKEN', stage='arguments')
        self.assert_refused_without_apt('linux-arm64', '--source=PRIVATE_TOKEN', stage='arguments')
        self.assertEqual([], self.query_calls)

    def test_wrong_or_malformed_os_refuses(self):
        fixtures = (
            (b'ID=debian\nVERSION_ID="24.04"\nVERSION_CODENAME=noble\n', 'os_id'),
            (b'ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\n', 'os_version'),
            (b'ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=PRIVATE_TOKEN\n', 'os_codename'),
            (b'ID=ubuntu\nID=ubuntu\n', 'os_duplicate_key'),
            (b'BROKEN_PRIVATE_TOKEN\n', 'os_assignment'),
            (b'ID="PRIVATE_TOKEN\n', 'os_value_syntax'),
            (b'ID=ubun"tu"\nVERSION_ID=24.04\nVERSION_CODENAME=noble\n', 'os_value_syntax'),
            (b'ID=ubuntu\x00PRIVATE_TOKEN\n', 'os_encoding'),
            (b'\n'*129, 'os_line_count'),
            (b'x'*8193, 'os_file_size'),
        )
        for content, check in fixtures:
            with self.subTest(check=check):
                self.os_release.write_bytes(content)
                self.assert_refused_without_apt(check=check)
                self.assertEqual([], self.query_calls)

    def test_conventional_os_release_link_only(self):
        vendor = self.host / 'vendor-os-release'
        vendor.write_bytes(self.os_release.read_bytes())
        self.os_release.unlink()
        self.os_release.symlink_to('/usr/lib/os-release')
        with mock.patch.object(apt, 'VENDOR_RELEASE', vendor):
            self.assertEqual((0, ''), self.run_main())
        self.record.unlink()
        self.os_release.unlink()
        self.os_release.symlink_to('PRIVATE_TOKEN')
        self.assert_refused_without_apt(check='os_link')

    def test_wrong_file_owner_refuses_before_apt(self):
        with mock.patch.object(apt, 'ROOT_UID', os.getuid() + 1):
            self.assert_refused_without_apt(check='os_file_owner')

    def test_wrong_architecture_still_refuses(self):
        self.query_fault=lambda args: subprocess.CompletedProcess(args,0,b'arm64\n')
        self.assert_refused_without_apt(check='host_architecture')

    def test_pinned_source_missing_size_mode_symlink_fifo_refuse(self):
        original=self.keyring.read_bytes()
        for content,mode in ((b'',0o644),(original+b'x',0o644),(original,0o666),(original,0o600)):
            self.keyring.write_bytes(content);self.keyring.chmod(mode)
            self.assert_refused_without_apt(check='pinned_keyring_read')
        self.keyring.unlink()
        self.assert_refused_without_apt(check='pinned_keyring_read')
        self.keyring.symlink_to(self.ambient)
        self.assert_refused_without_apt(check='pinned_keyring_read')
        self.keyring.unlink();os.mkfifo(self.keyring)
        self.assert_failure(self.run_main(),'prerequisites',check='pinned_keyring_read')
        self.assertEqual([],self.calls())

    def test_pinned_source_tampered_or_unknown_content_refuses_before_apt(self):
        original=self.keyring.read_bytes()
        for content in (b'X'+original[1:],b'?'*1660):
            self.keyring.write_bytes(content)
            self.assert_refused_without_apt(check='pinned_keyring_content')
            self.query_calls.clear()

    def test_os_read_failure_and_query_failure_redact(self):
        with mock.patch.object(apt.os,'read',side_effect=OSError('PRIVATE_TOKEN')):
            self.assert_refused_without_apt(check='os_read')
        def fail(args):
            raise subprocess.CalledProcessError(1,args,output=b'PRIVATE_TOKEN')
        self.query_fault=fail
        self.assert_refused_without_apt(exit_status=1, check='host_architecture_query')

    def test_configuration_failure_has_no_package_call_or_host_write(self):
        with mock.patch.object(apt, 'configure', side_effect=ValueError('PRIVATE_TOKEN /private/path')):
            self.assert_refused_without_apt(stage='private_configuration')
        self.assert_initial_queries_only()

    def test_cleanup_failure_has_configuration_stage_and_no_fallback(self):
        real_temp = apt.tempfile.TemporaryDirectory
        @contextlib.contextmanager
        def fail_cleanup(*args, **kwargs):
            with real_temp(*args, **kwargs) as directory:
                yield directory
            raise OSError('PRIVATE_TOKEN /private/path')
        before = self.snapshot()
        with mock.patch.object(apt.tempfile, 'TemporaryDirectory', side_effect=fail_cleanup):
            self.assert_failure(self.run_main(), 'private_configuration')
        self.assertEqual([['update'], ['install','-y','ninja-build'],
                          ['install','-y','gcc-aarch64-linux-gnu','g++-aarch64-linux-gnu','qemu-user','llvm']],
                         self.calls())
        self.assert_initial_queries_only()
        self.assertEqual(before, self.snapshot())
        self.assertEqual([], list(self.private.iterdir()))

    def test_only_bounded_integer_exit_status_is_reported(self):
        for status, expected in ((1,1), (100,100), (255,255), (0,None),
                                 (-9,None), (256,None), (True,None), ('PRIVATE_TOKEN',None)):
            with self.subTest(status=status):
                self.query_calls.clear()
                def fail(args):
                    raise subprocess.CalledProcessError(status, ['/PRIVATE_TOKEN'],
                        output=b'PRIVATE_TOKEN https://private.invalid', stderr=b'PRIVATE_TOKEN')
                self.query_fault = fail
                self.assert_refused_without_apt(exit_status=expected, check='host_architecture_query')
                self.assertEqual([['/usr/bin/dpkg','--print-architecture']], self.query_calls)

    def test_timeout_stage_does_not_echo_child_output_or_request_retry(self):
        def fail(args):
            raise subprocess.TimeoutExpired('/PRIVATE_TOKEN',30,
                output=b'PRIVATE_TOKEN https://private.invalid', stderr=b'PRIVATE_TOKEN')
        self.query_fault = fail
        self.assert_refused_without_apt(check='host_architecture_query')
        self.assertEqual([['/usr/bin/dpkg','--print-architecture']], self.query_calls)

    def test_os_file_type_mode_and_empty_size_refuse_without_queries(self):
        valid = self.os_release.read_bytes()
        for content, mode, check in ((b'',0o644,'os_file_size'),
                                     (valid,0o666,'os_file_mode')):
            self.os_release.write_bytes(content); self.os_release.chmod(mode)
            self.assert_refused_without_apt(check=check)
        self.os_release.unlink(); self.os_release.mkdir()
        self.assert_refused_without_apt(check='os_file_type')
        self.assertEqual([], self.query_calls)

    def test_file_stability_checks_preserve_read_count_and_host_files(self):
        real_read = apt.os.read
        for failed_read, check in ((1,'os_file_stability'), (2,'pinned_keyring_read')):
            with self.subTest(check=check):
                self.query_calls.clear()
                calls = []
                def short_read(fd, limit):
                    data = real_read(fd, limit)
                    calls.append(limit)
                    return data[:-1] if len(calls) == failed_read else data
                with mock.patch.object(apt.os,'read',side_effect=short_read):
                    self.assert_refused_without_apt(check=check)
                self.assertEqual([8193,1661][:failed_read], calls)
                if failed_read == 1:
                    self.assertEqual([], self.query_calls)
                else:
                    self.assert_initial_queries_only()

    def test_generic_root_file_owner_guard_remains_strict(self):
        actual=self.keyring.stat()
        metadata=SimpleNamespace(st_mode=actual.st_mode,st_uid=apt.ROOT_UID+1,
            st_size=actual.st_size,st_mtime_ns=actual.st_mtime_ns,st_ctime_ns=actual.st_ctime_ns)
        with mock.patch.object(apt.os,'fstat',return_value=metadata), \
                mock.patch.object(apt.os,'read',side_effect=AssertionError('unsafe read')):
            with self.assertRaises(apt._PhaseFailure) as failed:
                apt.root_file(self.keyring,1024*1024,public=True)
        self.assertEqual('keyring_file_owner',failed.exception.check.value)
        self.assertEqual([],self.query_calls)

    def test_os_value_arity_and_parser_error_refuse_before_queries(self):
        with mock.patch.object(apt.shlex,'split',return_value=[]):
            self.assert_refused_without_apt(check='os_value_arity')
        with mock.patch.object(apt.shlex,'split',side_effect=ValueError('PRIVATE_TOKEN')):
            self.assert_refused_without_apt(check='os_value_syntax')
        self.assertEqual([],self.query_calls)

    def test_architecture_query_failure_and_output_limit_still_refuse(self):
        for overflow in (False,True):
            self.query_calls.clear()
            def fail(args):
                if overflow:return subprocess.CompletedProcess(args,0,b'PRIVATE_TOKEN'*400)
                raise subprocess.CalledProcessError(1,['PRIVATE_TOKEN'],output=b'PRIVATE_TOKEN',stderr=b'PRIVATE_TOKEN')
            self.query_fault=fail
            self.assert_refused_without_apt(check='host_architecture_output_limit' if overflow else 'host_architecture_query',
                                           exit_status=None if overflow else 1)
            self.assert_initial_queries_only()

    def test_accepted_prerequisites_keep_exact_file_io_and_query_contract(self):
        before = self.snapshot()
        expected = self.keyring.read_bytes()
        with mock.patch.object(apt.os,'open',wraps=apt.os.open) as opened, \
                mock.patch.object(apt.os,'fstat',wraps=apt.os.fstat) as stated, \
                mock.patch.object(apt.os,'read',wraps=apt.os.read) as read, \
                mock.patch.object(apt.os,'close',wraps=apt.os.close) as closed:
            self.assertEqual(expected,apt.prerequisites())
        self.assertEqual(2,opened.call_count)
        self.assertEqual(4,stated.call_count)
        self.assertEqual([8193,1661],[call.args[1] for call in read.call_args_list])
        self.assertEqual(2,closed.call_count)
        self.assert_initial_queries_only()
        self.assertEqual(before,self.snapshot())
        self.assertEqual([],self.calls())
        self.assertEqual([],list(self.private.iterdir()))

    def test_mode_policy_truth_table_is_unchanged_for_all_permission_combinations(self):
        # Synthetic POSIX policy controls, never a claim about the failed runner.
        actual = self.keyring.stat()
        accepted = 0
        for permissions in range(512):
            expected = (bool(permissions & stat.S_IWGRP),
                        bool(permissions & stat.S_IWOTH),
                        not bool(permissions & stat.S_IROTH))
            metadata = SimpleNamespace(st_mode=stat.S_IFREG|permissions,st_uid=actual.st_uid,
                st_size=actual.st_size,st_mtime_ns=actual.st_mtime_ns,st_ctime_ns=actual.st_ctime_ns)
            with self.subTest(permissions=permissions), \
                    mock.patch.object(apt.os,'fstat',return_value=metadata) as stated, \
                    mock.patch.object(apt.os,'read',wraps=apt.os.read) as read:
                if any(expected):
                    with self.assertRaises(apt._PhaseFailure) as failed:
                        apt.root_file(self.keyring,1024*1024,public=True)
                    self.assertEqual('keyring_file_mode',failed.exception.check.value)
                    self.assertEqual(expected,failed.exception.keyring_mode)
                    self.assertEqual(1,stated.call_count)
                    read.assert_not_called()
                else:
                    self.assertEqual((ROOT/'tools/keys/ubuntu-archive-2018.asc').read_bytes(),
                                     apt.root_file(self.keyring,1024*1024,public=True))
                    self.assertEqual(2,stated.call_count)
                    self.assertEqual(1,read.call_count)
                    accepted += 1
        self.assertEqual(64,accepted)
        self.assertEqual([],self.query_calls)
        self.assertEqual([],self.calls())

    def test_ambient_keyring_is_never_read_even_when_it_matches(self):
        self.ambient.write_bytes(self.keyring.read_bytes())
        real_open=apt.os.open
        def reject_ambient(path,*args,**kwargs):
            self.assertNotEqual(str(self.ambient),str(path))
            return real_open(path,*args,**kwargs)
        before=self.snapshot()
        with mock.patch.object(apt.os,'open',side_effect=reject_ambient):
            self.assertEqual((0,''),self.run_main())
        self.assert_initial_queries_only()
        self.assertEqual(before,self.snapshot())

    def test_missing_pin_does_not_fall_back_to_matching_ambient_key(self):
        self.ambient.write_bytes(self.keyring.read_bytes())
        self.keyring.unlink()
        self.assert_refused_without_apt(check='pinned_keyring_read')
        self.assert_initial_queries_only()

    def test_source_race_and_read_error_do_not_reach_apt(self):
        real_stat=apt.os.fstat
        count=[]
        def changed(fd):
            result=real_stat(fd);count.append(result)
            if len(count)==4:
                fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')
                data={name:getattr(result,name)for name in fields};data['st_ctime_ns']+=1
                return SimpleNamespace(**data)
            return result
        with mock.patch.object(apt.os,'fstat',side_effect=changed):
            self.assert_refused_without_apt(check='pinned_keyring_read')
        self.assertEqual(4,len(count))
        self.assert_initial_queries_only()
        self.query_calls.clear()
        real_read=apt.os.read;reads=[]
        def unreadable(fd,limit):
            reads.append(limit)
            if len(reads)==2:raise OSError('PRIVATE_TOKEN /unknown/key')
            return real_read(fd,limit)
        with mock.patch.object(apt.os,'read',side_effect=unreadable):
            self.assert_refused_without_apt(check='pinned_keyring_read')
        self.assertEqual([8193,1661],reads)
        self.assert_initial_queries_only()

    def test_content_guard_bypass_is_caught_by_independent_private_key_oracle(self):
        unknown=b'?'*1660
        self.keyring.write_bytes(unknown)
        with mock.patch.object(apt,'pinned_keyring',return_value=unknown):
            with self.assertRaises(AssertionError):
                self.assert_refused_without_apt(check='pinned_keyring_content')

    def test_extra_package_command_mutation_is_rejected(self):
        with mock.patch.dict(apt.COMMANDS,{'linux-arm64':apt.COMMANDS['linux-arm64']+
                             (('install','-y','unapproved-package'),)}):
            with self.assertRaises(AssertionError):
                self.assertEqual((0,''), self.run_main())
                self.assertEqual([['update'], ['install','-y','ninja-build'],
                    ['install','-y','gcc-aarch64-linux-gnu','g++-aarch64-linux-gnu','qemu-user','llvm']],
                    self.calls())

    def test_nonroot_refuses_before_read_or_commands(self):
        with mock.patch.object(apt.os,'geteuid',return_value=123), \
                mock.patch.object(apt.os,'open',side_effect=AssertionError('read')):
            self.assert_refused_without_apt(check='root')
        self.assertEqual([],self.query_calls)



    def test_population_stays_private_until_files_have_final_modes(self):
        for mask in (0o000, 0o022, 0o077):
            with self.subTest(umask=mask):
                work=self.private/('publication-'+str(mask));work.mkdir(mode=0o700)
                held=[];published_during_population=[];written={}
                original_text=Path.write_text;original_bytes=Path.write_bytes
                def after_write(path):
                    if work in path.parents:
                        if stat.S_IMODE(work.stat().st_mode)&0o077:
                            published_during_population.append(path.name)
                        if path.name=='bootstrap.conf':
                            written['config']=path.read_text()
                            # Model an other-user traversal/open opportunity using
                            # captured permissions, then exercise held-FD behavior.
                            if (work.stat().st_mode&0o001 and path.stat().st_mode&0o002):
                                held.append(path.open('a'))
                def text(path,data,*args,**kwargs):
                    result=original_text(path,data,*args,**kwargs);after_write(path);return result
                def binary(path,data,*args,**kwargs):
                    result=original_bytes(path,data,*args,**kwargs);after_write(path);return result
                previous=os.umask(mask)
                try:
                    with mock.patch.object(Path,'write_text',text),mock.patch.object(Path,'write_bytes',binary):
                        config=apt.configure(work,self.keyring.read_bytes())
                    for stream in held:
                        stream.write('\nAPT::Get::AllowUnauthenticated "true";\n');stream.flush()
                    self.assertEqual(written['config'],config.read_text(),
                                     'Held descriptor changed published configuration')
                    self.assertEqual([],published_during_population)
                    self.assertEqual([],held)
                    self.assertEqual(0o755,stat.S_IMODE(work.stat().st_mode))
                    for item in work.rglob('*'):
                        self.assertEqual(0o755 if item.is_dir() else 0o644,
                                         stat.S_IMODE(item.stat().st_mode))
                finally:
                    os.umask(previous)
                    for stream in held:stream.close()

    def test_population_failure_keeps_private_boundary(self):
        work=self.private/'failed-publication';work.mkdir(mode=0o700)
        original=Path.write_text
        def fail(path,data,*args,**kwargs):
            if path.name=='bootstrap.conf':raise OSError('offline publication fault')
            return original(path,data,*args,**kwargs)
        previous=os.umask(0)
        try:
            with mock.patch.object(Path,'write_text',fail),self.assertRaises(OSError):
                apt.configure(work,self.keyring.read_bytes())
            self.assertEqual(0o700,stat.S_IMODE(work.stat().st_mode))
            self.assertEqual([],self.calls())
        finally:
            os.umask(previous)


if __name__ == '__main__':
    unittest.main()
