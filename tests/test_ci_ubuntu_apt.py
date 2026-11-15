from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
        self.keyring.write_bytes(b'FAKE_PACKAGED_KEYRING_NOT_A_REAL_KEY')
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
import json, os, pathlib, re, sys
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
        assert pathlib.Path(stanza['Signed-By']) == private/'ubuntu-archive-keyring.gpg'
        assert pathlib.Path(stanza['Signed-By']).read_bytes() == b'FAKE_PACKAGED_KEYRING_NOT_A_REAL_KEY'
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
        for name, value in [('OS_RELEASE', self.os_release), ('KEYRING', self.keyring),
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
        self.assertIn(args[0], ('/usr/bin/dpkg', '/usr/bin/dpkg-query'))
        self.assertEqual(apt.ENV, kwargs['env'])
        self.query_calls.append(args)
        if self.query_fault:
            return self.query_fault(args)
        outputs = {
            ('/usr/bin/dpkg', '--print-architecture'): b'amd64\n',
            ('/usr/bin/dpkg-query', '-S', str(self.keyring)):
                ('ubuntu-keyring: ' + str(self.keyring) + '\n').encode(),
            ('/usr/bin/dpkg', '--verify', 'ubuntu-keyring'): b'',
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

    def assert_refused_without_apt(self, lane='linux-arm64', *extra):
        before = self.snapshot()
        self.assertEqual((1, apt.REFUSAL+'\n'), self.run_main(lane, *extra))
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
                before = self.snapshot()
                with mock.patch.dict(os.environ, {'APT_CONFIG':'PRIVATE_TOKEN',
                        'https_proxy':'https://PRIVATE_TOKEN@unknown.invalid',
                        'DPKG_ROOT':'PRIVATE_TOKEN'}):
                    self.assertEqual((0, ''), self.run_main(lane))
                self.assertEqual(commands, self.calls())
                self.assertEqual(before, self.snapshot())
                self.assertEqual([], list(self.private.iterdir()))

    def test_update_failure_stops_install_and_redacts_child_output(self):
        self.fail_apt = 'update'
        before = self.snapshot()
        self.assertEqual((1, apt.REFUSAL+'\n'), self.run_main())
        self.assertEqual([['update']], self.calls())
        self.assertEqual(before, self.snapshot())
        self.assertEqual([], list(self.private.iterdir()))

    def test_install_failure_stops_later_commands(self):
        self.fail_apt = 'install'
        self.assertEqual((1, apt.REFUSAL+'\n'), self.run_main())
        self.assertEqual([['update'], ['install','-y','ninja-build']], self.calls())
        self.assertEqual([], list(self.private.iterdir()))

    def test_unknown_lane_and_extra_arguments_refuse(self):
        self.assert_refused_without_apt('PRIVATE_TOKEN')
        self.assert_refused_without_apt('linux-arm64', '--source=PRIVATE_TOKEN')

    def test_wrong_or_malformed_os_refuses(self):
        for content in (b'ID=debian\nVERSION_ID="24.04"\nVERSION_CODENAME=noble\n',
                        b'ID=ubuntu\nVERSION_ID=22.04\nVERSION_CODENAME=jammy\n',
                        b'ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=PRIVATE_TOKEN\n',
                        b'ID=ubuntu\nID=ubuntu\n', b'ID="PRIVATE_TOKEN\n',
                        b'ID=ubun"tu"\nVERSION_ID=24.04\nVERSION_CODENAME=noble\n',
                        b'ID=ubuntu\x00PRIVATE_TOKEN\n', b'x'*8193):
            self.os_release.write_bytes(content)
            self.assert_refused_without_apt()

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
        self.assert_refused_without_apt()

    def test_wrong_file_owner_refuses_before_apt(self):
        with mock.patch.object(apt, 'ROOT_UID', os.getuid() + 1):
            self.assert_refused_without_apt()

    def test_wrong_architecture_package_owner_or_keyring_integrity_refuses(self):
        real_route = self.route
        for target, replacement in (
            ('--print-architecture', b'arm64\n'),
            ('-S', b'other-package: PRIVATE_TOKEN\n'),
            ('--verify', b'??5?????? PRIVATE_TOKEN\n')):
            def fault(args, target=target, replacement=replacement):
                if args[1] == target:
                    return subprocess.CompletedProcess(args, 0, replacement)
                self.query_fault = None
                result = real_route(args, env=apt.ENV)
                self.query_fault = fault
                return result
            self.query_fault = fault
            self.assert_refused_without_apt()
        self.query_fault = None

    def test_missing_empty_unreadable_or_writable_keyring_refuses(self):
        for content, mode in ((b'',0o644), (b'x',0o600), (b'x',0o666), (b'x'*1048577,0o644)):
            self.keyring.write_bytes(content); self.keyring.chmod(mode)
            self.assert_refused_without_apt()
        self.keyring.unlink()
        self.assert_refused_without_apt()

    def test_symlink_and_fifo_keyring_refuse_without_read(self):
        self.keyring.unlink()
        self.keyring.symlink_to(self.os_release)
        self.assert_refused_without_apt()
        self.keyring.unlink(); os.mkfifo(self.keyring)
        # snapshot deliberately excludes FIFO; the read itself must not block.
        self.assertEqual((1,apt.REFUSAL+'\n'), self.run_main())
        self.assertEqual([], self.calls())

    def test_os_read_failure_and_query_failure_redact(self):
        with mock.patch.object(apt.os,'read',side_effect=OSError('PRIVATE_TOKEN')):
            self.assert_refused_without_apt()
        def fail(args):
            raise subprocess.CalledProcessError(1,args,output=b'PRIVATE_TOKEN')
        self.query_fault=fail
        self.assert_refused_without_apt()

    def test_nonroot_refuses_before_read_or_commands(self):
        with mock.patch.object(apt.os,'geteuid',return_value=123), \
                mock.patch.object(apt.os,'open',side_effect=AssertionError('read')):
            self.assert_refused_without_apt()
        self.assertEqual([],self.query_calls)


if __name__ == '__main__':
    unittest.main()
