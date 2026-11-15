#!/usr/bin/env python3
"""Opt-in CI policy: private Noble sources; never inspect host APT sources.

Proposed for disposable amd64 Ubuntu 24.04 jobs only. Not a mirror fallback.
"""
from __future__ import annotations

import contextlib
from enum import Enum
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile

OS_RELEASE = Path('/etc/os-release')
VENDOR_RELEASE = Path('/usr/lib/os-release')
KEYRING = Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg')
TMP_ROOT = '/tmp'
ROOT_UID = 0
ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C'}
REFUSAL = 'Private Ubuntu APT refused or failed; no host-source fallback.'
COMMANDS = {
    'kleidiai': (
        ('update',),
        ('install', '-y', 'ninja-build', 'gcc-aarch64-linux-gnu',
         'g++-aarch64-linux-gnu', 'qemu-user'),
    ),
    'linux-arm64': (
        ('update',),
        ('install', '-y', 'ninja-build'),
        ('install', '-y', 'gcc-aarch64-linux-gnu', 'g++-aarch64-linux-gnu',
         'qemu-user', 'llvm'),
    ),
}


class FailureStage(Enum):
    ARGUMENTS = 'arguments'
    PREREQUISITES = 'prerequisites'
    PRIVATE_CONFIGURATION = 'private_configuration'
    UPDATE = 'update'
    INSTALL = 'install'


class PrerequisiteCheck(Enum):
    ROOT = 'root'
    OS_LINK = 'os_link'
    OS_READ = 'os_read'
    OS_FILE_TYPE = 'os_file_type'
    OS_FILE_OWNER = 'os_file_owner'
    OS_FILE_MODE = 'os_file_mode'
    OS_FILE_SIZE = 'os_file_size'
    OS_FILE_STABILITY = 'os_file_stability'
    OS_ENCODING = 'os_encoding'
    OS_LINE_COUNT = 'os_line_count'
    OS_ASSIGNMENT = 'os_assignment'
    OS_DUPLICATE_KEY = 'os_duplicate_key'
    OS_VALUE_SYNTAX = 'os_value_syntax'
    OS_VALUE_ARITY = 'os_value_arity'
    OS_ID = 'os_id'
    OS_VERSION = 'os_version'
    OS_CODENAME = 'os_codename'
    HOST_ARCHITECTURE_QUERY = 'host_architecture_query'
    HOST_ARCHITECTURE_OUTPUT_LIMIT = 'host_architecture_output_limit'
    HOST_ARCHITECTURE = 'host_architecture'
    KEYRING_PACKAGE_QUERY = 'keyring_package_query'
    KEYRING_PACKAGE_OUTPUT_LIMIT = 'keyring_package_output_limit'
    KEYRING_PACKAGE_ATTRIBUTION = 'keyring_package_attribution'
    KEYRING_INTEGRITY_QUERY = 'keyring_integrity_query'
    KEYRING_INTEGRITY_OUTPUT_LIMIT = 'keyring_integrity_output_limit'
    KEYRING_INTEGRITY = 'keyring_integrity'
    KEYRING_READ = 'keyring_read'
    KEYRING_FILE_TYPE = 'keyring_file_type'
    KEYRING_FILE_OWNER = 'keyring_file_owner'
    KEYRING_FILE_MODE = 'keyring_file_mode'
    KEYRING_FILE_SIZE = 'keyring_file_size'
    KEYRING_FILE_STABILITY = 'keyring_file_stability'


class _PhaseFailure(Exception):
    def __init__(self, stage: FailureStage, exit_status: int | None,
                 check: PrerequisiteCheck | None = None):
        self.stage = stage
        self.exit_status = exit_status
        self.check = check


@contextlib.contextmanager
def _failure_stage(stage: FailureStage, check: PrerequisiteCheck | None = None):
    try:
        yield
    except _PhaseFailure:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        status = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        if type(status) is not int or not 1 <= status <= 255:
            status = None
        # Never copy child output, commands, paths or exception text to the receipt.
        raise _PhaseFailure(stage, status, check) from None


def _prerequisite_check(check: PrerequisiteCheck):
    return _failure_stage(FailureStage.PREREQUISITES, check)


def root_file(path: Path, limit: int, *, public: bool = False) -> bytes:
    # Only the keyring requires public readability. These are fixed labels for
    # existing failed predicates, using the captured stat data, never new IO.
    checks = (
        (PrerequisiteCheck.KEYRING_FILE_TYPE, PrerequisiteCheck.KEYRING_FILE_OWNER,
         PrerequisiteCheck.KEYRING_FILE_MODE, PrerequisiteCheck.KEYRING_FILE_SIZE,
         PrerequisiteCheck.KEYRING_FILE_STABILITY) if public else
        (PrerequisiteCheck.OS_FILE_TYPE, PrerequisiteCheck.OS_FILE_OWNER,
         PrerequisiteCheck.OS_FILE_MODE, PrerequisiteCheck.OS_FILE_SIZE,
         PrerequisiteCheck.OS_FILE_STABILITY)
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != ROOT_UID or
                before.st_mode & 0o022 or not 0 < before.st_size <= limit or
                (public and not before.st_mode & 0o004)):
            # Preserve the original short-circuit predicate order on refusal.
            if not stat.S_ISREG(before.st_mode):
                check = checks[0]
            elif before.st_uid != ROOT_UID:
                check = checks[1]
            elif before.st_mode & 0o022:
                check = checks[2]
            elif not 0 < before.st_size <= limit:
                check = checks[3]
            else:
                check = checks[2]
            raise _PhaseFailure(FailureStage.PREREQUISITES, None, check)
        data = os.read(fd, limit + 1)
        after = os.fstat(fd)
        if (len(data) != before.st_size or len(data) > limit or
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise _PhaseFailure(FailureStage.PREREQUISITES, None, checks[4])
        return data
    finally:
        os.close(fd)


def query(args: list[str], *, output_check: PrerequisiteCheck) -> bytes:
    result = subprocess.run(args, env=ENV, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=30, check=True)
    if len(result.stdout) > 4096:
        raise _PhaseFailure(FailureStage.PREREQUISITES, None, output_check)
    return result.stdout


def prerequisites() -> bytes:
    with _prerequisite_check(PrerequisiteCheck.ROOT):
        if os.geteuid() != 0:
            raise ValueError()
    # Honor /etc precedence. Permit only the conventional vendor-file link;
    # never follow an arbitrary OS-release link or source it as shell code.
    release = OS_RELEASE
    with _prerequisite_check(PrerequisiteCheck.OS_LINK):
        if OS_RELEASE.is_symlink():
            if os.readlink(OS_RELEASE) not in ('../usr/lib/os-release', '/usr/lib/os-release'):
                raise ValueError()
            release = VENDOR_RELEASE
    with _prerequisite_check(PrerequisiteCheck.OS_READ):
        data = root_file(release, 8192)
    with _prerequisite_check(PrerequisiteCheck.OS_ENCODING):
        if any(byte not in range(32, 127) and byte != 10 for byte in data):
            raise ValueError()
        fields = {}
        lines = data.decode('ascii').splitlines()
    with _prerequisite_check(PrerequisiteCheck.OS_LINE_COUNT):
        if len(lines) > 128:
            raise ValueError()
    for line in lines:
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Z][A-Z0-9_]*)=(.*)', line)
        if not match or match[1] in fields:
            check = PrerequisiteCheck.OS_ASSIGNMENT if not match else PrerequisiteCheck.OS_DUPLICATE_KEY
            raise _PhaseFailure(FailureStage.PREREQUISITES, None, check)
        # One quoted or plain value only; no concatenated shell fragments.
        with _prerequisite_check(PrerequisiteCheck.OS_VALUE_SYNTAX):
            if not re.fullmatch(r'''(?:"(?:[^"\\$`]|\\["\\$`])*"|'[^']*'|[A-Za-z0-9._-]+)''', match[2]):
                raise ValueError()
            values = shlex.split(match[2])
        with _prerequisite_check(PrerequisiteCheck.OS_VALUE_ARITY):
            if len(values) != 1:
                raise ValueError()
        fields[match[1]] = values[0]
    if any(fields.get(k) != v for k, v in
           {'ID': 'ubuntu', 'VERSION_ID': '24.04', 'VERSION_CODENAME': 'noble'}.items()):
        if fields.get('ID') != 'ubuntu':
            check = PrerequisiteCheck.OS_ID
        elif fields.get('VERSION_ID') != '24.04':
            check = PrerequisiteCheck.OS_VERSION
        else:
            check = PrerequisiteCheck.OS_CODENAME
        raise _PhaseFailure(FailureStage.PREREQUISITES, None, check)
    with _prerequisite_check(PrerequisiteCheck.HOST_ARCHITECTURE_QUERY):
        architecture = query(['/usr/bin/dpkg', '--print-architecture'],
                             output_check=PrerequisiteCheck.HOST_ARCHITECTURE_OUTPUT_LIMIT)
    with _prerequisite_check(PrerequisiteCheck.HOST_ARCHITECTURE):
        if architecture != b'amd64\n':
            raise ValueError()
    with _prerequisite_check(PrerequisiteCheck.KEYRING_PACKAGE_QUERY):
        attribution = query(['/usr/bin/dpkg-query', '-S', str(KEYRING)],
                            output_check=PrerequisiteCheck.KEYRING_PACKAGE_OUTPUT_LIMIT)
    with _prerequisite_check(PrerequisiteCheck.KEYRING_PACKAGE_ATTRIBUTION):
        if attribution != ('ubuntu-keyring: ' + str(KEYRING) + '\n').encode('ascii'):
            raise ValueError()
    # Installed-package integrity check, not a new trust root or cryptographic
    # provenance claim. Require no discrepancy output as well as exit success.
    with _prerequisite_check(PrerequisiteCheck.KEYRING_INTEGRITY_QUERY):
        integrity = query(['/usr/bin/dpkg', '--verify', 'ubuntu-keyring'],
                          output_check=PrerequisiteCheck.KEYRING_INTEGRITY_OUTPUT_LIMIT)
    with _prerequisite_check(PrerequisiteCheck.KEYRING_INTEGRITY):
        if integrity:
            raise ValueError()
    with _prerequisite_check(PrerequisiteCheck.KEYRING_READ):
        return root_file(KEYRING, 1024 * 1024, public=True)


def configure(work: Path, keyring: bytes) -> Path:
    # Paths are generated internally, never taken from CLI or environment.
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(work)) or '..' in work.parts:
        raise ValueError()
    work.chmod(0o755)  # _apt must be able to traverse to public config/keyring.
    etc = work / 'etc'
    for name in ('etc/apt.conf.d', 'etc/sources.list.d', 'etc/preferences.d',
                 'etc/trusted.gpg.d', 'etc/auth.conf.d', 'state/lists/partial',
                 'cache/archives/partial', 'log'):
        (work / name).mkdir(parents=True, exist_ok=True, mode=0o755)
    for name in ('apt.conf', 'preferences', 'trusted.gpg', 'auth.conf'):
        (etc / name).write_bytes(b'')
    private_keyring = work / 'ubuntu-archive-keyring.gpg'
    private_keyring.write_bytes(keyring)
    private_keyring.chmod(0o644)
    stanzas = []
    for uri, suites in (
        ('https://archive.ubuntu.com/ubuntu/', 'noble noble-updates noble-backports'),
        ('https://security.ubuntu.com/ubuntu/', 'noble-security'),
    ):
        stanzas.append(f'Types: deb\nURIs: {uri}\nSuites: {suites}\n'
                       'Components: main universe restricted multiverse\n'
                       f'Architectures: amd64\nSigned-By: {private_keyring}\n'
                       'Check-Valid-Until: yes\n')
    (etc / 'ubuntu.sources').write_text('\n'.join(stanzas), encoding='ascii')
    options = {
        'Dir::Etc': str(etc), 'Dir::Etc::sourcelist': 'ubuntu.sources',
        'Dir::State': str(work / 'state'), 'Dir::State::status': '/var/lib/dpkg/status',
        'Dir::Cache': str(work / 'cache'), 'Dir::Cache::pkgcache': '',
        'Dir::Cache::srcpkgcache': '', 'Dir::Log': str(work / 'log'),
        'APT::Update::Error-Mode': 'any', 'APT::Get::Remove': 'false',
        'APT::Get::AllowUnauthenticated': 'false',
        'Acquire::AllowInsecureRepositories': 'false',
        'Acquire::AllowDowngradeToInsecureRepositories': 'false',
        'Acquire::https::Verify-Peer': 'true', 'Acquire::https::Verify-Host': 'true',
        'Acquire::https::AllowRedirect': 'false',
    }
    config = work / 'bootstrap.conf'
    config.write_text(''.join(f'{k} "{v}";\n' for k, v in options.items()), encoding='ascii')
    # The process umask must not make the public configuration unreadable to _apt.
    for path in work.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return config


def main() -> int:
    try:
        with _failure_stage(FailureStage.ARGUMENTS):
            if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
                raise ValueError()
        with _failure_stage(FailureStage.PREREQUISITES):
            keyring = prerequisites()
        # This scope also covers private directory creation and cleanup.
        with _failure_stage(FailureStage.PRIVATE_CONFIGURATION):
            with tempfile.TemporaryDirectory(prefix='llamadart-ci-apt-', dir=TMP_ROOT) as temp:
                config = configure(Path(temp), keyring)
                for args in COMMANDS[sys.argv[1]]:
                    stage = FailureStage.UPDATE if args[0] == 'update' else FailureStage.INSTALL
                    with _failure_stage(stage):
                        # APT_CONFIG loads before Dir::Etc defaults. No inherited APT,
                        # proxy, credential or hook environment; no -c-only isolation.
                        subprocess.run(['/usr/bin/apt-get', *args],
                                       env={**ENV, 'APT_CONFIG': str(config)},
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, check=True)
        return 0
    except _PhaseFailure as failure:
        print(REFUSAL, file=sys.stderr)
        receipt = {'stage': failure.stage.value}
        if failure.check is not None:
            receipt['check'] = failure.check.value
        if failure.exit_status is not None:
            receipt['exit_status'] = failure.exit_status
        print('Private Ubuntu APT failure stage: ' +
              json.dumps(receipt, sort_keys=True, separators=(',', ':')), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
