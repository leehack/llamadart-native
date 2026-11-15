#!/usr/bin/env python3
"""Opt-in CI policy: private Noble sources; never inspect host APT sources.

Proposed for disposable amd64 Ubuntu 24.04 jobs only. Not a mirror fallback.
"""
from __future__ import annotations

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


def root_file(path: Path, limit: int, *, public: bool = False) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != ROOT_UID or
                before.st_mode & 0o022 or not 0 < before.st_size <= limit or
                (public and not before.st_mode & 0o004)):
            raise ValueError()
        data = os.read(fd, limit + 1)
        after = os.fstat(fd)
        if (len(data) != before.st_size or len(data) > limit or
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError()
        return data
    finally:
        os.close(fd)


def query(args: list[str]) -> bytes:
    result = subprocess.run(args, env=ENV, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=30, check=True)
    if len(result.stdout) > 4096:
        raise ValueError()
    return result.stdout


def prerequisites() -> bytes:
    if os.geteuid() != 0:
        raise ValueError()
    # Honor /etc precedence. Permit only the conventional vendor-file link;
    # never follow an arbitrary OS-release link or source it as shell code.
    release = OS_RELEASE
    if OS_RELEASE.is_symlink():
        if os.readlink(OS_RELEASE) not in ('../usr/lib/os-release', '/usr/lib/os-release'):
            raise ValueError()
        release = VENDOR_RELEASE
    data = root_file(release, 8192)
    if any(byte not in range(32, 127) and byte != 10 for byte in data):
        raise ValueError()
    fields = {}
    lines = data.decode('ascii').splitlines()
    if len(lines) > 128:
        raise ValueError()
    for line in lines:
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Z][A-Z0-9_]*)=(.*)', line)
        if not match or match[1] in fields:
            raise ValueError()
        # One quoted or plain value only; no concatenated shell fragments.
        if not re.fullmatch(r'''(?:"(?:[^"\\$`]|\\["\\$`])*"|'[^']*'|[A-Za-z0-9._-]+)''', match[2]):
            raise ValueError()
        values = shlex.split(match[2])
        if len(values) != 1:
            raise ValueError()
        fields[match[1]] = values[0]
    if any(fields.get(k) != v for k, v in
           {'ID': 'ubuntu', 'VERSION_ID': '24.04', 'VERSION_CODENAME': 'noble'}.items()):
        raise ValueError()
    if query(['/usr/bin/dpkg', '--print-architecture']) != b'amd64\n':
        raise ValueError()
    if query(['/usr/bin/dpkg-query', '-S', str(KEYRING)]) != (
            'ubuntu-keyring: ' + str(KEYRING) + '\n').encode('ascii'):
        raise ValueError()
    # Installed-package integrity check, not a new trust root or cryptographic
    # provenance claim. Require no discrepancy output as well as exit success.
    if query(['/usr/bin/dpkg', '--verify', 'ubuntu-keyring']):
        raise ValueError()
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
        if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
            raise ValueError()
        keyring = prerequisites()
        with tempfile.TemporaryDirectory(prefix='llamadart-ci-apt-', dir=TMP_ROOT) as temp:
            config = configure(Path(temp), keyring)
            for args in COMMANDS[sys.argv[1]]:
                # APT_CONFIG loads before Dir::Etc defaults. No inherited APT,
                # proxy, credential or hook environment; no -c-only isolation.
                subprocess.run(['/usr/bin/apt-get', *args],
                               env={**ENV, 'APT_CONFIG': str(config)},
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError):
        print(REFUSAL, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
