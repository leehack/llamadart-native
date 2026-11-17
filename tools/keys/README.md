# Pinned Ubuntu archive public key for disposable CI

This public key is a repository-controlled CI trust snapshot, not a copy of a
runner's installed keyring. Only `tools/ci_ubuntu_apt.py` consumes it, using a
private `Signed-By` file and the full fingerprint below. No global APT trust is
changed. No keyserver, runtime key fetch, host-key fallback or chmod repair is
permitted.

## Publisher and exact bytes

- Publisher source: [Canonical chisel-releases, Noble configuration](https://github.com/canonical/chisel-releases/blob/731a4b2b94bbab72082d74b3d62342f1a1c1ad9f/chisel.yaml#L38-L72).
- Immutable commit: `731a4b2b94bbab72082d74b3d62342f1a1c1ad9f`; resolved from its
  official `ubuntu-24.04` branch. The configuration identifies Ubuntu 24.04 and
  selects `ubuntu-archive-key-2018` for its public Ubuntu archive.
- Snapshot: only that named YAML armor block, removing six indentation spaces
  from nonblank lines, preserving blank lines and adding a final LF. No GPG
  re-export, ESM/Apps key or ambient file is included.
- ASCII armor: 1660 bytes; SHA256
  `2a3cc57ab6b47626b496a101c29af6dfe54d54d03d613f2326b9f2a30a15c39b`.
- Full primary fingerprint: `F6ECB3762474EDA9D21B7022871920D1991BC93C`.
- Ubuntu Archive Automatic Signing Key (2018), RSA4096, created
  `2018-09-17T15:01:46Z`; one primary key, zero subkeys, no expiry recorded.
- Independent publisher fingerprint: [Canonical Ubuntu security documentation](https://github.com/canonical/ubuntu-security-documentation/blob/5de5088956161cc976aa14b143405ab36de3097c/software-integrity/image-verification.rst)
  and its [published page](https://documentation.ubuntu.com/security/software-integrity/image-verification/).

This is versioned by the publisher commit and key creation identity; it is not
claimed to be the complete `ubuntu-keyring` package or a particular package
version. The source repository metadata reports AGPL-3.0; attribution and
immutable provenance are retained here. The Noble branch has no LICENSE file.

## Trust and rotation

The bootstrap is the authenticated Canonical publisher repositories over HTTPS,
the independently published full fingerprint, and review of these exact bytes
and digest into this repository. SHA256 binds bytes; a key self-signature checks
internal consistency. Neither alone establishes publisher identity. The GitHub
verified commit indicator is not an Ubuntu archive-key certification.

Runtime reads the fixed repository snapshot with no-follow/nonblocking regular
file checks, exact size, safe read mode, bounded IO and before/after descriptor
metadata checks. Its exact SHA256 must match before captured bytes are copied
into the fresh root-owned operation directory as a public 0644 `.asc` file.
Checkout ownership is not key authentication. OS-file/root/Ubuntu24.04/amd64
requirements remain; no ambient host keyring or package database authenticates
the key. The trusted checkout/helper code is part of the bootstrap boundary.

Rotation or newly published revocation requires separate review of an immutable
publisher capture, independent full fingerprint, OpenPGP structure and digest,
then updates to this file, helper constants, documentation and independent
fixture expectations. No runtime refresh/fallback occurs. A frozen snapshot
cannot automatically learn a subsequent revocation; a new unknown archive
signer must fail. Check-Valid-Until, signature and TLS checks remain enabled.

[APT 2.8.2 documentation](https://github.com/Debian/apt/blob/39d24618587e28b2dd2cae04ed2393056869f771/doc/apt-key.8.xml)
supports ASCII `.asc` files (APT >=1.4), and its
[Signed-By contract](https://github.com/Debian/apt/blob/39d24618587e28b2dd2cae04ed2393056869f771/doc/sources.list.5.xml)
supports a private absolute keyring path plus full fingerprint. These are format
contracts, not evidence that the hosted Noble resolver or every enabled suite
has passed with this snapshot. Actual hosted APT remains a required gate.
