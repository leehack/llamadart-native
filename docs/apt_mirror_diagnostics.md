# Failure-only APT mirror diagnostics

`tools/prioritize_ubuntu_archive.py` still accepts exactly its two existing
`ORIGINAL` / `PREFERRED` byte strings, from the reviewed
`actions/runner-images` tag `ubuntu24/20261101.360`. Recognition in a diagnostic
**does not admit a format or change priorities**. Unknown input exits 1 before
APT; the existing workflow shell failure behavior is retained.

The production CLI takes no arguments and opens only
`/etc/apt/apt-mirrors.txt`, with no-follow/nonblocking flags and regular-file
validation. Tests override that fixed path inside the imported module; the CLI
has no path override and reads no environment or image metadata.

## Receipt and limits

On an unexpected bounded regular file, stderr contains the fixed refusal and
one JSON receipt, schema `llamadart-apt-mirror-diagnostic/1`. Success is silent.
The receipt contains only:

- Total byte/line counts, final-LF flag and LF/CRLF/bare-CR counts.
- One ordered entry per LF-delimited row. A fully recognized row contains an
  ID, a decimal priority from 1 through 999, whitespace encoded as `s` (space)
  / `t` (tab), and ending `lf` / `crlf` / `none`.
- Every row additionally contains `known_url_id`, one of the four IDs below
  or `unclassified`. Only the complete first ASCII SP/TAB-delimited token,
  after at most 32 leading spaces/tabs, is compared by exact byte equality
  against only the four literals below. No prefix/substring matching, normalization
  or credential/query/fragment stripping occurs.
- An unrecognized full row keeps `mirror_id: unclassified`, even if its exact
  URL token is known. Its metadata and priority remain unclassified and
  unprinted. Tokens containing controls, URL suffixes or other unknown bytes
  have `known_url_id: unclassified`. No raw token is emitted.

The four exact diagnostic URL matches and emitted IDs are:

| Exact hard-coded URL | ID |
| --- | --- |
| `http://azure.archive.ubuntu.com/ubuntu/` | `azure_archive` |
| `https://archive.ubuntu.com/ubuntu/` | `ubuntu_archive` |
| `https://security.ubuntu.com/ubuntu/` | `ubuntu_security` |
| `http://us-east-1.ec2.archive.ubuntu.com/ubuntu/` | `ubuntu_ec2_us_east_1` |

Rows with credentials, query strings, fragments, different schemes/hosts/paths,
extra tokens, invalid priorities or other bytes remain unclassified as full
rows. A known URL with unrecognized or secret-bearing metadata can emit its
`known_url_id` only; this does not classify or print the metadata. Whitespace
fields are individually limited to 32 spaces/tabs. This narrow classification
is for understanding a refused file, not parsing general APT mirror syntax.

Input limits are 4096 bytes, 16 rows and 256 bytes per row. File size is checked
before reading, and one read requests at most 4097 bytes to detect growth. A
short read, size mismatch or size/mtime/ctime change during the read is refused
without a receipt. Symlinks, FIFOs, nonregular files, oversized files/rows, excess rows and I/O failures all produce
only the same generic refusal: no type, size, path, exception or content is
reported for them. JSON payloads are additionally capped at 8192 ASCII bytes.
No raw configuration, unknown URLs/tokens, hashes, credentials, query strings,
exception text or environment is logged. No file writes occur on refusal.

## Unresolved hosted input

At PR188 head `36c8ad731ec8c23d7ff05d30931f2a727ba6b11b`,
[the arm64/v0.9 job](https://github.com/leehack/llamadart-native/actions/runs/43737601152/job/132883696903)
reported Ubuntu image `20261111.127.1`, provisioner `20261103.629`, and links to
`ubuntu24/20261111.127`. Those exact public source/tag links were unavailable
during diagnosis; the log did not contain mirrorlist bytes. The different
same-date `.370` source is not a substitute for `.127.1` provenance.

At diagnostic head `796efdac8b72debf02a2eb3e042f851be506a04f`, the specified
[current arm64/v0.8 job](https://github.com/leehack/llamadart-native/actions/runs/43740410500/job/132892160801)
and [Kleidi v0.9 job](https://github.com/leehack/llamadart-native/actions/runs/43740410500/job/132892160757)
independently reported the same `.127.1` image and identical receipts: three
LF-terminated rows, first row unclassified, then recognized archive/security
rows. The exact `.127` official source references still returned 404. The first
row identity remains unknown; its byte count is not evidence of a URL.

The additive `known_url_id` comparison can distinguish a known URL with
unrecognized metadata from an unknown token without printing either. Its initial
public literal basis is the three rows in the
[official immutable source at `add4380b0`](https://github.com/actions/runner-images/blob/add4380b0d6685806e9192c33c37359ee0aa4023/images/ubuntu/scripts/build/configure-apt-sources.sh#L11-L13),
**not provenance of image `.127.1`**. If the next receipt remains unknown, stop
and obtain trusted exact-image source mapping or independent public evidence
for a specific literal before proposing further classification. No parser or
trust repair is authorized by this diagnostic.

At head `8274a8ab98db70cd0cd0045fc961bc4795f30165`, setup headers in
[job 132897054127](https://github.com/leehack/llamadart-native/actions/runs/43742014200/job/132897054127)
and [job 132897054056](https://github.com/leehack/llamadart-native/actions/runs/43742014200/job/132897054056)
both explicitly report cloud `AWS`, region `us-east-1`, and image
`ubuntu-24.04` / `20261111.127.1`. The earlier investigation focused on the
Azure/build-image source and missed the runtime cloud metadata.

Canonical's pinned
[cloud-init template, commit `6d221dd8`, lines 347-358](https://github.com/canonical/cloud-init/blob/6d221dd83eb897e8faac42f5709f33d0e7eb2faf/config/cloud.cfg.tmpl#L347-L358)
contains the EC2-region archive pattern at line 355. Its
[official APT docs](https://docs.cloud-init.io/en/latest/reference/yaml_examples/apt.html#specify-mirrors)
describe the EC2 datasource's regional mirror default. Combining that public
template with the observed region yields exactly the fourth literal above.
It is a **hypothesis for the unknown row**, not observed mirrorlist bytes or
exact `.127.1` image provenance, and was not inferred from byte count or `.370`.

Only that exact literal is added to diagnostic classification: it can emit
`known_url_id` and, only on an existing full-row match, the existing bounded
validated metadata fields. Unknown or secret-bearing metadata still leaves
`mirror_id: unclassified` and is never printed. No dynamic cloud/region/domain
matching, scheme/path variants or additional literals are recognized. The
production helper does not read cloud metadata. Both accepted byte strings and
all admission/write behavior remain unchanged; candidate input still refuses
with exit 1 before APT. A later observed receipt must confirm actual input
before any separately authorized parser/trust repair.

This change gathers missing structure through ordinary PR CI. It does not fix
or establish the cause of that refusal, add a guessed image fixture, broaden
URL trust, change runner/compiler/source pins or qualify an artifact. Synthetic
whitespace/newline/priority fixtures are structural probes, not claims about the
failed image. A subsequent parser/trust change needs confirmed input and
trusted provenance first; an unclassified row cannot be reconstructed from
this receipt and must not be guessed. No manual workflow retry/dispatch is
needed or performed by the helper.

Focused checks: `python3 -m unittest discover -s tests -p test_prioritize_ubuntu_archive.py`.
They cover both exact accepted formats, metadata/idempotence, bounded structural
receipts, all four exact URL tokens with unknown/secret-bearing metadata,
URL prefix/suffix/credential/query/fragment/case/control negatives, unsafe
inputs, I/O errors, and refusal
through the real workflow shell steps with fake APT commands. No local APT or
package operation is performed by the tests.

## Explicit private-source CI policy

The `kleidiai-dispatch-emulated` dependency step and the arm64 branch of the
`linux-artifact-contract` dependency step select `tools/ci_ubuntu_apt.py`
explicitly. This is a disposable-job source policy, not a fallback after this
mirror helper refuses. The strict mirror helper's default, admitted bytes and
refusal behavior remain unchanged. Linux x64 and other jobs retain their
existing acquisition paths.

The private helper requires Ubuntu 24.04 / noble on an amd64 host and the
repository-pinned [Ubuntu archive public key](../tools/keys/README.md). It creates a fresh root-owned `/tmp` directory
for configuration, sources, keyring copy, indexes, caches and APT logs. Only
fixed HTTPS archive/security URLs with Noble suites are used. `APT_CONFIG`
loads before system configuration; `Dir::Etc` points entirely into this private
area. No host source, mirrorlist, credential, pinning or hook file is imported.
`Signed-By` limits both sources to the exact private snapshot and full fingerprint; TLS,
signature and date checks remain enabled, redirects/insecure sources are
refused, and any update error stops installation. Installed-package state is
still `/var/lib/dpkg/status`; package installation itself remains a real change
to the disposable runner, not an isolated root filesystem.

The fixed package groups and `install -y` options are unchanged. No Microsoft,
Kitware or other third-party source is available to these private operations;
a future dependency needing one must stop for a separate policy decision. No
new key is downloaded, no global trust is changed, and child output or unknown
host input is not printed. Failure exits 1 without a host-source retry. Real
APT parsing, signature verification, dependency resolution and downstream
artifact/ISA execution require hosted validation; fixture tests do not qualify
them. The failed EC2 literal hypothesis does not identify the original input.

## Private-helper failure stages

The generic private-helper refusal and exit 1 are retained. Failure adds one
ASCII JSON line prefixed `Private Ubuntu APT failure stage: `. Its `stage` is
only `arguments`, `prerequisites`, `private_configuration`, `update` or
`install`. Private configuration includes directory creation and cleanup.
A failed child command may additionally report `exit_status` only for an
integer from 1 through 255. Signals, timeouts, OS errors and malformed statuses
have no numeric field. No child command/output, exception text, host value,
path, environment, credential or unknown content/hash is printed. The closed
fields bound the diagnostic line to at most 128 ASCII bytes. Success is silent.

These phases locate a refused operation; they do not establish why a
prerequisite, signature check, connection or resolver failed. In particular the
reported head `48c1ba8` setup failures are not evidence of successful APT or
failed downstream dispatch/numerical/artifact validation. No diagnostic triggers
another query, retry, install, write or host-source fallback. The approved
acquisition policy and package/option order remain unchanged. Actual hosted
APT and downstream qualification remain separate gates.

## Prerequisite check identifiers

Prerequisite refusals additionally emit one hardcoded `check` identifier. The
closed enum distinguishes root; OS link/read, file type/owner/mode/size/stability,
encoding/line count/assignment/duplicate key/value syntax/value arity, required
ID/version/codename; architecture query/output limit/value; keyring package query/output
limit/attribution and integrity query/output limit/result; and keyring read/type/owner/mode/size/stability.
Only captured metadata and existing validation results choose these labels.
No host value, file content, line/key name from input, child output, path or
exception text is exposed. Existing safe numeric status rules and the 128-byte
ASCII diagnostic-line bound remain. Other phases do not gain a `check` field.

These labels instrument the existing checks and retain their order, predicates,
limits, command sequence, cleanup and no-fallback policy. No additional query,
read, retry or write is introduced. The reported head `3b764a9` receipts localize
the failure before APT update/install, but do not identify the exact check or
cause. The identifiers gather that missing evidence without guessing inputs or
loosening the approved root/Ubuntu24.04/amd64/OS-release/keyring requirements.

## Public-keyring mode predicates

Only a `keyring_file_mode` refusal additionally emits one record prefixed
`Private Ubuntu keyring mode predicates: `. It contains exactly three boolean
fields: `group_writable`, `other_writable`, `other_read_missing`. All three are
reported together from the already captured mode of the public packaged
keyring. No new stat, read, query, mode change or child invocation occurs.
Other refusal checks, including OS-file mode and earlier keyring ownership/type
failures, do not emit this record. The existing generic refusal and stage
receipt remain unchanged; each diagnostic line stays within 128 ASCII bytes.
No raw mode, user/group identity, path, content, environment or exception is
reported. Success remains silent, and refusal still exits 1 with no fallback.

The approved permission predicates are unchanged: reject group/other-writable
files and require the other-read bit for this public keyring. The observed
head `d1a82df` receipts identify only the combined mode check, not actual bits,
package corruption, provisioning cause or an acquisition/network failure.
Primary Noble package metadata lists the keyring but does not establish this
runner's installed mode. Dpkg verification is an installed-file checksum check,
not proof of mode; the bounded booleans gather the remaining evidence without
loosening policy. Any permission-policy change requires a separate proposal.


## Proposed independently pinned CI keyring policy

The writable ambient packaged keyring is never authenticated by chmod, owner
checks or its local MD5 database. The private helper uses only the exact
repository snapshot documented in `tools/keys/README.md`. Its SHA256 binds the
captured public bytes to an independently reviewed Canonical publisher capture;
no host-key read, attribution/integrity query or runtime key download occurs.
Root, strict Ubuntu 24.04/noble OS-file validation and the amd64 architecture
query remain. Two ambient-key package queries are removed deliberately because
they no longer authenticate the selected trust source; the architecture query
is retained. This is a trust-source policy change, not a mode-guard relaxation.

The source snapshot must be a bounded regular no-follow/nonblocking file with
exact size, other-read permission and no group/other-write permission. Descriptor
identity, mode, ownership, size and nanosecond times must remain stable across
one bounded read. Captured bytes must match the pinned SHA256 before copying
them into the private root-owned operation. Only this authenticated copy is
referenced by `Signed-By`. Refusals emit fixed `pinned_keyring_read` or
`pinned_keyring_content` checks; no content, digest, metadata or exception is
printed. The older closed diagnostic identifiers remain reserved; the generic
strict file guard is retained, without selecting an ambient trust source.

At `cfe1c77`, all six supplied Kleidi/Linux arm64 receipts establish
`group_writable: true`, `other_writable: true`, `other_read_missing: false` for
the refused ambient public keyring. They establish neither its exact mode,
ownership, authenticity, integrity nor provisioning cause. The aggregate
correctly propagates those setup-blocked groups; it remains required. No
mirror identity or network/solver conclusion follows.

The canonical HTTPS sources, suites/components, no-remove policy, strict TLS,
signature/date/update checks, package/option order, x64 acquisition path and
downstream artifact/ISA validators are unchanged. An absent, modified, unsafe
or unknown snapshot refuses before APT, including when ambient bytes happen to
match. A future key rotation/revocation or dependency/source/solver conflict
requires a separate reviewed decision; no ambient fallback is available.

Private APT configuration stays behind the operation directory's `0700` boundary while files are populated and assigned final safe modes. Traversal for `_apt` is published only after those permissions are complete; setup failures leave the private boundary intact.
