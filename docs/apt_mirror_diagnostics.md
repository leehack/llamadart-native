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
- An unrecognized row contains only `mirror_id: unclassified`. No token or
  partial URL/priority is recovered from malformed data.

The three exact URL matches and emitted IDs are:

| Exact hard-coded URL | ID |
| --- | --- |
| `http://azure.archive.ubuntu.com/ubuntu/` | `azure_archive` |
| `https://archive.ubuntu.com/ubuntu/` | `ubuntu_archive` |
| `https://security.ubuntu.com/ubuntu/` | `ubuntu_security` |

Rows with credentials, query strings, fragments, different schemes/hosts/paths,
extra tokens, invalid priorities or other bytes remain unclassified. Whitespace
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
receipts, unknown/secret-bearing and unsafe inputs, I/O errors, and refusal
through the real workflow shell steps with fake APT commands. No local APT or
package operation is performed by the tests.
