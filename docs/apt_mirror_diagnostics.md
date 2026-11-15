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
- Every row additionally contains `known_url_id`, one of the three IDs below
  or `unclassified`. Only the complete first ASCII SP/TAB-delimited token,
  after at most 32 leading spaces/tabs, is compared by exact byte equality
  against the same three literals. No prefix/substring matching, normalization
  or credential/query/fragment stripping occurs.
- An unrecognized full row keeps `mirror_id: unclassified`, even if its exact
  URL token is known. Its metadata and priority remain unclassified and
  unprinted. Tokens containing controls, URL suffixes or other unknown bytes
  have `known_url_id: unclassified`. No raw token is emitted.

The three exact URL matches and emitted IDs are:

| Exact hard-coded URL | ID |
| --- | --- |
| `http://azure.archive.ubuntu.com/ubuntu/` | `azure_archive` |
| `https://archive.ubuntu.com/ubuntu/` | `ubuntu_archive` |
| `https://security.ubuntu.com/ubuntu/` | `ubuntu_security` |

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
unrecognized metadata from an unknown token without printing either. Its public
literal basis is only the existing three rows in the
[official immutable source at `add4380b0`](https://github.com/actions/runner-images/blob/add4380b0d6685806e9192c33c37359ee0aa4023/images/ubuntu/scripts/build/configure-apt-sources.sh#L11-L13),
**not provenance of image `.127.1`**. If the next receipt remains unknown, stop
and obtain trusted exact-image source mapping or independent public evidence
for a specific literal before proposing further classification. No parser or
trust repair is authorized by this diagnostic.

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
receipts, all three exact URL tokens with unknown/secret-bearing metadata,
URL prefix/suffix/credential/query/fragment/case/control negatives, unsafe
inputs, I/O errors, and refusal
through the real workflow shell steps with fake APT commands. No local APT or
package operation is performed by the tests.
