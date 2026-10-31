from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

import llama_cpp_patches as patches  # noqa: E402
import verify_release_provenance as provenance  # noqa: E402

UPSTREAM = ROOT / "third_party/llama.cpp"
SOURCE = "src/thing.cpp"
ORIGINAL = "int a = 1;\nint b = 2;\nint c = 3;\nint d = 4;\n"
EDIT_B = f"""\
diff --git a/{SOURCE} b/{SOURCE}
index 1111111..2222222 100644
--- a/{SOURCE}
+++ b/{SOURCE}
@@ -1,3 +1,4 @@
 int a = 1;
-int b = 2;
+int b = 20;
+int b2 = 21;
 int c = 3;
"""
EDIT_D = f"""\
--- a/{SOURCE}
+++ b/{SOURCE}
@@ -3,2 +3,2 @@ context that is not compared
 int c = 3;
-int d = 4;
+int d = 40;
"""


class Fixture:
    """An upstream tree and a patches directory holding the given patches."""

    def __init__(self, directory: str, named_patches: dict[str, str]) -> None:
        self.upstream = Path(directory) / "upstream"
        self.patches = Path(directory) / "patches"
        self.output = Path(directory) / "output"
        self.patches.mkdir()
        (self.upstream / SOURCE).parent.mkdir(parents=True)
        (self.upstream / SOURCE).write_text(ORIGINAL, encoding="utf-8")
        self.series = []
        for file, text in named_patches.items():
            (self.patches / file).write_text(text, encoding="utf-8")
            self.series.append({
                "file": file,
                "tracking": "https://example.invalid/issue",
                "upstream": "https://example.invalid/pull",
                "android_markers": [{"library": "libthing.so", "text": "marker " + file}],
            })
        self.write_series()

    def write_series(self) -> None:
        (self.patches / "series.json").write_text(
            json.dumps({"patches": self.series}), encoding="utf-8")

    def apply(self) -> dict[str, str]:
        return patches.apply_series(self.upstream, self.patches)


class ApplyTests(unittest.TestCase):
    def fixture(self, **named_patches: str) -> Fixture:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Fixture(directory.name, {
            name.replace("_", "-") + ".patch": text for name, text in named_patches.items()
        })

    def assert_rejected(self, fixture: Fixture, message: str) -> None:
        with self.assertRaises(patches.PatchError) as raised:
            fixture.apply()
        self.assertIn(message, str(raised.exception))

    def test_patches_of_one_file_apply_in_series_order(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B, "0002_edit_d": EDIT_D})
        self.assertEqual(
            {SOURCE: "int a = 1;\nint b = 20;\nint b2 = 21;\nint c = 3;\nint d = 40;\n"},
            fixture.apply())
        self.assertEqual(ORIGINAL, (fixture.upstream / SOURCE).read_text(encoding="utf-8"))

    def test_a_crlf_checkout_applies_like_an_lf_one(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        (fixture.upstream / SOURCE).write_bytes(ORIGINAL.replace("\n", "\r\n").encode())
        patch = fixture.patches / "0001-edit-b.patch"
        patch.write_bytes(EDIT_B.replace("\n", "\r\n").encode())
        self.assertEqual("int a = 1;\nint b = 20;\nint b2 = 21;\nint c = 3;\nint d = 4;\n",
                         fixture.apply()[SOURCE])

    def test_a_patch_that_does_not_fit_is_skipped_whole_and_never_applied_elsewhere(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B, "0002_edit_d": EDIT_D})
        (fixture.upstream / SOURCE).write_text(
            ORIGINAL.replace("int a = 1;", "int a = 10;"), encoding="utf-8")
        patched, skipped = patches.apply_available(fixture.upstream, fixture.patches)
        self.assertEqual({SOURCE: "int a = 10;\nint b = 2;\nint c = 3;\nint d = 40;\n"}, patched)
        self.assertEqual(1, len(skipped), skipped)
        self.assertIn("0001-edit-b.patch", skipped[0])
        self.assertIn("matches src/thing.cpp 0 times, expected 1", skipped[0])
        self.assert_rejected(fixture, "matches src/thing.cpp 0 times, expected 1")

    def test_an_ambiguous_hunk_does_not_fit(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        (fixture.upstream / SOURCE).write_text(ORIGINAL + ORIGINAL, encoding="utf-8")
        self.assertEqual(({}, 1), (lambda result: (result[0], len(result[1])))(
            patches.apply_available(fixture.upstream, fixture.patches)))
        self.assert_rejected(fixture, "matches src/thing.cpp 2 times, expected 1")

    def test_only_an_existing_compiled_source_can_be_patched(self) -> None:
        header = EDIT_B.replace(SOURCE, "src/thing.h")
        self.assert_rejected(self.fixture(**{"0001_header": header}),
                             "src/thing.h is not a compiled source")
        outside = EDIT_B.replace(SOURCE, "../outside.cpp")
        self.assert_rejected(self.fixture(**{"0001_outside": outside}),
                             "is not a compiled source inside llama.cpp")
        created = EDIT_B.replace("index 1111111..2222222 100644", "new file mode 100644")
        self.assert_rejected(self.fixture(**{"0001_created": created}),
                             "only edits of an existing text source")
        two_files = EDIT_B + EDIT_B.replace(SOURCE, "src/other.cpp")
        self.assert_rejected(self.fixture(**{"0001_two_files": two_files}), "unsupported line")
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        (fixture.upstream / SOURCE).unlink()
        self.assert_rejected(fixture, "does not exist")

    def test_the_series_and_the_directory_must_agree(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        (fixture.patches / "0002-unlisted.patch").write_text(EDIT_D, encoding="utf-8")
        self.assert_rejected(fixture, "0002-unlisted.patch")
        (fixture.patches / "0002-unlisted.patch").unlink()
        fixture.series.append(dict(fixture.series[0], file="0002-missing.patch"))
        fixture.write_series()
        self.assert_rejected(fixture, "0002-missing.patch")
        fixture.series = [dict(fixture.series[0], upstream="none")]
        fixture.write_series()
        self.assert_rejected(fixture, '"upstream" must be an https URL')
        fixture.series = [dict(fixture.series[0], upstream="https://example.invalid", note="x")]
        fixture.write_series()
        self.assert_rejected(fixture, "unknown keys ['note']")
        fixture.series = [dict(fixture.series[0], upstream="https://example.invalid")]
        del fixture.series[0]["note"]
        del fixture.series[0]["android_markers"]
        fixture.write_series()
        self.assert_rejected(fixture, 'must list at least one')
        fixture.series = [dict(fixture.series[0], upstream_fix=" ",
                               android_markers=[{"library": "libthing.so", "text": "m"}])]
        fixture.write_series()
        self.assert_rejected(fixture, '"upstream_fix" must be source text')

    def test_an_apple_marker_is_looked_for_in_apple_bundles_only(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        del fixture.series[0]["android_markers"]
        fixture.series[0]["apple_markers"] = [{"library": "libthing.dylib", "text": "b2 marker"}]
        fixture.write_series()
        bundle = fixture.output
        bundle.mkdir()
        (bundle / "libthing.dylib").write_bytes(b"without it")
        self.assertEqual([], patches.android_marker_errors(bundle, fixture.patches))
        self.assertEqual(
            ["libthing.dylib: built without 0001-edit-b.patch (no 'b2 marker')"],
            patches.apple_marker_errors(bundle, fixture.patches))
        (bundle / "libthing.dylib").write_bytes(b"b2 marker")
        self.assertEqual([], patches.apple_marker_errors(bundle, fixture.patches))

    def test_an_upstream_fix_in_another_file_retires_the_patch(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        fixture.series[0]["upstream_fix"] = "fixed\n"
        fixture.series[0]["upstream_fix_path"] = "other.txt"
        fixture.write_series()
        self.assertEqual(1, len(patches.required_series(fixture.upstream, fixture.patches)))
        (fixture.upstream / "other.txt").write_text("fixed\n", encoding="utf-8")
        self.assertEqual([], patches.required_series(fixture.upstream, fixture.patches))

    def test_an_upstream_fix_path_needs_the_fix_text(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        fixture.series[0]["upstream_fix_path"] = "other.txt"
        fixture.write_series()
        self.assert_rejected(fixture, '"upstream_fix_path" needs "upstream_fix"')

    def required_text_fixture(self) -> Fixture:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        fixture.series[0]["required_text"] = {"path": "other.txt", "text": "edit me"}
        fixture.series[0]["android_markers"] = [{"library": "libthing.so", "text": "b2 marker"}]
        fixture.write_series()
        return fixture

    def test_a_patch_whose_runtime_edit_has_nothing_to_edit_does_not_fit(self) -> None:
        fixture = self.required_text_fixture()
        (fixture.upstream / "other.txt").write_text("renamed\n", encoding="utf-8")
        patched, skipped = patches.apply_available(fixture.upstream, fixture.patches)
        self.assertEqual({}, patched)
        self.assertIn("other.txt", skipped[0])
        self.assert_rejected(fixture, "other.txt")
        bundle = fixture.output
        bundle.mkdir()
        (bundle / "libthing.so").write_bytes(b"without it")
        self.assertEqual(
            ["libthing.so: built without 0001-edit-b.patch (no 'b2 marker')"],
            patches.android_marker_errors(bundle, fixture.patches, fixture.upstream))

    def test_a_missing_file_of_required_text_does_not_fit(self) -> None:
        self.assert_rejected(self.required_text_fixture(), "other.txt")

    def test_a_patch_whose_runtime_edit_has_its_text_applies(self) -> None:
        fixture = self.required_text_fixture()
        (fixture.upstream / "other.txt").write_text("edit me\n", encoding="utf-8")
        self.assertIn("int b = 20;", fixture.apply()[SOURCE])

    def test_required_text_must_name_a_path_and_a_text(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        fixture.series[0]["required_text"] = {"path": "other.txt"}
        fixture.write_series()
        self.assert_rejected(fixture, '"required_text" must be {path, text}')

    def check_apple(self, fixture: Fixture) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/llama_cpp_patches.py"),
             "--patches", str(fixture.patches), "check-apple",
             "--upstream", str(fixture.upstream), "--dir", str(fixture.output)],
            capture_output=True, text=True, check=False)

    def test_the_check_apple_command_fails_on_a_bundle_without_the_marker(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        del fixture.series[0]["android_markers"]
        fixture.series[0]["apple_markers"] = [{"library": "libthing.dylib", "text": "b2 marker"}]
        fixture.write_series()
        fixture.output.mkdir()
        (fixture.output / "libthing.dylib").write_bytes(b"without it")
        result = self.check_apple(fixture)
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("built without 0001-edit-b.patch", result.stderr)
        (fixture.output / "libthing.dylib").write_bytes(b"b2 marker")
        result = self.check_apple(fixture)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_the_check_apple_command_fails_on_a_bundle_without_the_library(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        del fixture.series[0]["android_markers"]
        fixture.series[0]["apple_markers"] = [{"library": "libthing.dylib", "text": "b2 marker"}]
        fixture.write_series()
        fixture.output.mkdir()
        result = self.check_apple(fixture)
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("libthing.dylib", result.stderr)

    def test_an_unchanged_copy_keeps_its_timestamp(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        self.assertEqual(([SOURCE], []),
                         patches.write_patched(fixture.upstream, fixture.output, fixture.patches))
        copy = fixture.output / SOURCE
        os.utime(copy, (1, 1))
        patches.write_patched(fixture.upstream, fixture.output, fixture.patches)
        self.assertEqual(1, copy.stat().st_mtime)
        (fixture.patches / "0001-edit-b.patch").write_text(
            EDIT_B.replace("int b2 = 21;", "int b2 = 22;"), encoding="utf-8")
        patches.write_patched(fixture.upstream, fixture.output, fixture.patches)
        self.assertNotEqual(1, copy.stat().st_mtime)
        self.assertIn("int b2 = 22;", copy.read_text(encoding="utf-8"))

    def test_the_tool_prints_what_cmake_reads_and_names_a_skipped_patch(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        command = [sys.executable, str(ROOT / "tools/llama_cpp_patches.py"),
                   "--patches", str(fixture.patches), "apply",
                   "--upstream", str(fixture.upstream), "--output", str(fixture.output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual((0, SOURCE + "\n", ""),
                         (result.returncode, result.stdout, result.stderr))
        (fixture.upstream / SOURCE).write_text("int z;\n", encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual((0, "\n"), (result.returncode, result.stdout))
        self.assertIn("skipped 0001-edit-b.patch", result.stderr)
        result = subprocess.run(command + ["--strict"], capture_output=True, text=True)
        self.assertEqual(1, result.returncode)
        self.assertIn("0001-edit-b.patch", result.stderr)
        fixture.series[0]["tracking"] = "none"
        fixture.write_series()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(1, result.returncode, "a malformed series is never skipped")

    def test_a_bundle_library_without_its_marker_is_reported(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        fixture.series[0]["android_markers"] = [{"library": "libthing.so", "text": "b2 marker"}]
        fixture.write_series()
        # The check reads the bundle only, so it also catches a patch that the
        # build skipped.
        bundle = fixture.output
        bundle.mkdir()
        self.assertEqual([], patches.android_marker_errors(bundle, fixture.patches))
        (bundle / "libthing.so").write_bytes(b"\x7fELF without it")
        self.assertEqual(
            ["libthing.so: built without 0001-edit-b.patch (no 'b2 marker')"],
            patches.android_marker_errors(bundle, fixture.patches))
        (bundle / "libthing.so").write_bytes(b"\x7fELF b2 marker\x00")
        self.assertEqual([], patches.android_marker_errors(bundle, fixture.patches))

    def test_a_patch_whose_upstream_fix_is_present_is_retired_everywhere(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B, "0002_edit_d": EDIT_D})
        fixture.series[0]["upstream_fix"] = "int b = 200;\n"
        fixture.write_series()
        bundle = fixture.output
        bundle.mkdir()
        (bundle / "libthing.so").write_bytes(b"\x7fELF marker 0002-edit-d.patch\x00")
        self.assertEqual(
            "int a = 1;\nint b = 20;\nint b2 = 21;\nint c = 3;\nint d = 40;\n",
            fixture.apply()[SOURCE], "the fix is absent, so the patch applies")
        self.assertEqual(["libthing.so: built without 0001-edit-b.patch (no 'marker "
                          "0001-edit-b.patch')"],
                         patches.android_marker_errors(bundle, fixture.patches, fixture.upstream))
        (fixture.upstream / SOURCE).write_text(
            ORIGINAL.replace("int b = 2;", "int b = 200;"), encoding="utf-8")
        # The retired hunk would not fit either; it must not be reported as skipped.
        self.assertEqual(
            ({SOURCE: "int a = 1;\nint b = 200;\nint c = 3;\nint d = 40;\n"}, []),
            patches.apply_available(fixture.upstream, fixture.patches))
        self.assertEqual(["0002-edit-d.patch"],
                         [entry["file"] for entry in patches.manifest_entries(
                             fixture.patches, fixture.upstream)])
        self.assertEqual([], patches.android_marker_errors(
            bundle, fixture.patches, fixture.upstream))
        (fixture.upstream / SOURCE).unlink()
        self.assertEqual(["0001-edit-b.patch", "0002-edit-d.patch"],
                         [entry["file"] for entry in patches.required_series(
                             fixture.upstream, fixture.patches)],
                         "a missing source never retires a patch")

    def test_the_manifest_names_each_patch_and_its_digest(self) -> None:
        fixture = self.fixture(**{"0001_edit_b": EDIT_B})
        self.assertEqual(
            [{"file": "0001-edit-b.patch",
              "sha256": hashlib.sha256(EDIT_B.encode()).hexdigest(),
              "tracking": "https://example.invalid/issue"}],
            patches.manifest_entries(fixture.patches, fixture.upstream))


class CarriedSeriesTests(unittest.TestCase):
    def test_the_series_is_well_formed(self) -> None:
        for entry in patches.load_series():
            patches.parse_patch(entry["file"], (patches.PATCHES_DIR / entry["file"]).read_text(
                encoding="utf-8"))

    def test_the_build_applies_the_series_after_adding_llama_cpp(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        added = cmake.index("add_subdirectory(third_party/llama.cpp)")
        applied = cmake.index("llamadart_apply_llama_cpp_patches()")
        self.assertLess(added, applied)
        self.assertLess(applied, cmake.index("add_library(llamadart_lib SHARED"))

    def carried_after(self, context_source: str | None, newline: str = "\n") -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            if context_source is not None:
                source = Path(directory) / "src/llama-context.cpp"
                source.parent.mkdir()
                source.write_bytes(context_source.replace("\n", newline).encode())
            return [entry["file"] for entry in patches.manifest_entries(
                patches.PATCHES_DIR, Path(directory))]

    def test_the_embedding_reserve_patch_is_retired_only_where_upstream_has_the_fix(self) -> None:
        entry = next(entry for entry in patches.load_series()
                     if entry["file"] == "0003-embedding-mode-scheduler-reserve.patch")
        # v0.8.0 (upstream ggml-org/llama.cpp#31086) toggles embeddings like this.
        self.assertEqual("    if (cparams.embeddings != value) {\n"
                         "        sched_need_reserve = true;\n"
                         "    }\n", entry.get("upstream_fix"))
        fixed = ("void llama_context::set_embeddings(bool value) {\n"
                 + entry["upstream_fix"] + "    cparams.embeddings = value;\n}\n")
        unfixed = ("void llama_context::set_embeddings(bool value) {\n"
                   "    cparams.embeddings = value;\n}\n")
        patch = "0003-embedding-mode-scheduler-reserve.patch"
        self.assertNotIn(patch, self.carried_after(fixed))
        self.assertNotIn(patch, self.carried_after(fixed, "\r\n"), "a CRLF checkout")
        self.assertIn(patch, self.carried_after(unfixed))
        self.assertIn(patch, self.carried_after(None), "a missing source retires nothing")

    METAL_PATCH = "0004-metal-unary-cast-older-os-compilers.patch"
    UNARY = "ggml/src/ggml-metal/kernels/unary.metal"

    def metal_patch_entry(self) -> dict:
        return next(entry for entry in patches.load_series()
                    if entry["file"] == self.METAL_PATCH)

    def test_the_verifier_and_the_tool_agree_on_retiring_the_metal_patch(self) -> None:
        entry = self.metal_patch_entry()
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            unary = upstream / self.UNARY
            unary.parent.mkdir(parents=True)
            for text, retired in (("constexpr specialization\n", False),
                                  (entry["upstream_fix"], True)):
                unary.write_text(text, encoding="utf-8")
                self.assertEqual(retired, provenance.fixed_upstream(
                    entry, patches.PATCHES_DIR, upstream))
                carried = [e["file"] for e in patches.manifest_entries(
                    patches.PATCHES_DIR, upstream)]
                self.assertEqual(not retired, self.METAL_PATCH in carried)

    def test_the_metal_patch_edits_the_text_it_requires(self) -> None:
        entry = self.metal_patch_entry()
        self.assertEqual(self.UNARY, entry["required_text"]["path"])
        patch = (patches.PATCHES_DIR / self.METAL_PATCH).read_text(encoding="utf-8")
        edited = re.search(r'@"(inline constexpr [^"]+)"', patch).group(1)
        self.assertTrue(entry["required_text"]["text"].startswith(edited))

    @unittest.skipUnless((UPSTREAM / "CMakeLists.txt").is_file(),
                         "third_party/llama.cpp is not checked out")
    def test_the_required_text_of_each_carried_patch_is_upstream(self) -> None:
        for entry in patches.required_series(UPSTREAM):
            required = entry.get("required_text")
            if required is not None:
                self.assertIn(required["text"],
                              (UPSTREAM / required["path"]).read_text(encoding="utf-8"))

    def test_the_manifest_command_reads_the_given_upstream_not_the_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src/llama-context.cpp"
            source.parent.mkdir()
            source.write_text("void llama_context::set_embeddings(bool value) {}\n",
                              encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/llama_cpp_patches.py"),
                 "manifest", "--upstream", directory], capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(["0001", "0002", "0003", "0004"],
                         [e["file"][:4] for e in json.loads(result.stdout)])

    @unittest.skipUnless((UPSTREAM / "CMakeLists.txt").is_file(),
                         "third_party/llama.cpp is not checked out")
    def test_the_series_applies_to_the_checked_out_llama_cpp(self) -> None:
        patched = patches.apply_series(UPSTREAM)
        for entry in patches.required_series(UPSTREAM):
            path = patches.parse_patch(entry["file"], (patches.PATCHES_DIR / entry["file"])
                                       .read_text(encoding="utf-8")).path
            original = (UPSTREAM / path).read_text(encoding="utf-8")
            for marker in entry.get("android_markers", []) + entry.get("apple_markers", []):
                # A marker tells a patched library from an unpatched one only
                # if the patch is what introduces it.
                self.assertNotIn(marker["text"], original, entry["file"])
                self.assertIn(marker["text"], patched[path], entry["file"])
        status = subprocess.run(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"],
            capture_output=True, text=True, check=True)
        self.assertEqual("", status.stdout, "applying must not modify the submodule")


if __name__ == "__main__":
    unittest.main()
