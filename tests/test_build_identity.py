from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_identity  # noqa: E402

SOURCE = "ggml/src/thing.cpp"
ORIGINAL = "int a = 1;\nint b = 2;\nint c = 3;\n"
PATCH = f"""\
--- a/{SOURCE}
+++ b/{SOURCE}
@@ -1,3 +1,3 @@
 int a = 1;
-int b = 2;
+int b = {{value}};
 int c = 3;
"""
FIELDS = {
    "config": "Release",
    "compiler": "AppleClang-17.0.0,AppleClang-17.0.0",
    "target": "Darwin-arm64-macosx",
    "backends": "cpu,metal",
    "options": "0123456789abcdef",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def commit_all(repo: Path, message: str) -> None:
    git(repo, "-c", "advice.addEmbeddedRepo=false", "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def fields_of(value: str) -> dict[str, str]:
    head, *pairs = value.split(";")
    assert head == build_identity.FORMAT, value
    return dict(pair.split("=", 1) for pair in pairs)


class BuildIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        # The user's git config (signing, diff settings) must not reach the
        # fixture commits or the identity.
        empty_config = self.temp / "gitconfig"
        empty_config.write_text("", encoding="utf-8")
        environment = mock.patch.dict(os.environ, {
            "GIT_CONFIG_GLOBAL": str(empty_config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        })
        environment.start()
        self.addCleanup(environment.stop)

        self.source = self.temp / "checkout"
        self.upstream = self.source / "third_party" / "llama.cpp"
        (self.upstream / SOURCE).parent.mkdir(parents=True)
        (self.upstream / SOURCE).write_text(ORIGINAL, encoding="utf-8")
        (self.upstream / "README.md").write_text("upstream\n", encoding="utf-8")
        git(self.upstream, "init", "-q")
        commit_all(self.upstream, "upstream")

        patches = self.source / "patches" / "llama.cpp"
        patches.mkdir(parents=True)
        (patches / "series.json").write_text(json.dumps({"patches": [{
            "file": "0001-edit-thing.patch",
            "tracking": "https://example.invalid/issue",
            "upstream": "https://example.invalid/pull",
            "android_markers": [{"library": "libggml.so", "text": "marker"}],
        }]}), encoding="utf-8")
        self.write_patch("20")
        (self.source / "src").mkdir()
        (self.source / "src" / "wrapper.cpp").write_text("int wrapper;\n", encoding="utf-8")
        git(self.source, "init", "-q")
        commit_all(self.source, "owner")

    def write_patch(self, value: str) -> None:
        (self.source / "patches" / "llama.cpp" / "0001-edit-thing.patch").write_text(
            PATCH.format(value=value), encoding="utf-8")

    def identity(self, source: Path | None = None, **overrides: str) -> str:
        return build_identity.identity(source or self.source, **{**FIELDS, **overrides})

    def test_names_every_input_and_no_path(self):
        patched = ORIGINAL.replace("int b = 2;", "int b = 20;")
        patches = hashlib.sha256(
            SOURCE.encode() + b"\0" + patched.encode() + b"\0").hexdigest()[:16]
        expected = ";".join([
            "llamadart-native-build/1",
            f"owner={git(self.source, 'rev-parse', 'HEAD')}",
            f"llama.cpp={git(self.upstream, 'rev-parse', 'HEAD')}",
            f"ggml={git(self.upstream, 'rev-parse', 'HEAD:ggml')}",
            f"patches={patches}",
            "config=Release",
            "compiler=AppleClang-17.0.0,AppleClang-17.0.0",
            "target=Darwin-arm64-macosx",
            "backends=cpu,metal",
            "options=0123456789abcdef",
        ])
        value = self.identity()
        self.assertEqual(value, expected)
        self.assertNotIn(self.temp.name, value)

    def test_same_inputs_at_another_path_give_the_same_identity(self):
        copy = self.temp / "elsewhere" / "other-checkout"
        shutil.copytree(self.source, copy, symlinks=True)
        self.assertEqual(self.identity(), self.identity())
        self.assertEqual(self.identity(copy), self.identity())

    def test_each_input_changes_only_its_fields(self):
        baseline = fields_of(self.identity())

        def commit_owner() -> None:
            (self.source / "src" / "wrapper.cpp").write_text("int wrapper2;\n", encoding="utf-8")
            commit_all(self.source, "wrapper")

        def edit_owner(text: str):
            def edit() -> None:
                (self.source / "src" / "wrapper.cpp").write_text(text, encoding="utf-8")
            return edit

        def commit_upstream_outside_ggml() -> None:
            (self.upstream / "README.md").write_text("upstream 2\n", encoding="utf-8")
            commit_all(self.upstream, "readme")

        def edit_upstream_ggml() -> None:
            (self.upstream / "ggml" / "new.h").write_text("", encoding="utf-8")
            git(self.upstream, "add", "ggml/new.h")

        def change_patch() -> None:
            self.write_patch("30")
            commit_all(self.source, "patch")

        def break_patch() -> None:
            (self.upstream / SOURCE).write_text(ORIGINAL.replace("2", "4"), encoding="utf-8")
            commit_all(self.upstream, "upstream moves the patched line")

        mutations = [
            ("owner commit", commit_owner, {"owner"}),
            ("owner edit", edit_owner("int edited;\n"), {"owner"}),
            ("other owner edit", edit_owner("int edited2;\n"), {"owner"}),
            # The new llama.cpp commit also moves the owner's gitlink.
            ("llama.cpp commit", commit_upstream_outside_ggml, {"owner", "llama.cpp"}),
            ("llama.cpp ggml edit", edit_upstream_ggml, {"llama.cpp"}),
            ("patch text", change_patch, {"owner", "patches"}),
            ("patch no longer applies", break_patch,
             {"owner", "llama.cpp", "ggml", "patches"}),
        ]
        mutations += [
            (key, None, {key}) for key in FIELDS
        ]
        seen = {self.identity()}
        for name, mutate, changed in mutations:
            with self.subTest(name):
                snapshot = self.temp / "snapshot"
                shutil.copytree(self.source, snapshot, symlinks=True)
                try:
                    if mutate is None:
                        value = self.identity(**{name: FIELDS[name] + "x"})
                    else:
                        mutate()
                        value = self.identity()
                    mutated = fields_of(value)
                    differs = {key for key in baseline if mutated[key] != baseline[key]}
                    self.assertEqual(differs, changed)
                    self.assertNotIn(value, seen)
                    seen.add(value)
                finally:
                    shutil.rmtree(self.source)
                    snapshot.rename(self.source)
        self.assertEqual(fields_of(self.identity()), baseline)

    def test_dirty_trees_are_marked_with_a_digest_of_the_change(self):
        (self.source / "src" / "wrapper.cpp").write_text("int edited;\n", encoding="utf-8")
        (self.upstream / SOURCE).write_text(ORIGINAL + "int d;\n", encoding="utf-8")
        fields = fields_of(self.identity())
        self.assertRegex(fields["owner"], r"^[0-9a-f]{40}\+dirty\.[0-9a-f]{16}$")
        self.assertRegex(fields["llama.cpp"], r"^[0-9a-f]{40}\+dirty\.[0-9a-f]{16}$")

    def test_untracked_files_do_not_change_it(self):
        baseline = self.identity()
        (self.source / "build").mkdir()
        (self.source / "build" / "output.o").write_text("object", encoding="utf-8")
        (self.upstream / "local.txt").write_text("note", encoding="utf-8")
        self.assertEqual(self.identity(), baseline)

    def test_unidentified_builds_report_an_empty_identity(self):
        exported = self.temp / "exported"
        shutil.copytree(self.source, exported, ignore=shutil.ignore_patterns(".git"))
        nested = self.temp / "nested"
        shutil.copytree(self.source, nested, symlinks=True)
        # Without its own repository, llama.cpp would report the owner's.
        shutil.rmtree(nested / "third_party" / "llama.cpp" / ".git")
        for name, source, path in (
            ("no git checkout", exported, None),
            ("llama.cpp is not a checkout", nested, None),
            ("no git", self.source, str(self.temp / "no-bin")),
        ):
            with self.subTest(name):
                environment = {"PATH": path} if path else {}
                with mock.patch.dict(os.environ, environment), \
                        mock.patch("sys.stderr"):
                    self.assertEqual(self.identity(source), "")

    def test_a_directory_inside_another_checkout_has_no_revision(self):
        with self.assertRaises(build_identity.Unidentified):
            build_identity.revision(self.source / "src")

    def test_git_config_does_not_change_the_dirty_digest(self):
        lines = [f"line {i}" if i % 3 else "" for i in range(40)]
        (self.source / "src" / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.source / "src" / "b.txt").write_text("x\n", encoding="utf-8")
        commit_all(self.source, "files")
        lines[4], lines[16] = "edited 4", "edited 16"
        (self.source / "src" / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.source / "src" / "b.txt").write_text("y\n", encoding="utf-8")
        order = self.temp / "order"
        order.write_text("src/b.txt\nsrc/a.txt\n", encoding="utf-8")
        baseline = self.identity()
        for setting in (
            ("diff.orderFile", str(order)), ("diff.suppressBlankEmpty", "true"),
            ("diff.interHunkContext", "20"), ("diff.context", "9"),
            ("diff.algorithm", "patience"), ("diff.indentHeuristic", "true"),
            ("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"),
            ("diff.renames", "copies"),
        ):
            with self.subTest(setting=setting):
                git(self.source, "config", "--global", *setting)
                try:
                    self.assertEqual(self.identity(), baseline)
                finally:
                    git(self.source, "config", "--global", "--unset", setting[0])

    def test_a_checkout_owned_by_another_user_is_still_identified(self):
        probe = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            env={**os.environ, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe.returncode == 0:
            self.skipTest("this git does not check the checkout's owner")
        with mock.patch.dict(os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}):
            self.assertNotEqual(self.identity(), "")

    def test_a_toplevel_that_cannot_be_compared_is_unidentified(self):
        with mock.patch("os.path.samefile", side_effect=FileNotFoundError("/c/x")):
            with self.assertRaises(build_identity.Unidentified):
                build_identity.revision(self.source)
        with mock.patch("os.path.samefile", side_effect=OSError("bad")), \
                mock.patch("sys.stderr"):
            self.assertEqual(self.identity(), "")

    def test_rejects_values_that_could_hold_a_path(self):
        for key, value in (
            ("target", "/Applications/Xcode.app/MacOSX.sdk"),
            ("compiler", "C:\\clang"),
            ("compiler", "GNU 13"),
            ("options", ""),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.identity(**{key: value})

    def test_header_is_rewritten_only_when_the_identity_changes(self):
        header = self.temp / "generated" / "llama_dart_build_identity.h"
        self.assertTrue(build_identity.write_header(header, "a"))
        self.assertIn('#define LLAMA_DART_BUILD_IDENTITY "a"\n', header.read_text())
        os.utime(header, (1, 1))
        self.assertFalse(build_identity.write_header(header, "a"))
        self.assertEqual(header.stat().st_mtime, 1)
        self.assertTrue(build_identity.write_header(header, "b"))
        self.assertIn('"b"', header.read_text())


if __name__ == "__main__":
    unittest.main()
