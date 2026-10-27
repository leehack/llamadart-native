import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_build_identity  # noqa: E402

IDENTITY = (
    "llamadart-native-build/1;owner=" + "a" * 40 + ";llama.cpp=" + "b" * 40
    + ";ggml=" + "c" * 40 + ";patches=none;config=Release;compiler=Clang-17,Clang-17"
    ";target=Linux-x86_64;backends=cpu;options=0123456789abcdef"
)


class ValidateBuildIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)

    def library(self, *strings: str) -> Path:
        path = self.temp / "libllamadart.bin"
        path.write_bytes(b"\0".join(s.encode() for s in ("junk", *strings, "tail")))
        return path

    def test_accepts_a_complete_identity_including_a_dirty_one(self):
        for value in (IDENTITY, IDENTITY.replace("a" * 40, "a" * 40 + "+dirty.0123456789abcdef")):
            with self.subTest(value=value[:60]):
                self.assertEqual(validate_build_identity.check(self.library(value)), [])

    def test_rejects_a_library_without_an_identity(self):
        for strings in ((), ("llamadart-native-build/1",), ("",)):
            with self.subTest(strings=strings):
                self.assertNotEqual(validate_build_identity.check(self.library(*strings)), [])

    def test_rejects_an_incomplete_identity(self):
        for value in (
            IDENTITY.replace("owner=" + "a" * 40, "owner="),
            IDENTITY.replace(";options=0123456789abcdef", ""),
            IDENTITY.replace(";ggml=", ";gg=").replace("c" * 40, "c"),
        ):
            with self.subTest(value=value[:60]):
                self.assertNotEqual(validate_build_identity.check(self.library(value)), [])

    def test_release_lanes_check_every_library_they_validate_exports_of(self):
        text = (ROOT / ".github" / "workflows" / "native_release.yml").read_text(encoding="utf-8")
        self.assertGreater(text.count("tools/validate_exports.py"), 0)
        self.assertEqual(
            text.count("tools/validate_build_identity.py"),
            text.count("tools/validate_exports.py"))

    def test_command_line_exit_status(self):
        def run(path: Path) -> int:
            return subprocess.run(
                [sys.executable, str(ROOT / "tools" / "validate_build_identity.py"), str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
        self.assertEqual(run(self.library(IDENTITY)), 0)
        self.assertEqual(run(self.library()), 1)


if __name__ == "__main__":
    unittest.main()
