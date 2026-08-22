from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from smoke_linux_bundle import smoke  # noqa: E402


class LinuxBundleSmokeTests(unittest.TestCase):
    def test_uses_current_interpreter_and_absolute_bundle_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            bundle.mkdir()
            wrapper = bundle / "libllamadart.so"
            wrapper.touch()
            unresolved_bundle = bundle / ".." / "bundle"

            with patch(
                "smoke_linux_bundle.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "probe passed\n", ""),
            ) as run:
                smoke(unresolved_bundle)

            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertEqual(sys.executable, command[0])
            self.assertEqual(str(wrapper.resolve()), command[3])
            self.assertEqual(str(bundle.resolve()), environment["LD_LIBRARY_PATH"])


if __name__ == "__main__":
    unittest.main()
