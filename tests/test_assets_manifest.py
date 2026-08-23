from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AssetsManifestTests(unittest.TestCase):
    def test_dynamic_metadata_is_json_escaped(self) -> None:
        values = {
            "LLAMADART_LLAMA_CPP_TAG": 'tag"\\\nvalue',
            "LLAMADART_LLAMA_CPP_COMMIT": 'commit"\\\nvalue',
            "LLAMADART_NATIVE_COMMIT": 'native"\\\nvalue',
            "LLAMADART_CORRELATION_ID": 'correlation"\\\nvalue',
            "LLAMADART_SMOKE_POLICY": 'policy"\\\nvalue',
            "LLAMADART_SMOKE_CONCLUSION": 'conclusion"\\\nvalue',
            "LLAMADART_WORKFLOW_RUN_ID": 'run"\\\nvalue',
            "LLAMADART_WORKFLOW_RUN_URL": 'url"\\\nvalue',
            "LLAMADART_WORKFLOW_HEAD_SHA": 'head"\\\nvalue',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "fixture.bin").write_bytes(b"fixture")
            output = root / "assets.json"
            checksums = root / "SHA256SUMS"
            environment = os.environ.copy()
            environment.update(values)
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/generate_assets_manifest.sh"),
                    "v0.2.0",
                    str(assets),
                    str(output),
                    str(checksums),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))

        for environment_name, expected in values.items():
            manifest_name = environment_name.removeprefix("LLAMADART_").lower()
            self.assertEqual(expected, manifest[manifest_name])
