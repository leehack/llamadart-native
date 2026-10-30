import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import expected_tts_model_types as expected

HEADER = ROOT / 'src/llama_dart_wrapper.h'
QWEN3, LFM3, SOPRANO = 1 << 1, 1 << 2, 1 << 3

MTMD = 'enum mtmd_gen_audio_type {\n    MTMD_GEN_AUDIO_TYPE_NONE = 0,\n    MTMD_GEN_AUDIO_TYPE_QWEN3TTS,\n%s};\n'


def checkout(root: Path, *, soprano=False, lfm3_enum=False, helper=False, prompt=False) -> Path:
    enums = ('    MTMD_GEN_AUDIO_TYPE_SOPRANO,\n' if soprano else '') + (
        '    MTMD_GEN_AUDIO_TYPE_LFM3A,\n' if lfm3_enum else '')
    for name, text in {
        'tools/mtmd/mtmd.h': MTMD % enums,
        'tools/mtmd/mtmd-helper.h': 'bool mtmd_helper_gen_audio_next_is_text(void*);\n' if helper else '',
        'common/chat.h': 'std::string common_chat_get_tts_prompt();\n' if prompt else '',
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class ExpectedTtsModelTypesTest(unittest.TestCase):
    def mask(self, **features) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            return expected.expected_model_types(checkout(Path(tmp), **features), HEADER)

    def test_a_checkout_without_optional_generators_reports_qwen3_only(self) -> None:
        self.assertEqual(QWEN3, self.mask())

    def test_soprano_follows_its_enumerator(self) -> None:
        self.assertEqual(QWEN3 | SOPRANO, self.mask(soprano=True))

    def test_lfm3_needs_the_enumerator_the_helper_and_the_prompt_function(self) -> None:
        self.assertEqual(QWEN3 | LFM3, self.mask(lfm3_enum=True, helper=True, prompt=True))
        for missing in ({'helper': True, 'prompt': True}, {'lfm3_enum': True, 'prompt': True}, {'lfm3_enum': True, 'helper': True}):
            self.assertEqual(QWEN3, self.mask(**missing), missing)

    def test_the_v080_shape_reports_the_mask_the_macos_and_android_bundles_do(self) -> None:
        self.assertEqual(0xE, self.mask(soprano=True, lfm3_enum=True, helper=True, prompt=True))

    def test_a_moved_mtmd_header_is_an_error_not_an_empty_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = checkout(Path(tmp))
            (root / 'tools/mtmd/mtmd.h').write_text('')
            with self.assertRaises(SystemExit):
                expected.expected_model_types(root, HEADER)

    def test_the_command_prints_the_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = checkout(Path(tmp), soprano=True)
            out = subprocess.run([sys.executable, str(ROOT / 'tools/expected_tts_model_types.py'), str(root)],
                                 check=True, capture_output=True, text=True).stdout
        self.assertEqual('0xa\n', out)


class PinnedCheckoutTest(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'third_party/llama.cpp/tools/mtmd/mtmd.h').exists(), 'no llama.cpp checkout')
    def test_the_checked_out_llama_cpp_has_a_mask_with_qwen3(self) -> None:
        mask = expected.expected_model_types(ROOT / 'third_party/llama.cpp', HEADER)
        self.assertTrue(mask & QWEN3)


if __name__ == '__main__':
    unittest.main()
