#!/usr/bin/env python3
"""Print the TTS model types a wrapper built against a llama.cpp checkout reports.

Reads the upstream headers instead of the CMake probes that gate the same
features, so a broken probe cannot make the expectation wrong with it
(llamadart-native#146).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    ("SOPRANO", (("tools/mtmd/mtmd.h", r"^\s*MTMD_GEN_AUDIO_TYPE_SOPRANO\s*[,=]"),)),
    (
        "LFM3_AUDIO",
        (
            ("tools/mtmd/mtmd.h", r"^\s*MTMD_GEN_AUDIO_TYPE_LFM3A\s*[,=]"),
            ("tools/mtmd/mtmd-helper.h", r"\bmtmd_helper_gen_audio_next_is_text\b"),
            ("common/chat.h", r"\bcommon_chat_get_tts_prompt\b"),
        ),
    ),
)


def model_type_values(wrapper_header: Path) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(
            r"LLAMA_DART_TTS_MODEL_TYPE_(\w+)\s*=\s*(\d+)", wrapper_header.read_text()
        )
    }


def expected_model_types(llama_cpp: Path, wrapper_header: Path) -> int:
    values = model_type_values(wrapper_header)
    mtmd = (llama_cpp / "tools/mtmd/mtmd.h").read_text()
    # An anchor, so that a moved header cannot pass as "no optional types".
    if "MTMD_GEN_AUDIO_TYPE_QWEN3TTS" not in mtmd:
        raise SystemExit(f"{llama_cpp}: MTMD_GEN_AUDIO_TYPE_QWEN3TTS not found in mtmd.h")
    mask = 1 << values["QWEN3"]
    for name, needles in FEATURES:
        if all(
            re.search(pattern, (llama_cpp / header).read_text(), re.MULTILINE)
            for header, pattern in needles
        ):
            mask |= 1 << values[name]
    return mask


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_cpp", nargs="?", type=Path, default=ROOT / "third_party/llama.cpp")
    parser.add_argument("--wrapper-header", type=Path, default=ROOT / "src/llama_dart_wrapper.h")
    args = parser.parse_args(argv)
    print(f"0x{expected_model_types(args.llama_cpp, args.wrapper_header):x}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
