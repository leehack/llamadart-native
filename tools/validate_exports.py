#!/usr/bin/env python3
"""Validate that a native library exports required C symbols."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_REQUIRED_SYMBOLS = [
    "llama_dart_last_error",
    "llama_dart_clear_last_error",
    "llama_dart_vulkan_get_device_count",
    "llama_dart_vulkan_get_device_info",
    "llama_dart_exit_track",
    "llama_dart_exit_untrack",
    "llama_dart_exit_free",
    "llama_dart_exit_tracked_count",
    "llama_dart_exit_call_begin",
    "llama_dart_exit_call_end",
    "llama_dart_exit_set_wait_ms",
    "llama_dart_exit_teardown",
    "llama_dart_model_load_from_file",
    "llama_dart_init_from_model",
    "llama_dart_mtmd_init_from_file",
    "llama_dart_mtmd_supports_image_token_limit",
    "llama_dart_decode",
    "llama_dart_encode",
    "llama_dart_synchronize",
    "llama_dart_sampler_sample",
    "llama_dart_sampler_accept",
    "llama_dart_sampler_init_grammar_lazy_patterns",
    "llama_dart_tokenize",
    "llama_dart_token_to_piece",
    "llama_dart_memory_clear",
    "llama_dart_state_save_file",
    "llama_dart_state_load_file",
    "llama_dart_state_seq_get_size_ext",
    "llama_dart_state_seq_get_data_ext",
    "llama_dart_state_seq_set_data_ext",
    "llama_dart_adapter_lora_init",
    "llama_dart_mtmd_bitmap_init_from_audio",
    "llama_dart_mtmd_bitmap_init_from_buf",
    "llama_dart_mtmd_bitmap_init_from_file",
    "llama_dart_mtmd_tokenize",
    "llama_dart_mtmd_encode_chunk",
    "llama_dart_mtmd_helper_eval_chunks",
    "llama_dart_mtmd_helper_eval_chunk_single",
    "llama_dart_mtmd_helper_decode_image_chunk",
    "llama_dart_ggml_backend_sched_graph_compute",
    "llama_dart_ggml_backend_dev_init",
    "llama_dart_ggml_backend_dev_memory",
    "llama_dart_ggml_backend_dev_get_props",
    "llama_dart_ggml_backend_alloc_ctx_tensors",
    "llama_dart_ggml_backend_tensor_set",
    "llama_dart_ggml_backend_tensor_get",
    "llama_dart_ggml_backend_sched_alloc_graph",
    "llama_dart_ggml_backend_sched_synchronize",
    "llama_dart_tts_api_version",
    "llama_dart_tts_request_default",
    "llama_dart_tts_get_info",
    "llama_dart_tts_init",
    "llama_dart_tts_free",
    "llama_dart_tts_start",
    "llama_dart_tts_step",
    "llama_dart_tts_cancel",
    "llama_dart_tts_set_cancel_flag",
    "llama_dart_tts_eval_callback",
    "llama_dart_tts_reset",
    "llama_dart_tts_get_output_info",
    "llama_dart_tts_read_pcm",
    "llama_dart_tts_last_error",
    "llama_dart_speculative_init",
    "llama_dart_speculative_free",
    "llama_dart_speculative_get_draft_context",
    "llama_dart_speculative_need_embd",
    "llama_dart_speculative_need_embd_nextn",
    "llama_dart_speculative_begin",
    "llama_dart_speculative_process_batch",
    "llama_dart_speculative_draft",
    "llama_dart_speculative_accept",
    "llama_dart_mtp_init",
    "llama_dart_mtp_init_with_draft_model",
    "llama_dart_mtp_free",
    "llama_dart_mtp_get_draft_context",
    "llama_dart_mtp_begin",
    "llama_dart_mtp_process_batch",
    "llama_dart_mtp_draft",
    "llama_dart_mtp_accept",
    "llama_dart_ngram_simple_init",
    "llama_dart_ngram_free",
    "llama_dart_ngram_begin",
    "llama_dart_ngram_process_batch",
    "llama_dart_ngram_draft",
    "llama_dart_ngram_accept",
    "llama_dart_sampler_sample_and_accept_n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="Native library to inspect")
    parser.add_argument(
        "--format",
        choices=("readelf", "nm", "dumpbin"),
        required=True,
        help="Symbol tool output format",
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Symbol inspection executable, such as readelf, nm, or dumpbin",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Required symbol. Defaults to the llamadart wrapper export set.",
    )
    parser.add_argument(
        "--forbid-import",
        action="append",
        dest="forbidden_imports",
        default=[],
        help=(
            "Symbol the library must not import. Supported for --format nm "
            "and single-architecture files."
        ),
    )
    return parser.parse_args()


_UNIVERSAL_MAGICS = (
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
)


def is_universal_binary(library: Path) -> bool:
    with library.open("rb") as file:
        return file.read(4) in _UNIVERSAL_MAGICS


def run_tool(args: argparse.Namespace) -> str:
    if not args.library.is_file():
        raise FileNotFoundError(f"Missing native library: {args.library}")

    command = {
        "readelf": [args.tool, "-Ws", str(args.library)],
        "nm": [args.tool, "-g", str(args.library)],
        "dumpbin": [args.tool, "/EXPORTS", str(args.library)],
    }[args.format]

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def exported_symbols_from_readelf(output: str) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0].rstrip(":") == "Num":
            continue
        if parts[6] == "UND":
            continue
        name = parts[7].split("@", 1)[0]
        symbols.add(name)
    return symbols


def exported_symbols_from_nm(output: str) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[-1]
        if len(parts) >= 2 and parts[-2].upper() == "U":
            continue
        symbols.add(symbol.removeprefix("_"))
    return symbols


def imported_symbols_from_nm(output: str) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2].upper() == "U":
            symbols.add(parts[-1].removeprefix("_"))
    return symbols


_DUMPBIN_EXPORT_RE = re.compile(
    r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)"
)


def exported_symbols_from_dumpbin(output: str) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        match = _DUMPBIN_EXPORT_RE.match(line)
        if match:
            symbols.add(match.group(1))
    return symbols


def main() -> int:
    args = parse_args()
    if args.forbidden_imports and args.format != "nm":
        print("--forbid-import requires --format nm", file=sys.stderr)
        return 2
    # nm lists only the host architecture of a universal file, so the other
    # slices would go unchecked.
    if args.forbidden_imports and is_universal_binary(args.library):
        print(
            f"--forbid-import needs a single-architecture file: {args.library}",
            file=sys.stderr,
        )
        return 2
    output = run_tool(args)
    exported = {
        "readelf": exported_symbols_from_readelf,
        "nm": exported_symbols_from_nm,
        "dumpbin": exported_symbols_from_dumpbin,
    }[args.format](output)
    required = args.symbols or DEFAULT_REQUIRED_SYMBOLS
    missing = [symbol for symbol in required if symbol not in exported]
    if missing:
        print(f"Missing exports in {args.library}:", file=sys.stderr)
        for symbol in missing:
            print(f"  - {symbol}", file=sys.stderr)
        return 1

    imported = imported_symbols_from_nm(output)
    forbidden = [s for s in args.forbidden_imports if s in imported]
    if forbidden:
        print(f"Forbidden imports in {args.library}:", file=sys.stderr)
        for symbol in forbidden:
            print(f"  - {symbol}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(required)} required export(s) in {args.library}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
