#!/usr/bin/env python3
"""Audit the non-SVE Android armv8.2 CPU artifact, including dispatched KleidiAI.

This is a bounded SVE/SME containment check, not a general ISA verifier or a
proof of runtime reachability. PR qualification runs the compiled selectors and
matmuls; release fingerprints bind the prequalified source. Real-device matmul
evidence is a separate qualification layer.
Source fingerprints deliberately fail closed on upstream changes: do not refresh
them without reviewing feature detection, every selector/table and its callers.
"""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys


# llama.cpp v0.4.0 / KleidiAI v1.24.0. Bind the complete source subtrees, not
# merely a selector fragment: a new caller can invalidate a dispatch audit.
SOURCE_TREES = {
    "ggml/src": "c4dc92a7d95ebfad7f5f55e75be2ae773b7d95faf72a9581c9479c42bc41bca0",
    "kai": "64189fc613c1c4c3aaeeb6bb12b38d85dd6728cafd2261a5a88f1b77b10fe59c",
}

# Exact ELF STT_FUNC ranges; never allow by kai_* prefix or disassembly label.
# Assembly kernels have nested local labels, which are NOT function boundaries.
# Vector-length helpers are reached by the selected kernels' size/packing APIs.
DISPATCHED_FUNCTIONS = frozenset("""
kai_get_sme_vector_length_u8
kai_get_sve_vector_length_u8
kai_kernel_lhs_pack_f32p2vlx1_f32_sme
kai_kernel_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa
kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa
kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla
kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa
kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa
kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa
kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot
kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot
kai_kernel_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod
kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm
kai_kernel_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme
kai_run_lhs_pack_bf16p2vlx2_f32_sme
kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot
kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme
""".split())

SCALABLE_REG = re.compile(r"\b(?:z\d+|p\d+|pn\d+|za\d*[hv]?|zt\d+)\b")
SCALABLE_SCALAR = re.compile(
    r"^(?:add(?:s?v|s?p)l|rd(?:s?v)l|cnts?[bhwd]|(?:sq|uq)?(?:inc|dec)[bhwd]|"
    r"setffr|smstart|smstop)$"
)
SYMBOL = re.compile(
    r"\s*\d+:\s+([0-9a-fA-F]+)\s+(\d+)\s+FUNC\s+\S+\s+\S+\s+(\S+)\s+(\S+)"
)
INSTRUCTION = re.compile(r"\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s+(.+)")


def tree_digest(root):
    root = Path(root)
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"Missing or empty audited source tree: {root}")
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink in audited source tree: {path}")
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def validate_sources(llama_source, kleidiai_source):
    for base, subtree in ((llama_source, "ggml/src"), (kleidiai_source, "kai")):
        actual = tree_digest(Path(base) / subtree)
        if actual != SOURCE_TREES[subtree]:
            raise ValueError(f"Unaudited {subtree} source fingerprint {actual}; review dispatch before updating policy")


def function_ranges(symbol_text):
    result = []
    for line in symbol_text.splitlines():
        match = SYMBOL.fullmatch(line)
        if match and match[3] != "UND" and int(match[2]) > 0:
            start, size = int(match[1], 16), int(match[2])
            if start % 4 or size % 4:
                raise ValueError(f"Unaligned AArch64 function: {match[4]}")
            result.append((start, start + size, match[4]))
    if not result:
        raise ValueError("No defined, sized ELF function symbols")
    return result


def is_scalable(word, instruction):
    # Arm's SVE major encoding group catches all SVE forms, including scalar
    # operands and aliases. SME also occupies other groups, so inspect its
    # registers and streaming-mode/vector-length scalar instructions below.
    return ((word & 0x1E000000) == 0x04000000 or
            bool(SCALABLE_REG.search(instruction.split("//", 1)[0])) or
            bool(SCALABLE_SCALAR.fullmatch(instruction.split()[0])))


def validate_disassembly(symbol_text, disassembly, allowed=DISPATCHED_FUNCTIONS):
    if "file format elf64-littleaarch64" not in disassembly:
        raise ValueError("Expected an AArch64 ELF artifact")
    ranges = function_ranges(symbol_text)
    missing = allowed - {name for _, _, name in ranges}
    if missing:
        raise ValueError("Missing audited optimized functions: " + ", ".join(sorted(missing)))
    seen_addresses = set()
    advanced = {}
    failures = []
    for line in disassembly.splitlines():
        match = INSTRUCTION.fullmatch(line)
        if not match:
            # Do not silently ignore changed tool formatting or undecoded data.
            if re.match(r"\s*[0-9a-fA-F]+:", line):
                raise ValueError(f"Unparsed instruction: {line.strip()}")
            continue
        address, word = int(match[1], 16), int(match[2], 16)
        instruction = match[3].strip()
        if address in seen_addresses or address % 4:
            raise ValueError(f"Duplicate or unaligned instruction address: {address:x}")
        seen_addresses.add(address)
        if "<unknown>" in instruction or instruction.startswith("."):
            raise ValueError(f"Undecoded instruction at {address:x}: {instruction}")
        if not is_scalable(word, instruction):
            continue
        owners = [name for start, end, name in ranges if start <= address < end]
        if len(owners) != 1 or owners[0] not in allowed:
            failures.append(f"0x{address:x}: {instruction} ({', '.join(owners) or 'no sized function'})")
        else:
            advanced[owners[0]] = advanced.get(owners[0], 0) + 1
    if not seen_addresses:
        raise ValueError("Empty disassembly")
    if failures:
        raise ValueError("SVE/SME escaped audited dispatch ranges:\n" + "\n".join(failures[:20]))
    if allowed - advanced.keys():
        raise ValueError("Audited optimized function lost its scalable instructions: " +
                         ", ".join(sorted(allowed - advanced.keys())))
    return len(seen_addresses), sum(advanced.values()), len(advanced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--kleidiai-source", required=True, type=Path)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    try:
        symbols = subprocess.check_output([args.readelf, "--dyn-syms", "--wide", str(args.library)], text=True)
        # Decode extensions rather than letting objdump hide them as <unknown>.
        disassembly = subprocess.check_output([
            args.objdump, "-d", "--mattr=+v9.4a,+sve,+sve2,+sme,+sme2,+mte", str(args.library)
        ], text=True)
        has_scalable = any(
            is_scalable(int(match[2], 16), match[3].strip())
            for line in disassembly.splitlines()
            if (match := INSTRUCTION.fullmatch(line))
        )
        if has_scalable:
            validate_sources(args.llama_source, args.kleidiai_source)
        total, scalable, functions = validate_disassembly(
            symbols, disassembly, DISPATCHED_FUNCTIONS if has_scalable else frozenset()
        )
        print(f"PASS: {total} instructions; {scalable} SVE/SME instructions contained in {functions} exact audited functions")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Android CPU ISA audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
