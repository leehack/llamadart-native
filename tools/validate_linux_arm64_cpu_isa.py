#!/usr/bin/env python3
"""Audit a Linux arm64 CPU backend against the ISA its variant declares.

Bounded containment check for the feature families ggml variants are scored
on, not a general ISA verifier:

- Outside sized ``kai_*`` functions, FP16 arithmetic, FP16FML, DOTPROD, I8MM,
  BF16, RDM, LSE, CRC, LRCPC and LRCPC2 instructions may only appear when the
  variant's ``-march`` enables them. The armv8.0 baseline allows none of them,
  except LSE atomics inside libgcc's outline-atomic helpers, which branch to
  them only after testing ``__aarch64_have_lse_atomics`` at run time. objdump
  does not name that flag in the guard, so the guard's flag address must equal
  the symbol's when the library keeps ``.symtab``, and every guard must test
  one address either way. KleidiAI
  kernels keep them because KleidiAI selects kernels from the runtime HWCAP
  probe (``ggml_feats_get_arch64_runtime``), never from the compile flags.
- SVE and SME stay inside the exact audited KleidiAI functions of
  ``validate_android_cpu_isa.py``, for the same pinned source pairs.
- The family lists are a denylist: an instruction of an ARMv8.3+ extension
  that is not listed here (for example FCMA or JSCVT) passes, and only the
  QEMU smoke catches it.
- A variant must contain each family it declares outside KleidiAI, so its
  ``-march`` took effect, and must export ``ggml_backend_score``.

Loading every variant on lower CPUs is checked by execution under QEMU.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build  # noqa: E402
import validate_android_cpu_isa as android_isa  # noqa: E402


BASELINE = "baseline"
BASELINE_MARCH = "armv8-a"
FEATURE_FAMILIES = {
    "GGML_USE_DOTPROD": "dotprod",
    "GGML_USE_FP16_VECTOR_ARITHMETIC": "fp16",
    "GGML_USE_MATMUL_INT8": "i8mm",
}
# Families GCC enables from the -march base version alone.
MARCH_IMPLIED_FAMILIES = (
    ((8, 1), {"rdm", "lse"}),
    ((8, 3), {"rcpc"}),
    ((8, 4), {"dotprod", "rcpc2"}),
    ((8, 6), {"bf16", "i8mm"}),
)
HALF_OPERAND = re.compile(r"\bv\d+\.(?:4|8)h\b|\bv\d+\.h\[|\bh\d+\b")
# Base ARMv8.0 converts between half and wider precision.
BASE_HALF_CONVERSIONS = frozenset({"fcvt", "fcvtl", "fcvtl2", "fcvtn", "fcvtn2"})
MNEMONIC_FAMILIES = {
    **{name: "dotprod" for name in ("sdot", "udot")},
    **{name: "i8mm" for name in ("smmla", "ummla", "usmmla", "usdot", "sudot")},
    **{name: "bf16" for name in ("bfdot", "bfmmla", "bfmlalb", "bfmlalt", "bfcvt", "bfcvtn", "bfcvtn2")},
    **{name: "fhm" for name in ("fmlal", "fmlal2", "fmlsl", "fmlsl2")},
    **{name: "rdm" for name in ("sqrdmlah", "sqrdmlsh")},
    **{name: "rcpc" for name in ("ldapr", "ldaprb", "ldaprh")},
}
LSE_MNEMONIC = re.compile(
    r"(?:casp?|swp|ld(?:add|clr|eor|set|smax|smin|umax|umin)|st(?:add|clr|eor|set|smax|smin|umax|umin))"
    r"(?:a|l|al)?[bh]?"
)
CRC_MNEMONIC = re.compile(r"crc32c?[bhwx]")
# libgcc's outline-atomic helpers open with these three instructions before
# the single LSE instruction, then fall back to an exclusive-access loop.
OUTLINE_ATOMIC_GUARD = (
    re.compile(r"adrp\s+x16,\s*(0x[0-9a-fA-F]+)"),
    re.compile(r"ldrb\s+w16,\s*\[x16(?:,\s*#(0x[0-9a-fA-F]+|\d+))?\]"),
    re.compile(r"cbz\s+w16,"),
)
LSE_FLAG_SYMBOL = "__aarch64_have_lse_atomics"
FAMILIES = ("fp16", "fhm", "dotprod", "i8mm", "bf16", "rdm", "lse", "rcpc", "rcpc2", "crc")
MARCH = re.compile(r"armv(\d+)(?:\.(\d+))?-a((?:\+[a-z0-9]+)*)")


def variant_spec(name: str) -> tuple[str, tuple[str, ...]]:
    if name == BASELINE:
        return BASELINE_MARCH, ()
    for variant_name, march, features in build.LINUX_ARM64_CPU_VARIANTS:
        if variant_name == name:
            return march, features
    raise ValueError(f"Unknown Linux arm64 CPU variant: {name}")


def allowed_families(march: str, features: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """The families a variant declares, and every family it may contain."""
    match = MARCH.fullmatch(march)
    if not match:
        raise ValueError(f"Unsupported -march value: {march}")
    version = (int(match[1]), int(match[2] or 0))
    declared = {FEATURE_FAMILIES[feature] for feature in features}
    allowed = set(declared)
    for minimum, families in MARCH_IMPLIED_FAMILIES:
        if version >= minimum:
            allowed |= families
    if "fp16" in allowed and version >= (8, 4):
        allowed.add("fhm")
    return declared, allowed


def instruction_family(instruction: str) -> str | None:
    parts = instruction.split("//", 1)[0].split(None, 1)
    mnemonic = parts[0]
    operands = parts[1] if len(parts) > 1 else ""
    family = MNEMONIC_FAMILIES.get(mnemonic)
    if family:
        return family
    if mnemonic.startswith(("stlur", "ldapur")):
        return "rcpc2"
    if LSE_MNEMONIC.fullmatch(mnemonic):
        return "lse"
    if CRC_MNEMONIC.fullmatch(mnemonic):
        return "crc"
    if HALF_OPERAND.search(operands) and (
        (mnemonic.startswith("f") and mnemonic not in BASE_HALF_CONVERSIONS)
        or mnemonic in ("scvtf", "ucvtf")
    ):
        return "fp16"
    return None


def outline_atomic_flag_address(preceding: list[str]) -> int | None:
    """The flag address an outline-atomic guard tests, or None when it is no guard."""
    if len(preceding) != len(OUTLINE_ATOMIC_GUARD):
        return None
    matches = [guard.match(instruction) for guard, instruction in zip(OUTLINE_ATOMIC_GUARD, preceding)]
    if not all(matches):
        return None
    return int(matches[0][1], 16) + int(matches[1][1] or "0", 0)


def lse_flag_address(symtab_text: str) -> int | None:
    pattern = rf"^\s*\d+:\s+([0-9a-fA-F]+)\s+\d+\s+OBJECT\s.*\s{LSE_FLAG_SYMBOL}$"
    match = re.search(pattern, symtab_text, re.MULTILINE)
    return int(match[1], 16) if match else None


def audit(
    symbol_text: str, disassembly: str, allowed: set[str], flag_address: int | None = None
) -> tuple[dict, dict, list[str]]:
    kai_ranges = [
        (start, end) for start, end, name in android_isa.function_ranges(symbol_text) if name.startswith("kai_")
    ]
    inside: dict[str, int] = {}
    outside: dict[str, int] = {}
    failures: list[str] = []
    preceding: deque[str] = deque(maxlen=len(OUTLINE_ATOMIC_GUARD))
    guard_addresses: set[int] = set()
    for line in disassembly.splitlines():
        match = android_isa.INSTRUCTION.fullmatch(line)
        if not match:
            continue
        instruction = match[3].strip()
        previous = list(preceding)
        preceding.append(instruction)
        if android_isa.is_scalable(int(match[2], 16), instruction):
            continue
        family = instruction_family(instruction)
        if family is None:
            continue
        address = int(match[1], 16)
        if any(start <= address < end for start, end in kai_ranges):
            inside[family] = inside.get(family, 0) + 1
            continue
        outside[family] = outside.get(family, 0) + 1
        if family in allowed:
            continue
        guard_address = outline_atomic_flag_address(previous) if family == "lse" else None
        if guard_address is None:
            failures.append(f"0x{address:x}: {instruction} ({family})")
        elif flag_address is not None and guard_address != flag_address:
            failures.append(
                f"0x{address:x}: {instruction} (lse, guard tests 0x{guard_address:x} "
                f"not {LSE_FLAG_SYMBOL} at 0x{flag_address:x})"
            )
        else:
            guard_addresses.add(guard_address)
    if len(guard_addresses) > 1:
        failures.append(
            "outline-atomic guards test different flags: " + ", ".join(f"0x{a:x}" for a in sorted(guard_addresses))
        )
    return inside, outside, failures


def validate(
    variant: str,
    symbol_text: str,
    disassembly: str,
    scalable_allowed: frozenset[str],
    flag_address: int | None = None,
) -> str:
    march, features = variant_spec(variant)
    declared, allowed = allowed_families(march, features)
    total, scalable, scalable_functions = android_isa.validate_disassembly(
        symbol_text, disassembly, scalable_allowed
    )
    inside, outside, failures = audit(symbol_text, disassembly, allowed, flag_address)
    if failures:
        raise ValueError(
            f"{variant} ({march}) contains ISA outside KleidiAI that it does not declare:\n"
            + "\n".join(failures[:20])
        )
    missing = sorted(family for family in declared if not outside.get(family))
    if missing:
        raise ValueError(f"{variant} ({march}) lacks declared ISA outside KleidiAI: {', '.join(missing)}")
    if variant != BASELINE and not re.search(r"\sFUNC\s.*\sggml_backend_score$", symbol_text, re.MULTILINE):
        raise ValueError(f"{variant} does not export ggml_backend_score")
    counts = ", ".join(
        f"{family}={outside.get(family, 0)}/{inside.get(family, 0)}"
        for family in FAMILIES
        if outside.get(family) or inside.get(family)
    )
    return (
        f"PASS {variant} ({march}): {total} instructions; outside/inside KleidiAI: {counts or 'none'}; "
        f"{scalable} SVE/SME instructions in {scalable_functions} audited functions"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--kleidiai-source", required=True, type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=[BASELINE, *(name for name, _march, _features in build.LINUX_ARM64_CPU_VARIANTS)],
    )
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    try:
        symbols = subprocess.check_output([args.readelf, "--dyn-syms", "--wide", str(args.library)], text=True)
        flag_address = lse_flag_address(
            subprocess.check_output([args.readelf, "--syms", "--wide", str(args.library)], text=True)
        )
        disassembly = subprocess.check_output(
            [
                args.objdump,
                "-d",
                "--mattr=+v9.4a,+sve,+sve2,+sme,+sme2,+fullfp16,+fp16fml,+dotprod,+i8mm,+bf16,+rdm,+rcpc-immo,+rcpc,+lse,+crc",
                str(args.library),
            ],
            text=True,
        )
        has_scalable = any(
            android_isa.is_scalable(int(match[2], 16), match[3].strip())
            for line in disassembly.splitlines()
            if (match := android_isa.INSTRUCTION.fullmatch(line))
        )
        scalable_allowed = (
            android_isa.validate_sources(args.llama_source, args.kleidiai_source) if has_scalable else frozenset()
        )
        print(validate(args.variant, symbols, disassembly, scalable_allowed, flag_address))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Linux arm64 CPU ISA audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
