#!/usr/bin/env python3
"""Exercise the candidate runtime's public grammar sampler at its size limit.

Qualification only: the production v0.4.0 pin is known to fail the exact 2000
boundary. No model is needed because these grammars contain only characters;
the vocabulary is not accessed until token sampling.
"""

import argparse
import ctypes
from pathlib import Path


def validate(library: Path) -> None:
    runtime = ctypes.CDLL(str(library.resolve()))
    initialize = runtime.llama_sampler_init_grammar
    initialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    initialize.restype = ctypes.c_void_p
    free = runtime.llama_sampler_free
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    for grammar, expected in (
        (b'root ::= "a"{1999}', True),
        (b'root ::= "a"{2000}', True),
        (b'root ::= "a"{2001}', False),
        (b'root ::= "a"{', False),
    ):
        sampler = initialize(None, grammar, b"root")
        accepted = bool(sampler)
        if sampler:
            free(sampler)
        if accepted != expected:
            raise RuntimeError(
                f"grammar boundary mismatch: {grammar!r}: "
                f"accepted={accepted}, expected={expected}"
            )
    print("Validated grammar sampler acceptance/rejection at repetition 2000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    validate(parser.parse_args().library)
