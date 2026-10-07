"""Shared helpers for the tools/gen_public_vectors_*.py scripts.

The scripts write reference vectors for the Rust tests of `oura-analysis` from the Python ports in
this directory, on random synthetic inputs. They import only the Python ports and read no captured data.
"""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "crates/oura-analysis/src/ported/vectors"

FNV_OFFSET, FNV_PRIME, M64 = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1


def fnv(data, h=FNV_OFFSET):
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & M64
    return h


def rle(xs):
    """Run-length encoding as a Rust list body: (value, count) pairs."""
    return ", ".join(f"({k}, {len(list(g))})" for k, g in itertools.groupby(xs))


def nums(xs):
    return ", ".join(str(x) for x in xs)


def rust_bool(b):
    return "true" if b else "false"


def write(name, text):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.rs"
    path.write_text(text)
    print(f"{path.name}: {path.stat().st_size} bytes")


