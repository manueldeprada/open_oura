#!/usr/bin/env python3
"""Write the weights file that `StepCounter::from_bytes` reads, from a user-supplied model.

Reads a serialized model archive that you supply (the step counter network, loaded with
`torch.jit.load`) and writes the binary layout documented in
crates/oura-analysis/src/ported/step_counter.rs. Nothing is bundled with this repository, and
the weights stay on your machine.

Tensors are matched by shape and order, not by name. The state dict is scanned for the first
chain of batch norm and linear layers with widths 19, 32 (seven times), 6, then the first chain
with widths 6, 4 (four times), then the first (1, 4) linear layer. `beta` and `tau` are the
scalar tensors whose names end in those words.

Usage: python3 tools/export_step_counter_weights.py MODEL_FILE OUT_FILE
Needs torch.
"""
import sys

import numpy as np

MAGIC = b"STP1"
DIMS_A = [19] + [32] * 7 + [6]
DIMS_B = [6, 4, 4, 4, 4]


def tensors(model):
    """State dict entries in order, without the batch norm step counters."""
    return [
        (k, v.detach().numpy().astype("<f4"))
        for k, v in model.state_dict().items()
        if not k.endswith("num_batches_tracked")
    ]


def parse_mlp(entries, pos, dims):
    """Parts of one MLP starting at `pos`, and the next position; None if the shapes differ."""
    parts = []

    def take(shape):
        nonlocal pos
        if pos >= len(entries) or entries[pos][1].shape != shape:
            raise ValueError
        parts.append(entries[pos][1].ravel())
        pos += 1

    def batch_norm(n):
        for _ in range(4):  # weight, bias, running mean, running variance
            take((n,))

    try:
        batch_norm(dims[0])
        for n_in, n_out in zip(dims, dims[1:]):
            take((n_out, n_in))
            take((n_out,))
            batch_norm(n_out)
    except ValueError:
        return None
    return parts, pos


def find_mlp(entries, dims):
    for pos in range(len(entries)):
        found = parse_mlp(entries, pos, dims)
        if found:
            return found[0]
    raise SystemExit(f"no layer chain with widths {dims} in the model")


def export(model):
    entries = tensors(model)
    core = find_mlp(entries, DIMS_A)
    second = find_mlp(entries, DIMS_B)
    head = next(
        (i for i, (_, v) in enumerate(entries) if v.shape == (1, 4)), None
    )
    if head is None:
        raise SystemExit("no (1, 4) linear layer in the model")
    scalars = {k.rsplit(".", 1)[-1]: v for k, v in entries if v.shape == ()}
    if "beta" not in scalars or "tau" not in scalars:
        raise SystemExit("no scalar tensors named beta and tau in the model")
    parts = core + second + [entries[head][1].ravel(), entries[head + 1][1].ravel()]
    parts += [scalars["beta"].ravel(), scalars["tau"].ravel()]
    return MAGIC + b"".join(np.ascontiguousarray(p).tobytes() for p in parts)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    import torch  # only the conversion needs it

    model = torch.jit.load(sys.argv[1], map_location="cpu").eval()
    blob = export(model)
    with open(sys.argv[2], "wb") as f:
        f.write(blob)
    print(f"wrote {len(blob)} bytes to {sys.argv[2]}")


if __name__ == "__main__":
    main()
