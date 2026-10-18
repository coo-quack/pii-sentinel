"""Average the weights of several checkpoints (a model soup) into one checkpoint.

python -m pii_sentinel.average models/run/s0 models/run/s1 [more] --out models/run/avg

The checkpoints must share the span labels, the document categories and the base revision. The averaged
checkpoint takes the tokenizer and settings of the first one and should be evaluated like any other before it is
adopted.
"""

import argparse
from pathlib import Path

from . import model as M


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint directories (at least two)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if len(a.checkpoints) < 2:
        ap.error("averaging needs at least two checkpoints")
    if a.out.exists() and any(a.out.iterdir()):
        ap.error(f"{a.out} is not empty")

    loaded = [M.load(path, "cpu") for path in a.checkpoints]
    model, tok, ref = loaded[0]
    for path, (other, _, meta) in zip(a.checkpoints, loaded, strict=True):
        if other.bio != model.bio:
            raise RuntimeError(f"{path} has different span labels from {a.checkpoints[0]}")
        if meta["categories"] != ref["categories"]:
            raise RuntimeError(f"{path} has different document categories from {a.checkpoints[0]}")
        if meta["base_revision"] != ref["base_revision"]:
            raise RuntimeError(f"{path} has a different base revision from {a.checkpoints[0]}")
    states = [other.state_dict() for other, _, _ in loaded]
    if any(set(s) != set(states[0]) for s in states):
        raise RuntimeError("the checkpoints do not have the same weights")

    # Uniform average in float32; tensors that are not floating point (none so far) come from the first checkpoint.
    avg = {
        k: sum(s[k].float() for s in states) / len(states) if v.is_floating_point() else v.clone()
        for k, v in states[0].items()
    }
    model.load_state_dict(avg)
    model.sync_head()
    meta = {
        **ref,
        "format": M.FORMAT,
        "step": 0,
        "averaged_from": [str(p) for p in a.checkpoints],
        "valid": None,
    }
    M.save(model, a.out, meta, tok, ref["base_model"], ref["base_revision"])
    print(f"averaged {len(loaded)} checkpoints into {a.out}", flush=True)


if __name__ == "__main__":
    main()
