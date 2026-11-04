"""Write a copy of a prepared sequence with the side views perturbed.

Front frames are copied byte for byte. Side frames get a constant depth offset and
the same offset added to their landmark z, so their own pose fit may move. This is
the input for the front-only invariance check: a shared-geometry stage that still
reads side data would change its result, and one that does not must not.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-mm", type=float, default=6.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    sequence = json.loads((args.source / "sequence.json").read_text(encoding="utf-8"))
    touched = []
    for frame in sequence["frames"]:
        source = args.source / frame["file"]
        target = args.output / frame["file"]
        if frame.get("view") == "front":
            shutil.copyfile(source, target)
            continue
        with np.load(source, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        depth = payload["depth"].astype(np.int32) + int(round(args.offset_mm))
        payload["depth"] = np.clip(depth, 0, 65535).astype(payload["depth"].dtype)
        xyz = np.array(payload["xyz"], np.float32)
        xyz[:, 2] -= np.float32(args.offset_mm / 1000.)
        payload["xyz"] = xyz
        np.savez_compressed(target, **payload)
        touched.append(frame["id"])
    (args.output / "sequence.json").write_text(json.dumps(sequence, indent=1), encoding="utf-8")
    print(f"perturbed {len(touched)} side frames by {args.offset_mm:g} mm: {touched}")


if __name__ == "__main__":
    main()
