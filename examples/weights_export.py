"""Export and re-import a nested FugaTensor weight PyTree.

Run: .venv/bin/python examples/weights_export.py --output weights-demo
"""
from __future__ import annotations

import argparse
import json

import fugatensor as ft


def demo_tree():
    return {
        "encoder": {
            "weight": ft.tensor([[0.5, -1.0], [2.0, 0.25]]),
            "bias": ft.tensor([0.1, -0.2]),
        },
        "blocks": [
            {"weight": ft.tensor([1.5, -2.0, 3.25], dtype=ft.fp16)},
            {"weight": ft.tensor([0.125], dtype=ft.fp16)},
        ],
        "step": ft.tensor(1200, dtype=ft.int32),
        "empty": ft.tensor([], shape=(0, 4)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="weights-demo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tree = demo_tree()
    target = ft.weights.save(args.output, tree, metadata={"run": "weights-demo"},
                             overwrite=args.overwrite)
    manifest = ft.weights.read_manifest(target)
    loaded = ft.weights.load(target)
    checks = {
        "encoder.weight": loaded["encoder"]["weight"].tolist(),
        "blocks": [item["weight"].tolist() for item in loaded["blocks"]],
        "step": loaded["step"].tolist(),
    }
    print(json.dumps({
        "output": str(target),
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "arrays": len(manifest["arrays"]),
        "metadata": manifest["metadata"],
        "roundtrip": checks,
        "empty_shape": loaded["empty"].shape,
    }, indent=2))


if __name__ == "__main__":
    main()
