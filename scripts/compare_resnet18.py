#!/usr/bin/env python3
"""Compare ported ResNet-18 tensor dumps with LiteRT reference dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_metadata(directory: Path) -> dict:
    path = directory / "validation_metadata.json"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Generate the ported tensor dump first; "
            "the comparison command does not create it."
        )
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def tensor_file(directory: Path, tensor: dict) -> Path:
    filename = tensor.get("file", f"tensor_{tensor['index']}.bin")
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return directory / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare complete INT8 tensor dumps from host and ported model"
    )
    parser.add_argument("--host-dir", type=Path, required=True)
    parser.add_argument("--ported-dir", type=Path, required=True)
    args = parser.parse_args()

    host_metadata = load_metadata(args.host_dir)
    ported_metadata = load_metadata(args.ported_dir)
    host_tensors = {
        int(tensor["index"]): tensor for tensor in host_metadata["intermediates"]
    }
    ported_tensors = {
        int(tensor["index"]): tensor for tensor in ported_metadata["intermediates"]
    }

    failures = 0
    compared = 0
    first_failure: tuple[int, int] | None = None

    for tensor_id in sorted(host_tensors):
        host_record = host_tensors[tensor_id]
        if not host_record.get("available", False):
            continue
        if tensor_id not in ported_tensors:
            print(f"FAIL tensor {tensor_id}: missing from ported metadata")
            failures += 1
            continue

        ported_record = ported_tensors[tensor_id]
        if not ported_record.get("available", False):
            continue
        host_path = tensor_file(args.host_dir, host_record)
        ported_path = tensor_file(args.ported_dir, ported_record)
        if not host_path.exists() or not ported_path.exists():
            missing = host_path if not host_path.exists() else ported_path
            print(f"FAIL tensor {tensor_id}: missing file {missing}")
            failures += 1
            continue

        expected = np.fromfile(host_path, dtype=np.int8)
        actual = np.fromfile(ported_path, dtype=np.int8)
        compared += 1
        if expected.shape != actual.shape:
            print(
                f"FAIL tensor {tensor_id}: shape {actual.shape} != {expected.shape}"
            )
            failures += 1
            continue

        differences = np.flatnonzero(expected != actual)
        if differences.size:
            index = int(differences[0])
            error = int(
                np.max(
                    np.abs(
                        expected.astype(np.int16) - actual.astype(np.int16)
                    )
                )
            )
            print(
                f"FAIL tensor {tensor_id}: mismatches={differences.size} "
                f"max_abs_error={error} first_index={index} "
                f"expected={int(expected[index])} actual={int(actual[index])}"
            )
            failures += 1
            if first_failure is None:
                first_failure = (tensor_id, index)

    print(
        f"Compared {compared} tensors; "
        f"mismatching tensors: {failures}"
    )
    if first_failure is not None:
        print(
            f"First mismatch: tensor {first_failure[0]}, "
            f"index {first_failure[1]}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
