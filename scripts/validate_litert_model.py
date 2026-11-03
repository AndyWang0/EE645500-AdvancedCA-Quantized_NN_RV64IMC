#!/usr/bin/env python3
"""Run deterministic host validation for a quantized LiteRT model."""

import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def tensor_stats(value: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.name,
        "bytes": int(value.nbytes),
        "min": int(value.min()) if value.size else None,
        "max": int(value.max()) if value.size else None,
        "sum": int(value.astype(np.int64).sum()) if value.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    interpreter = Interpreter(
        model_path=str(args.model),
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"expected one input and one output, got {len(inputs)} and {len(outputs)}"
        )

    input_detail = inputs[0]
    if input_detail["dtype"] is not np.int8:
        raise TypeError(f"expected int8 input, got {input_detail['dtype']}")
    input_shape = tuple(int(value) for value in input_detail["shape"])
    rng = np.random.default_rng(args.seed)
    input_value = rng.integers(-128, 128, size=input_shape, dtype=np.int8)

    interpreter.set_tensor(int(input_detail["index"]), input_value)
    interpreter.invoke()
    output_detail = outputs[0]
    output_value = interpreter.get_tensor(int(output_detail["index"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.out_dir / "input_0.bin"
    output_path = args.out_dir / "reference_output_0.bin"
    input_value.tofile(input_path)
    output_value.tofile(output_path)

    intermediate_dir = args.out_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    tensor_records = []
    available_tensor_count = 0
    for detail in interpreter.get_tensor_details():
        index = int(detail["index"])
        try:
            value = interpreter.get_tensor(index)
        except ValueError as error:
            tensor_records.append(
                {
                    "index": index,
                    "name": detail["name"],
                    "available": False,
                    "reason": str(error),
                }
            )
            continue
        path = intermediate_dir / f"tensor_{index}.bin"
        value.tofile(path)
        available_tensor_count += 1
        tensor_records.append(
            {
                "index": index,
                "name": detail["name"],
                "available": True,
                "file": str(path.relative_to(args.out_dir)),
                "stats": tensor_stats(value),
            }
        )

    output_flat = output_value.reshape(-1)
    predicted_class = int(np.argmax(output_flat))
    metadata = {
        "model": str(args.model),
        "seed": args.seed,
        "input": {
            "index": int(input_detail["index"]),
            "name": input_detail["name"],
            "shape": list(input_shape),
            "dtype": np.dtype(input_detail["dtype"]).name,
            "quantization": {
                "scale": float(input_detail["quantization"][0]),
                "zero_point": int(input_detail["quantization"][1]),
            },
            "file": str(input_path.relative_to(args.out_dir)),
            "stats": tensor_stats(input_value),
        },
        "output": {
            "index": int(output_detail["index"]),
            "name": output_detail["name"],
            "shape": list(output_value.shape),
            "dtype": output_value.dtype.name,
            "quantization": {
                "scale": float(output_detail["quantization"][0]),
                "zero_point": int(output_detail["quantization"][1]),
            },
            "file": str(output_path.relative_to(args.out_dir)),
            "stats": tensor_stats(output_value),
            "predicted_class": predicted_class,
        },
        "intermediate_tensor_count": len(tensor_records),
        "intermediates": tensor_records,
    }
    (args.out_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Host validation passed: {output_path}")
    print(f"Predicted class index: {predicted_class}")
    print(
        f"Recorded {len(tensor_records)} tensor records "
        f"({available_tensor_count} snapshots available)"
    )


if __name__ == "__main__":
    main()
