#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def build_sample_model(m=8, n=16, k=32):
    weights = []
    for r in range(k):
        for c in range(n):
            weights.append(((r * 17 + c * 13) % 29) - 14)

    bias = [((j * 11) % 31) - 15 for j in range(n)]
    input_data = []
    for i in range(m):
        for r in range(k):
            input_data.append(((i * 19 + r * 7) % 29) - 14)

    multiplier = []
    shift = []
    zero_point = []
    for j in range(n):
        multiplier.append((j % 5) + 1)
        shift.append((j % 4) + 1)
        zero_point.append((j % 17) - 8)

    return {
        "name": "demo_qgemm_int8",
        "description": "Synthetic quantized GEMM used for the RV64IMC baseline.",
        "shape": {"M": m, "N": n, "K": k},
        "input_zero_point": 0,
        "weights": weights,
        "bias": bias,
        "input": input_data,
        "output_multiplier": multiplier,
        "output_shift": shift,
        "output_zero_point": zero_point,
    }


def requantize_reference(accumulator, multiplier, shift, zero_point):
    product = accumulator * multiplier
    if shift == 0:
        scaled = product
    else:
        magnitude = abs(product)
        rounded = (magnitude + (1 << (shift - 1))) >> shift
        scaled = -rounded if product < 0 else rounded

    value = scaled + zero_point
    if value > 127:
        return 127
    if value < -128:
        return -128
    return int(value)


def export_model(model_path, out_dir, test_dir):
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    m = model["shape"]["M"]
    n = model["shape"]["N"]
    k = model["shape"]["K"]

    weights = model["weights"]
    input_data = model["input"]
    bias = model["bias"]
    multiplier = model["output_multiplier"]
    shift = model["output_shift"]
    zero_point = model["output_zero_point"]

    if len(weights) != k * n:
        raise ValueError(f"weight length mismatch: expected {k * n}, got {len(weights)}")
    if len(input_data) != m * k:
        raise ValueError(f"input length mismatch: expected {m * k}, got {len(input_data)}")
    if len(bias) != n:
        raise ValueError(f"bias length mismatch: expected {n}, got {len(bias)}")
    if len(multiplier) != n or len(shift) != n or len(zero_point) != n:
        raise ValueError("quantization parameter lengths do not match N")

    out_dir = Path(out_dir)
    test_dir = Path(test_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    ref_output = []
    for i in range(m):
        for j in range(n):
            acc = bias[j]
            for r in range(k):
                acc += int(input_data[i * k + r]) * int(weights[r * n + j])
            ref_output.append(requantize_reference(acc, multiplier[j], shift[j], zero_point[j]))

    model_meta = {
        "name": model["name"],
        "description": model["description"],
        "input_zero_point": model["input_zero_point"],
        "weights_shape": [k, n],
        "bias_shape": [n],
        "output_shift": shift,
        "output_multiplier": multiplier,
        "output_zero_point": zero_point,
    }

    out_dir.joinpath("weights.h").write_text(
        "#ifndef MODEL_WEIGHTS_H\n"
        "#define MODEL_WEIGHTS_H\n\n"
        "#include <stdint.h>\n\n"
        f"#define MODEL_M {m}\n"
        f"#define MODEL_N {n}\n"
        f"#define MODEL_K {k}\n\n"
        "static const int8_t model_weights[MODEL_K * MODEL_N] = {\n"
        + ", ".join(str(v) for v in weights)
        + "\n};\n\n"
        "static const int64_t model_bias[MODEL_N] = {\n"
        + ", ".join(str(v) for v in bias)
        + "\n};\n\n"
        "#endif\n",
        encoding="utf-8",
    )

    out_dir.joinpath("quant_params.h").write_text(
        "#ifndef MODEL_QUANT_PARAMS_H\n"
        "#define MODEL_QUANT_PARAMS_H\n\n"
        "#include <stdint.h>\n\n"
        f"static const int32_t model_multiplier[MODEL_N] = {{ {', '.join(str(v) for v in multiplier)} }};\n"
        f"static const uint8_t model_shift[MODEL_N] = {{ {', '.join(str(v) for v in shift)} }};\n"
        f"static const int8_t model_zero_point[MODEL_N] = {{ {', '.join(str(v) for v in zero_point)} }};\n\n"
        "#endif\n",
        encoding="utf-8",
    )

    out_dir.joinpath("model_metadata.h").write_text(
        "#ifndef MODEL_METADATA_H\n"
        "#define MODEL_METADATA_H\n\n"
        "#include <stdint.h>\n\n"
        f"static const int32_t model_input_zero_point = {model['input_zero_point']};\n"
        "#endif\n",
        encoding="utf-8",
    )

    with open(test_dir / "input.bin", "wb") as f:
        for value in input_data:
            f.write(int(value).to_bytes(1, byteorder="little", signed=True))

    with open(test_dir / "reference_output.bin", "wb") as f:
        for value in ref_output:
            f.write(int(value).to_bytes(1, byteorder="little", signed=True))

    with open(test_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    print(f"Exported model metadata, weights, quantization params, and test vectors to {out_dir} and {test_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export a synthetic INT8 quantized GEMM model for the RV64IMC kernel workflow."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the input model JSON file.")
    parser.add_argument("--out-dir", type=str, default="model/exported", help="Directory for exported C headers.")
    parser.add_argument("--test-dir", type=str, default="model/test_vectors", help="Directory for exported test vectors.")
    args = parser.parse_args()

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(build_sample_model(), f, indent=2)
        print(f"Created a synthetic model at {model_path}")

    export_model(str(model_path), args.out_dir, args.test_dir)


if __name__ == "__main__":
    main()
