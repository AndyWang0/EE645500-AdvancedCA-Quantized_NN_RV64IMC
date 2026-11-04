#!/usr/bin/env python3
"""Export a fixed LiteRT graph and its constant tensors as C artifacts."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert import schema_py_generated as schema


TYPE_INFO = {
    schema.TensorType.INT8: ("int8_t", np.int8),
    schema.TensorType.INT32: ("int32_t", np.int32),
    schema.TensorType.INT64: ("int64_t", np.int64),
    schema.TensorType.UINT8: ("uint8_t", np.uint8),
    schema.TensorType.UINT32: ("uint32_t", np.uint32),
    schema.TensorType.BOOL: ("bool", np.bool_),
}


def c_name(value: str, prefix: str = "tensor") -> str:
    value = re.sub(r"[^a-zA-Z0-9_]", "_", value.decode() if isinstance(value, bytes) else value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value or value[0].isdigit():
        value = f"{prefix}_{value}"
    return value


def array_literal(values: np.ndarray, per_line: int = 16) -> str:
    flat = values.reshape(-1)
    rendered = []
    for start in range(0, len(flat), per_line):
        rendered.append("    " + ", ".join(str(int(x)) for x in flat[start:start + per_line]))
    return ",\n".join(rendered)


def enum_name(enum_type: Any, value: int) -> str:
    for name in dir(enum_type):
        if name.isupper() and getattr(enum_type, name) == value:
            return name
    return str(value)


def quantization(tensor: Any) -> dict[str, Any]:
    q = tensor.Quantization()
    if q is None:
        return {"scales": [], "zero_points": [], "quantized_dimension": 0}
    return {
        "scales": [float(q.Scale(i)) for i in range(q.ScaleLength())],
        "zero_points": [int(q.ZeroPoint(i)) for i in range(q.ZeroPointLength())],
        "quantized_dimension": int(q.QuantizedDimension()),
    }


def fixed_multiplier(real_value: float, max_shift: int = 63) -> tuple[int, int] | None:
    if real_value <= 0.0:
        return 0, 0
    significand, exponent = math.frexp(real_value)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent
    if not -(1 << 31) <= multiplier < (1 << 31) or not 0 <= shift <= max_shift:
        return None
    return multiplier, shift


def options_record(operator: Any) -> dict[str, Any]:
    option_type = operator.BuiltinOptionsType()
    offset = operator.BuiltinOptions()
    if option_type == schema.BuiltinOptions.Conv2DOptions:
        value = schema.Conv2DOptions()
        value.Init(offset.Bytes, offset.Pos)
        return {
            "padding": enum_name(schema.Padding, value.Padding()),
            "stride_h": value.StrideH(),
            "stride_w": value.StrideW(),
            "dilation_h": value.DilationHFactor(),
            "dilation_w": value.DilationWFactor(),
            "fused_activation": enum_name(schema.ActivationFunctionType, value.FusedActivationFunction()),
        }
    if option_type == schema.BuiltinOptions.Pool2DOptions:
        value = schema.Pool2DOptions()
        value.Init(offset.Bytes, offset.Pos)
        return {
            "padding": enum_name(schema.Padding, value.Padding()),
            "stride_h": value.StrideH(),
            "stride_w": value.StrideW(),
            "filter_h": value.FilterHeight(),
            "filter_w": value.FilterWidth(),
            "fused_activation": enum_name(schema.ActivationFunctionType, value.FusedActivationFunction()),
        }
    if option_type == schema.BuiltinOptions.AddOptions:
        value = schema.AddOptions()
        value.Init(offset.Bytes, offset.Pos)
        return {"fused_activation": enum_name(schema.ActivationFunctionType, value.FusedActivationFunction())}
    if option_type == schema.BuiltinOptions.ReshapeOptions:
        value = schema.ReshapeOptions()
        value.Init(offset.Bytes, offset.Pos)
        return {"new_shape": list(value.NewShape() or [])}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = args.model.read_bytes()
    model = schema.Model.GetRootAsModel(raw, 0)
    subgraph = model.Subgraphs(0)
    tensors = [subgraph.Tensors(i) for i in range(subgraph.TensorsLength())]
    tensor_names = [c_name(t.Name() or "", f"tensor_{i}") for i, t in enumerate(tensors)]

    tensor_records = []
    constant_records = []
    weights_records = []
    bias_records = []
    quant_records = []
    for index, tensor in enumerate(tensors):
        type_name, dtype = TYPE_INFO.get(tensor.Type(), ("uint8_t", np.uint8))
        shape = [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]
        buffer = model.Buffers(tensor.Buffer())
        data = buffer.DataAsNumpy()
        data = np.array([], dtype=np.uint8) if isinstance(data, int) else np.asarray(data, dtype=np.uint8)
        values = data.view(dtype) if data.size and dtype != np.bool_ else data.astype(dtype)
        q = quantization(tensor)
        record = {
            "index": index,
            "name": tensor.Name().decode() if tensor.Name() else "",
            "symbol": tensor_names[index],
            "type": type_name,
            "shape": shape,
            "buffer": int(tensor.Buffer()),
            "constant": bool(data.size),
            "bytes": int(data.nbytes),
            "quantization": q,
        }
        tensor_records.append(record)
        if data.size:
            constant_records.append((record, values))
            if tensor.Type() == schema.TensorType.INT8:
                weights_records.append((record, values))
            elif tensor.Type() == schema.TensorType.INT32:
                bias_records.append((record, values))
        if q["scales"]:
            quant_records.append(record)

    operators = []
    for index in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(index)
        opcode = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
        name = enum_name(schema.BuiltinOperator, opcode)
        inputs = [int(operator.Inputs(i)) for i in range(operator.InputsLength()) if operator.Inputs(i) >= 0]
        outputs = [int(operator.Outputs(i)) for i in range(operator.OutputsLength()) if operator.Outputs(i) >= 0]
        options = options_record(operator)
        fixed = {}
        if name in {"CONV_2D", "FULLY_CONNECTED"} and len(inputs) >= 3:
            input_q = quantization(tensors[inputs[0]])
            weight_q = quantization(tensors[inputs[1]])
            output_q = quantization(tensors[outputs[0]])
            if input_q["scales"] and weight_q["scales"] and output_q["scales"]:
                multipliers = []
                shifts = []
                real_multipliers = [
                    input_q["scales"][0] * scale / output_q["scales"][0]
                    for scale in weight_q["scales"]
                ]
                for real_value in real_multipliers:
                    fixed = fixed_multiplier(real_value)
                    multipliers.append(fixed[0] if fixed else 0)
                    shifts.append(fixed[1] if fixed else 0)
                fixed = {
                    "real_multipliers": real_multipliers,
                    "multipliers": multipliers,
                    "shifts": shifts,
                    "fallback_multipliers": [
                        fixed_multiplier(value, max_shift=127)[0]
                        for value in real_multipliers
                    ],
                    "fallback_shifts": [
                        fixed_multiplier(value, max_shift=127)[1]
                        for value in real_multipliers
                    ],
                    "use_scalar_fallback": [
                        fixed_multiplier(value) is None
                        for value in real_multipliers
                    ],
                    "unrepresentable_channels": [
                        index for index, value in enumerate(real_multipliers)
                        if fixed_multiplier(value) is None
                    ],
                    "representable": all(
                        fixed_multiplier(value) is not None
                        for value in real_multipliers
                    ),
                    "output_zero_point": output_q["zero_points"][0],
                }
        operators.append({
            "index": index,
            "operator": name,
            "inputs": inputs,
            "outputs": outputs,
            "options": options,
            "fixed_point": fixed,
        })

    def write_arrays(path: Path, records: list[tuple[dict[str, Any], np.ndarray]]) -> None:
        with path.open("w", encoding="utf-8") as out:
            out.write("#ifndef RESNET18_EXPORTED_ARRAYS_H\n#define RESNET18_EXPORTED_ARRAYS_H\n#include <stdint.h>\n#include <stdbool.h>\n\n")
            for record, values in records:
                out.write(f"static const {record['type']} {record['symbol']}[{values.size}] __attribute__((aligned(8))) = {{\n")
                out.write(array_literal(values))
                out.write("\n};\n\n")
            out.write("#endif\n")

    write_arrays(args.out_dir / "resnet18_weights.h", weights_records)
    write_arrays(args.out_dir / "resnet18_biases.h", bias_records)

    with (args.out_dir / "resnet18_shapes.h").open("w", encoding="utf-8") as out:
        out.write("#ifndef RESNET18_EXPORTED_SHAPES_H\n#define RESNET18_EXPORTED_SHAPES_H\n#include <stdint.h>\n\n")
        for record in tensor_records:
            shape = ", ".join(str(x) for x in record["shape"]) or "0"
            out.write(f"static const int32_t {record['symbol']}_shape[] = {{{shape}}};\n")
            out.write(f"static const uint32_t {record['symbol']}_rank = {len(record['shape'])};\n")
        out.write("\n#endif\n")

    with (args.out_dir / "resnet18_quant_params.h").open("w", encoding="utf-8") as out:
        out.write("#ifndef RESNET18_EXPORTED_QUANT_PARAMS_H\n#define RESNET18_EXPORTED_QUANT_PARAMS_H\n#include <stdint.h>\n\n")
        for record in quant_records:
            q = record["quantization"]
            out.write(f"static const float {record['symbol']}_scales[{len(q['scales'])}] = {{{', '.join(format(x, '.17g') for x in q['scales'])}}};\n")
            out.write(f"static const int32_t {record['symbol']}_zero_points[{len(q['zero_points'])}] = {{{', '.join(str(x) for x in q['zero_points'])}}};\n")
            out.write(f"static const int32_t {record['symbol']}_quantized_dimension = {q['quantized_dimension']};\n")
        for op in operators:
            fixed = op["fixed_point"]
            if fixed:
                values = ", ".join(str(x) for x in fixed["multipliers"])
                shifts = ", ".join(str(x) for x in fixed["shifts"])
                out.write(f"static const int32_t resnet18_op_{op['index']}_multipliers[{len(fixed['multipliers'])}] = {{{values}}};\n")
                out.write(f"static const uint8_t resnet18_op_{op['index']}_shifts[{len(fixed['shifts'])}] = {{{shifts}}};\n")
                fallback_multipliers = ", ".join(str(x) for x in fixed["fallback_multipliers"])
                fallback_shifts = ", ".join(str(x) for x in fixed["fallback_shifts"])
                fallback_flags = ", ".join("1" if x else "0" for x in fixed["use_scalar_fallback"])
                out.write(f"static const int32_t resnet18_op_{op['index']}_fallback_multipliers[{len(fixed['fallback_multipliers'])}] = {{{fallback_multipliers}}};\n")
                out.write(f"static const uint8_t resnet18_op_{op['index']}_fallback_shifts[{len(fixed['fallback_shifts'])}] = {{{fallback_shifts}}};\n")
                out.write(f"static const uint8_t resnet18_op_{op['index']}_use_scalar_fallback[{len(fixed['use_scalar_fallback'])}] = {{{fallback_flags}}};\n")
                out.write(f"static const int8_t resnet18_op_{op['index']}_output_zero_point = {fixed['output_zero_point']};\n")
        out.write("\n#endif\n")

    with (args.out_dir / "resnet18_graph.h").open("w", encoding="utf-8") as out:
        out.write("#ifndef RESNET18_EXPORTED_GRAPH_H\n#define RESNET18_EXPORTED_GRAPH_H\n#include <stdint.h>\n\ntypedef struct { uint8_t operator_kind; uint16_t input_count; uint16_t output_count; const int32_t *inputs; const int32_t *outputs; } resnet18_operator_t;\n")
        out.write(f"static const uint32_t resnet18_operator_count = {len(operators)};\n")
        for op in operators:
            ins = ", ".join(str(x) for x in op["inputs"]) or "0"
            outs = ", ".join(str(x) for x in op["outputs"]) or "0"
            out.write(f"static const int32_t resnet18_op_{op['index']}_inputs[] = {{{ins}}};\n")
            out.write(f"static const int32_t resnet18_op_{op['index']}_outputs[] = {{{outs}}};\n")
        out.write("\n#endif\n")

    metadata = {
        "model": str(args.model),
        "tensor_count": len(tensor_records),
        "constant_tensor_count": len(constant_records),
        "tensors": tensor_records,
        "operators": operators,
    }
    (args.out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.test_dir:
        args.test_dir.mkdir(parents=True, exist_ok=True)
        for source, target in [("input_0.bin", "input_0.bin"), ("reference_output_0.bin", "reference_output_0.bin")]:
            candidate = args.test_dir.parent / "host_validation" / source
            if candidate.exists():
                (args.test_dir / target).write_bytes(candidate.read_bytes())
    print(f"Exported {len(tensor_records)} tensors and {len(operators)} operators to {args.out_dir}")


if __name__ == "__main__":
    main()
