#!/usr/bin/env python3
"""Inspect a LiteRT FlatBuffer for the bare-metal ResNet port."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter


SUPPORTED_OPERATORS = {
    "ADD",
    "AVERAGE_POOL_2D",
    "CONV_2D",
    "FULLY_CONNECTED",
    "MAX_POOL_2D",
    "PAD",
    "RELU",
    "RESHAPE",
    "TRANSPOSE",
}


def enum_name(enum_type: Any, value: int) -> str:
    for name in dir(enum_type):
        if name.isupper() and getattr(enum_type, name) == value:
            return name
    return str(value)


def tensor_record(detail: dict[str, Any]) -> dict[str, Any]:
    quant = detail["quantization_parameters"]
    scales = np.asarray(quant["scales"]).tolist()
    zero_points = np.asarray(quant["zero_points"]).tolist()
    shape = np.asarray(detail["shape"]).tolist()
    return {
        "index": int(detail["index"]),
        "name": detail["name"],
        "shape": shape,
        "shape_signature": np.asarray(detail["shape_signature"]).tolist(),
        "dtype": np.dtype(detail["dtype"]).name,
        "bytes": int(np.prod(shape, dtype=np.int64) * np.dtype(detail["dtype"]).itemsize),
        "quantization": {
            "scales": scales,
            "zero_points": zero_points,
            "quantized_dimension": int(quant["quantized_dimension"]),
            "block_size": int(quant.get("block_size", 0)),
        },
    }


def options_record(model: Any, operator: Any, op_name: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    option_type = operator.BuiltinOptionsType()
    option_offset = operator.BuiltinOptions()

    if option_type == schema.BuiltinOptions.Conv2DOptions:
        value = schema.Conv2DOptions()
        value.Init(option_offset.Bytes, option_offset.Pos)
        options = {
            "padding": enum_name(schema.Padding, value.Padding()),
            "stride_h": value.StrideH(),
            "stride_w": value.StrideW(),
            "dilation_h": value.DilationHFactor(),
            "dilation_w": value.DilationWFactor(),
            "fused_activation": enum_name(
                schema.ActivationFunctionType, value.FusedActivationFunction()
            ),
        }
    elif option_type == schema.BuiltinOptions.Pool2DOptions:
        value = schema.Pool2DOptions()
        value.Init(option_offset.Bytes, option_offset.Pos)
        options = {
            "padding": enum_name(schema.Padding, value.Padding()),
            "stride_h": value.StrideH(),
            "stride_w": value.StrideW(),
            "filter_h": value.FilterHeight(),
            "filter_w": value.FilterWidth(),
            "fused_activation": enum_name(
                schema.ActivationFunctionType, value.FusedActivationFunction()
            ),
        }
    elif option_type == schema.BuiltinOptions.AddOptions:
        value = schema.AddOptions()
        value.Init(option_offset.Bytes, option_offset.Pos)
        options = {
            "fused_activation": enum_name(
                schema.ActivationFunctionType, value.FusedActivationFunction()
            )
        }
    elif option_type == schema.BuiltinOptions.ReshapeOptions:
        value = schema.ReshapeOptions()
        value.Init(option_offset.Bytes, option_offset.Pos)
        options = {"new_shape": list(value.NewShape() or [])}

    return options


def inspect_model(model_path: Path) -> dict[str, Any]:
    interpreter = Interpreter(model_path=str(model_path), experimental_delegates=[])
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    by_index = {int(detail["index"]): detail for detail in details}

    model_bytes = model_path.read_bytes()
    model = schema.Model.GetRootAsModel(model_bytes, 0)
    subgraph = model.Subgraphs(0)
    operations = interpreter._get_ops_details()[: subgraph.OperatorsLength()]
    operator_codes = []
    for index in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(index)
        builtin = code.BuiltinCode()
        operator_codes.append(enum_name(schema.BuiltinOperator, builtin))

    records = []
    unsupported = set()
    for index, operation in enumerate(operations):
        op_name = operation["op_name"]
        if op_name not in SUPPORTED_OPERATORS:
            unsupported.add(op_name)
        operator = subgraph.Operators(index)
        inputs = [int(value) for value in operation["inputs"] if int(value) >= 0]
        outputs = [int(value) for value in operation["outputs"] if int(value) >= 0]
        records.append(
            {
                "index": index,
                "operator": op_name,
                "opcode": int(operator.OpcodeIndex()),
                "inputs": inputs,
                "outputs": outputs,
                "options": options_record(model, operator, op_name),
                "input_tensors": [tensor_record(by_index[value]) for value in inputs],
                "output_tensors": [tensor_record(by_index[value]) for value in outputs],
            }
        )

    tensor_records = [tensor_record(detail) for detail in details]
    weights = [
        tensor
        for tensor in tensor_records
        if tensor["dtype"] in {"int8", "int32"} and tensor["index"] not in {
            int(item["index"]) for item in interpreter.get_input_details()
        }
    ]
    operator_counts: dict[str, int] = {}
    for operation in records:
        operator_counts[operation["operator"]] = (
            operator_counts.get(operation["operator"], 0) + 1
        )

    return {
        "model": str(model_path),
        "subgraphs": int(model.SubgraphsLength()),
        "inputs": [
            tensor_record(detail) for detail in interpreter.get_input_details()
        ],
        "outputs": [
            tensor_record(detail) for detail in interpreter.get_output_details()
        ],
        "tensor_count": len(tensor_records),
        "tensors": tensor_records,
        "operator_codes": operator_codes,
        "operator_counts": operator_counts,
        "unsupported_operators": sorted(unsupported),
        "operators": records,
        "int8_or_int32_tensor_bytes": sum(item["bytes"] for item in weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    report = inspect_model(args.model)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.summary:
        lines = [
            f"Model: {report['model']}",
            f"Subgraphs: {report['subgraphs']}",
            f"Tensors: {report['tensor_count']}",
            f"Operators: {sum(report['operator_counts'].values())}",
            f"Operator counts: {json.dumps(report['operator_counts'], sort_keys=True)}",
            f"Unsupported operators: {report['unsupported_operators'] or 'none'}",
            f"INT8/INT32 tensor bytes: {report['int8_or_int32_tensor_bytes']}",
            "Inputs:",
        ]
        lines.extend(
            f"  {item['index']}: {item['name']} {item['shape']} "
            f"{item['dtype']} quant={item['quantization']}"
            for item in report["inputs"]
        )
        lines.append("Outputs:")
        lines.extend(
            f"  {item['index']}: {item['name']} {item['shape']} "
            f"{item['dtype']} quant={item['quantization']}"
            for item in report["outputs"]
        )
        args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.json and not args.summary:
        print(text)


if __name__ == "__main__":
    main()
