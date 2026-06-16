from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

from landmarklab.core import append_optimization_log, compute_metrics, ensure_dir, load_config, write_json
from landmarklab.data import create_dataloaders
from landmarklab.model import build_model


class LandmarkCalibrationReader(CalibrationDataReader):
    def __init__(self, loader: torch.utils.data.DataLoader, max_batches: int) -> None:
        self.batches: list[dict[str, np.ndarray]] = []
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            images = batch["image"].numpy().astype(np.float32)
            for image in images:
                self.batches.append({"images": image[None, ...]})
        self.iterator = iter(self.batches)


    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.iterator, None)


def run_onnx(
    session: ort.InferenceSession,
    loader: torch.utils.data.DataLoader,
    eye_groups: list[list[int]],
) -> dict[str, float]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].numpy().astype(np.float32)
        for index, image in enumerate(images):
            output = session.run(None, {"images": image[None, ...]})[0]
            predictions.append(torch.from_numpy(output))
            targets.append(batch["landmarks"][index : index + 1])
    return compute_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        eye_groups=eye_groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an INT8 ONNX face landmark model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--note", default="ptq_static")
    parser.add_argument("--quant-mode", default="full", choices=["full", "conv_only"],
                        help="full: quantize all ops (default); conv_only: only Conv ops, keep decode chain FP32")
    parser.add_argument("--calibrate-method", default="MinMax", choices=["MinMax", "Percentile", "Entropy"],
                        help="Calibration method for activation ranges")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = ensure_dir(args.run)
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_size = int(checkpoint["config"]["data"]["image_size"])
    fp32_onnx = run_dir / "model_fp32.onnx"
    int8_onnx = run_dir / "model_int8.onnx"
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        fp32_onnx,
        input_names=["images"],
        output_names=["landmarks"],
        opset_version=18,
        do_constant_folding=True,
    )

    quant_config = checkpoint["config"].copy()
    quant_config["data"] = dict(quant_config["data"])
    quant_config["data"]["download"] = False
    dataloaders = create_dataloaders(quant_config)
    calibration_reader = LandmarkCalibrationReader(dataloaders.valid, max_batches=args.calibration_batches)
    cal_method_map = {
        "MinMax": CalibrationMethod.MinMax,
        "Percentile": CalibrationMethod.Percentile,
        "Entropy": CalibrationMethod.Entropy,
    }
    calibrate_method = cal_method_map[args.calibrate_method]

    nodes_to_quantize = None
    nodes_to_exclude = None
    if args.quant_mode == "conv_only":
        # Only quantize Conv ops, keep Softmax + arithmetic decode chain in FP32
        onnx_model = onnx.load(str(fp32_onnx))
        conv_node_names = [n.name for n in onnx_model.graph.node if n.op_type == "Conv"]
        nodes_to_quantize = conv_node_names
        print(f"quant_mode=conv_only: quantizing {len(conv_node_names)} Conv nodes, keeping decode chain FP32")

    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=calibrate_method,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
    )

    session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    metrics = run_onnx(session, dataloaders.test, quant_config["data"]["metric_eye_groups"])
    summary = {
        "run_name": run_dir.name,
        "checkpoint": str(checkpoint_path),
        "quant_model_size_mb": int8_onnx.stat().st_size / (1024 * 1024),
        "fp32_model_size_mb": fp32_onnx.stat().st_size / (1024 * 1024),
        **metrics,
    }
    write_json(run_dir / "quant_summary.json", summary)
    append_optimization_log(Path("optimization_log.md"), stage="quant", run_name=run_dir.name, note=args.note, summary=summary)
    print(summary)


if __name__ == "__main__":
    main()