import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from onnxruntime.quantization import (
    CalibrationMethod,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quantize import (
    IntegerOpsRegistry,
    ONNXQuantizer,
    QDQRegistry,
    QLinearOpsRegistry,
    QuantizationMode,
    model_has_pre_process_metadata,
    update_opset_version,
)
from onnxruntime.quantization.quant_utils import add_infer_metadata, add_pre_process_metadata

from mobilenetv2 import MobileNetV2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_num_classes(state_dict):
    weight = state_dict.get("classifier.1.weight")
    bias = state_dict.get("classifier.1.bias")
    if weight is not None:
        return int(weight.shape[0])
    if bias is not None:
        return int(bias.shape[0])
    raise KeyError("Cannot infer num_classes from classifier weights.")


def build_model(weight_path, device):
    state_dict = torch.load(weight_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{weight_path} is not a state_dict file.")

    num_classes = infer_num_classes(state_dict)
    model = MobileNetV2(num_classes=num_classes).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"load_state_dict mismatch, missing={missing_keys}, unexpected={unexpected_keys}"
        )

    model.eval()
    return model, num_classes


def export_to_onnx(model, output_path, input_size, opset):
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    with torch.no_grad():
        reference_output = model(dummy_input).cpu()

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        keep_initializers_as_inputs=False,
        external_data=False,
    )

    return reference_output, dummy_input


def verify_embedded_weights(output_path):
    model = onnx.load(output_path.as_posix(), load_external_data=False)
    onnx.checker.check_model(model)

    for initializer in model.graph.initializer:
        if initializer.external_data:
            raise RuntimeError(
                "The ONNX file references external tensor data; expected a single-file model."
            )

    return len(model.graph.initializer)


def verify_runtime(output_path, dummy_input, reference_output):
    session = ort.InferenceSession(
        output_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(
        None,
        {"input": dummy_input.cpu().numpy().astype(np.float32)},
    )[0]

    max_abs_diff = float(np.max(np.abs(reference_output.numpy() - ort_output)))
    return max_abs_diff


def verify_onnx_inference(output_path, sample_input):
    session = ort.InferenceSession(
        output_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    output = session.run(None, {"input": sample_input.astype(np.float32)})[0]
    return tuple(output.shape)


def find_calibration_images(calib_dir, limit):
    image_paths = sorted(
        path
        for path in calib_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found under: {calib_dir}")
    return image_paths[:limit] if limit is not None else image_paths


def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    resize_size = int(round(input_size / 224 * 256))
    image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)

    left = max((resize_size - input_size) // 2, 0)
    top = max((resize_size - input_size) // 2, 0)
    right = left + input_size
    bottom = top + input_size
    image = image.crop((left, top, right, bottom))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)
    return array.astype(np.float32)


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_size, input_name="input"):
        self.input_name = input_name
        self.samples = [
            {self.input_name: preprocess_image(path, input_size)} for path in image_paths
        ]
        self.iterator = iter(self.samples)

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.samples)


def quantize_onnx_model(
    fp32_path,
    int8_path,
    quant_method,
    input_size,
    calib_dir=None,
    calib_count=100,
    per_channel=False,
):
    quant_method = quant_method.lower()
    temp_root = fp32_path.parent / "ort_quant_work"
    temp_root.mkdir(parents=True, exist_ok=True)

    old_tempdir = tempfile.tempdir
    old_tmp = os.environ.get("TMP")
    old_temp = os.environ.get("TEMP")
    tempfile.tempdir = temp_root.as_posix()
    os.environ["TMP"] = temp_root.as_posix()
    os.environ["TEMP"] = temp_root.as_posix()

    try:
        if quant_method == "dynamic":
            import onnxruntime.quantization.base_quantizer as base_quantizer
            import onnxruntime.quantization.onnx_quantizer as ort_onnx_quantizer

            model = onnx.load(fp32_path.as_posix(), load_external_data=False)
            model = onnx.shape_inference.infer_shapes(model)
            model = update_opset_version(model, QuantType.QInt8)

            pre_processed = model_has_pre_process_metadata(model)
            if not pre_processed:
                print(
                    "Warning: ONNX model does not contain preprocessing metadata. Quantization still proceeds."
                )

            old_shape_infer_fn = base_quantizer.save_and_reload_model_with_shape_infer
            old_has_infer_fn = base_quantizer.model_has_infer_metadata
            old_onnx_quantizer_shape_infer_fn = (
                ort_onnx_quantizer.save_and_reload_model_with_shape_infer
            )
            base_quantizer.save_and_reload_model_with_shape_infer = lambda onnx_model: onnx_model
            base_quantizer.model_has_infer_metadata = lambda onnx_model: True
            ort_onnx_quantizer.save_and_reload_model_with_shape_infer = (
                lambda onnx_model: onnx_model
            )

            try:
                quantizer = ONNXQuantizer(
                    model,
                    per_channel,
                    False,
                    QuantizationMode.IntegerOps,
                    False,
                    QuantType.QInt8,
                    QuantType.QUInt8,
                    None,
                    [],
                    [],
                    list(IntegerOpsRegistry.keys()),
                    {
                        "MatMulConstBOnly": True,
                        "DefaultTensorType": onnx.TensorProto.FLOAT,
                    },
                )
            finally:
                base_quantizer.save_and_reload_model_with_shape_infer = old_shape_infer_fn
                base_quantizer.model_has_infer_metadata = old_has_infer_fn
                ort_onnx_quantizer.save_and_reload_model_with_shape_infer = (
                    old_onnx_quantizer_shape_infer_fn
                )

            quantizer.quantize_model()
            quantizer.model.save_model_to_file(
                int8_path.as_posix(), use_external_data_format=False
            )
            return {"quant_method": "dynamic", "calibration_images": 0}

        if quant_method != "static":
            raise ValueError(f"Unsupported quantization method: {quant_method}")

        if calib_dir is None:
            raise ValueError(
                "Static quantization requires --calib-dir to provide calibration images."
            )

        image_paths = find_calibration_images(calib_dir, calib_count)
        data_reader = ImageCalibrationDataReader(image_paths, input_size=input_size)
        calibration_work_dir = temp_root / "static_calibration"
        if calibration_work_dir.exists():
            shutil.rmtree(calibration_work_dir, ignore_errors=True)
        calibration_work_dir.mkdir(parents=True, exist_ok=True)

        op_types_to_quantize = ["Conv", "Add", "GlobalAveragePool", "Gemm"]
        calibrator = create_calibrator(
            fp32_path,
            op_types_to_calibrate=op_types_to_quantize,
            augmented_model_path=(calibration_work_dir / "augmented_model.onnx").as_posix(),
            calibrate_method=CalibrationMethod.MinMax,
            use_external_data_format=False,
            providers=["CPUExecutionProvider"],
            extra_options={},
        )
        calibrator.collect_data(data_reader)
        tensors_range = calibrator.compute_data()

        model = onnx.load(fp32_path.as_posix(), load_external_data=False)
        model = onnx.shape_inference.infer_shapes(model)
        add_infer_metadata(model)
        add_pre_process_metadata(model)
        model = update_opset_version(model, QuantType.QInt8)

        quantizer = QDQQuantizer(
            model,
            per_channel,
            False,
            QuantType.QInt8,
            QuantType.QUInt8,
            tensors_range,
            [],
            [],
            op_types_to_quantize,
            {
                "QuantizeBias": True,
                "AddQDQPairToWeight": False,
                "DedicatedQDQPair": False,
            },
        )
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(
            int8_path.as_posix(), use_external_data_format=False
        )
        return {"quant_method": "static", "calibration_images": len(image_paths)}
    finally:
        tempfile.tempdir = old_tempdir
        if old_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = old_tmp

        if old_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = old_temp

        shutil.rmtree(temp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export a MobileNetV2 .pth state_dict to FP32 ONNX and optional INT8 quantized ONNX."
    )
    parser.add_argument(
        "--weights",
        default="MobileNetV2.pth",
        help="Path to the input .pth state_dict file.",
    )
    parser.add_argument(
        "--output",
        default="MobileNetV2.onnx",
        help="Path to the output FP32 .onnx file.",
    )
    parser.add_argument(
        "--int8-output",
        default="MobileNetV2_int8.onnx",
        help="Path to the output INT8 .onnx file.",
    )
    parser.add_argument(
        "--export-int8",
        action="store_true",
        help="Export an INT8 quantized ONNX model after FP32 export.",
    )
    parser.add_argument(
        "--quant-method",
        choices=["dynamic", "static"],
        default="dynamic",
        help="INT8 quantization method. Default: dynamic.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Directory containing calibration images for static quantization.",
    )
    parser.add_argument(
        "--calib-count",
        type=int,
        default=100,
        help="Maximum number of calibration images used for static quantization.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization. For STM32Cube.AI, keeping this disabled is usually safer.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
        help="Square input size used for export. Default: 96.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version. Default: 13.",
    )
    args = parser.parse_args()

    weight_path = Path(args.weights)
    output_path = Path(args.output)
    int8_output_path = Path(args.int8_output)
    device = torch.device("cpu")

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    if args.quant_method == "static" and not args.export_int8:
        raise ValueError("--quant-method static requires --export-int8.")

    if args.quant_method == "static" and args.calib_dir is None:
        raise ValueError("Static quantization requires --calib-dir.")

    model, num_classes = build_model(weight_path, device)
    reference_output, dummy_input = export_to_onnx(
        model=model,
        output_path=output_path,
        input_size=args.input_size,
        opset=args.opset,
    )
    initializer_count = verify_embedded_weights(output_path)
    max_abs_diff = verify_runtime(output_path, dummy_input, reference_output)

    print(f"FP32 export success: {output_path.resolve()}")
    print(f"num_classes={num_classes}")
    print(f"input_shape={tuple(dummy_input.shape)}")
    print(f"output_shape={tuple(reference_output.shape)}")
    print(f"initializer_count={initializer_count}")
    print(f"max_abs_diff_vs_onnxruntime={max_abs_diff:.8f}")

    if args.export_int8:
        quant_info = quantize_onnx_model(
            fp32_path=output_path,
            int8_path=int8_output_path,
            quant_method=args.quant_method,
            input_size=args.input_size,
            calib_dir=args.calib_dir,
            calib_count=args.calib_count,
            per_channel=args.per_channel,
        )
        int8_initializer_count = verify_embedded_weights(int8_output_path)
        int8_output_shape = verify_onnx_inference(
            int8_output_path, dummy_input.cpu().numpy().astype(np.float32)
        )

        print(f"INT8 export success: {int8_output_path.resolve()}")
        print(f"quant_method={quant_info['quant_method']}")
        print(f"calibration_images={quant_info['calibration_images']}")
        print(f"int8_initializer_count={int8_initializer_count}")
        print(f"int8_output_shape={int8_output_shape}")

    print("The ONNX file contains both graph structure and embedded weights in one file.")


if __name__ == "__main__":
    main()
