"""
导出轻量级 MobileNetV2 为 ONNX 格式
支持 FP32 和 INT8 量化
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from mobilenetv2 import MobileNetV2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path, input_size):
    """预处理图片为模型输入格式"""
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


def main():
    parser = argparse.ArgumentParser(description="Export lightweight MobileNetV2 to ONNX")
    parser.add_argument("--weights", default="MobileNetV2_lightweight.pth",
                        help="Path to the lightweight .pth file")
    parser.add_argument("--output", default="MobileNetV2_light.onnx",
                        help="Output FP32 ONNX path")
    parser.add_argument("--int8-output", default="MobileNetV2_light_int8.onnx",
                        help="Output INT8 ONNX path")
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Width multiplier (must match training)")
    parser.add_argument("--input-size", type=int, default=64,
                        help="Input image size (must match training)")
    parser.add_argument("--export-int8", action="store_true",
                        help="Also export INT8 quantized model")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version")
    args = parser.parse_args()

    device = torch.device("cpu")
    weight_path = Path(args.weights)

    if not weight_path.exists():
        print(f"Error: Weight file not found: {weight_path}")
        print("Please run train_lightweight.py first to train the model.")
        return

    # 加载模型
    print(f"Loading model with alpha={args.alpha}, input_size={args.input_size}")
    model = MobileNetV2(num_classes=5, alpha=args.alpha)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 统计模型大小
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = weight_path.stat().st_size / (1024 * 1024)
    print(f"Model params: {total_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    # 导出 FP32 ONNX
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    output_path = Path(args.output)

    print(f"\nExporting FP32 ONNX to {output_path}")
    with torch.no_grad():
        reference_output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # 验证 ONNX
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(output_path.as_posix())
    onnx.checker.check_model(onnx_model)
    print(f"FP32 ONNX exported successfully!")

    # 验证推理一致性
    session = ort.InferenceSession(output_path.as_posix(), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": dummy_input.numpy()})[0]
    max_diff = float(np.max(np.abs(reference_output.numpy() - ort_output)))
    print(f"Max diff vs PyTorch: {max_diff:.8f}")

    onnx_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX size: {onnx_size_mb:.2f} MB")

    # 导出 INT8 量化模型
    if args.export_int8:
        print(f"\nExporting INT8 ONNX...")
        try:
            from export_onnx import quantize_onnx_model, find_calibration_images

            # 查找校准图片
            calib_dir = Path("data_set/fruit_data/train")
            if calib_dir.exists():
                calib_images = find_calibration_images(calib_dir, limit=50)
                print(f"Using {len(calib_images)} calibration images")

                quant_info = quantize_onnx_model(
                    fp32_path=output_path,
                    int8_path=Path(args.int8_output),
                    quant_method="static",
                    input_size=args.input_size,
                    calib_dir=calib_dir,
                    calib_count=50,
                    per_channel=False,
                )

                int8_path = Path(args.int8_output)
                int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
                print(f"\nINT8 ONNX exported: {int8_path}")
                print(f"INT8 ONNX size: {int8_size_mb:.2f} MB")
                print(f"Size reduction: {onnx_size_mb:.2f} MB -> {int8_size_mb:.2f} MB "
                      f"({100*(1-int8_size_mb/onnx_size_mb):.1f}% smaller)")
            else:
                print(f"Calibration directory not found: {calib_dir}")
                print("Skipping INT8 quantization")

        except Exception as e:
            print(f"INT8 export failed: {e}")
            print("You can manually quantize later using export_onnx.py")

    # 推理速度测试
    print("\nBenchmarking inference speed...")
    import time

    # Warmup
    for _ in range(10):
        session.run(None, {"input": dummy_input.numpy()})

    # 测试
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {"input": dummy_input.numpy()})
    elapsed = time.time() - start

    avg_ms = (elapsed / num_runs) * 1000
    print(f"Average inference time: {avg_ms:.1f} ms (FP32, {num_runs} runs)")

    print("\n" + "="*50)
    print("Summary:")
    print(f"  Input size: {args.input_size}x{args.input_size}")
    print(f"  Alpha: {args.alpha}")
    print(f"  FP32 ONNX: {onnx_size_mb:.2f} MB, ~{avg_ms:.0f} ms")
    if args.export_int8 and Path(args.int8_output).exists():
        int8_size_mb = Path(args.int8_output).stat().st_size / (1024 * 1024)
        print(f"  INT8 ONNX: {int8_size_mb:.2f} MB, ~{avg_ms*0.3:.0f} ms (estimated)")
    print("="*50)


if __name__ == "__main__":
    main()
