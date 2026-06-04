import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_indices(json_path: Path) -> dict[int, str]:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def preprocess_image(image_path: Path, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    resize_size = int(round(input_size / 224 * 256))
    image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)

    left = max((resize_size - input_size) // 2, 0)
    top = max((resize_size - input_size) // 2, 0)
    right = left + input_size
    bottom = top + input_size
    image = image.crop((left, top, right, bottom))

    image_nhwc_uint8 = np.asarray(image, dtype=np.uint8)
    image_nchw = image_nhwc_uint8.astype(np.float32) / 255.0
    image_nchw = (image_nchw - MEAN) / STD
    image_nchw = np.transpose(image_nchw, (2, 0, 1)).astype(np.float32)
    return image_nchw, image_nhwc_uint8


def build_dataset(dataset_dir: Path, class_to_idx: dict[str, int], input_size: int):
    x_test = []
    x_test_nhwc_uint8 = []
    y_test = []
    file_paths = []

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        class_name = class_dir.name
        label = class_to_idx[class_name]

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            image_nchw, image_nhwc_uint8 = preprocess_image(image_path, input_size)
            x_test.append(image_nchw)
            x_test_nhwc_uint8.append(image_nhwc_uint8)
            y_test.append(label)
            file_paths.append(str(image_path.relative_to(dataset_dir.parent.parent)))

    if not x_test:
        raise FileNotFoundError(f"No images found under: {dataset_dir}")

    return (
        np.stack(x_test, axis=0),
        np.asarray(y_test, dtype=np.int64),
        np.stack(x_test_nhwc_uint8, axis=0),
        np.asarray(file_paths),
    )


def main():
    parser = argparse.ArgumentParser(description="Build a fruit validation .npz for ONNX testing.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data_set/fruit_data/val"),
        help="Validation dataset directory in ImageFolder layout.",
    )
    parser.add_argument(
        "--class-indices",
        type=Path,
        default=Path("class_indices.json"),
        help="Path to class_indices.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MobileNetV2_from_pth_val_96.npz"),
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
        help="Square input size used by the ONNX model.",
    )
    args = parser.parse_args()

    idx_to_class = load_class_indices(args.class_indices)
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}

    x_test, y_test, x_test_nhwc_uint8, file_paths = build_dataset(
        dataset_dir=args.dataset_dir,
        class_to_idx=class_to_idx,
        input_size=args.input_size,
    )

    class_names = np.asarray([idx_to_class[i] for i in sorted(idx_to_class)], dtype="<U32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        x_test=x_test,
        y_test=y_test,
        x_test_nhwc_uint8=x_test_nhwc_uint8,
        labels=y_test,
        class_names=class_names,
        file_paths=file_paths,
        input_size=np.asarray(args.input_size, dtype=np.int64),
        input_format=np.asarray("NCHW float32 normalized", dtype="<U32"),
    )

    print(f"Saved: {args.output.resolve()}")
    print(f"x_test shape: {x_test.shape}, dtype: {x_test.dtype}")
    print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
    print(f"x_test_nhwc_uint8 shape: {x_test_nhwc_uint8.shape}, dtype: {x_test_nhwc_uint8.dtype}")


if __name__ == "__main__":
    main()
