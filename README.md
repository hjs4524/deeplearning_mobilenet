📁 项目结构说明
✨ 功能特性介绍
🚀 快速开始指南（环境依赖、数据准备、训练、预测、ONNX 导出）
📊 标准版 vs 轻量级版参数对比表
🔧 量化方式说明
🍎 支持的水果类别列表
# MobileNetV2 水果分类系统

基于 PyTorch 的 MobileNetV2 水果图像分类项目，支持标准版和轻量级版本训练、ONNX 导出及 INT8 量化，可部署到移动端和嵌入式设备。

## 项目结构

```
├── mobilenetv2.py              # MobileNetV2 模型定义
├── train.py                    # 标准版训练 (224x224, alpha=1.0)
├── train_lightweight.py        # 轻量级训练 (64x64, alpha=0.75)
├── predict.py                  # 图像预测
├── export_onnx.py              # ONNX 导出 + INT8 量化
├── export_lightweight.py       # 轻量级 ONNX 导出
├── make_fruit_npz.py           # 生成验证数据集 .npz
├── spilt_data.py               # 数据集划分工具
├── class_indices.json          # 标准版类别索引
├── class_indices_lightweight.json  # 轻量级类别索引
└── data_set/                   # 数据集目录
    └── fruit_data/
        ├── train/              # 训练集 (ImageFolder 格式)
        ├── val/                # 验证集
        └── spilt_data.py       # 数据划分脚本
```

## 功能特性

- **双版本训练**：标准版 (224x224) 和轻量级版 (64x64, alpha=0.75)
- **预训练权重加载**：支持 PyTorch 官方 MobileNetV2 预训练权重迁移学习
- **ONNX 导出**：支持 FP32 和 INT8 量化导出
- **INT8 量化**：支持动态量化和静态量化（需校准图片）
- **轻量化部署**：轻量级版本参数量减少约 44%，推理速度提升约 3x

## 快速开始

### 环境依赖

```bash
pip install torch torchvision pillow numpy matplotlib tqdm onnx onnxruntime
```

### 1. 数据集准备

将水果图片按以下结构组织（ImageFolder 格式）：

```
data_set/fruit_data/
├── train/
│   ├── apple/
│   │   ├── 0.jpg
│   │   └── ...
│   ├── banana/
│   ├── grape/
│   ├── orange/
│   └── pear/
└── val/
    ├── apple/
    └── ...
```

如果需要从统一目录划分数据集：

```bash
python spilt_data.py
```

### 2. 训练模型

**标准版训练**（推荐用于高精度场景）：

```bash
# 需要先下载预训练权重
# 下载地址: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
python train.py
```

**轻量级训练**（推荐用于移动端/嵌入式部署）：

```bash
python train_lightweight.py
```

训练参数对比：

| 参数 | 标准版 | 轻量级版 |
|------|--------|----------|
| 输入尺寸 | 224×224 | 64×64 |
| Alpha (宽度乘子) | 1.0 | 0.75 |
| Batch Size | 32 | 64 |
| Epochs | 15 | 20 |
| 学习率 | 0.0001 | 0.001 |

### 3. 预测

```bash
python predict.py
```

修改 `predict.py` 中的 `img_path` 和 `model_weight_path` 即可预测自己的图片。

### 4. 导出 ONNX

**标准版导出**：

```bash
# FP32 导出
python export_onnx.py --weights MobileNetV2.pth --output MobileNetV2.onnx --input-size 224

# FP32 + INT8 动态量化
python export_onnx.py --weights MobileNetV2.pth --export-int8 --quant-method dynamic --input-size 224

# FP32 + INT8 静态量化（需要校准图片）
python export_onnx.py --weights MobileNetV2.pth --export-int8 --quant-method static \
    --calib-dir data_set/fruit_data/val --input-size 224
```

**轻量级导出**：

```bash
# FP32 导出
python export_lightweight.py --weights MobileNetV2_lightweight.pth --output MobileNetV2_light.onnx --input-size 64

# FP32 + INT8 量化
python export_lightweight.py --weights MobileNetV2_lightweight.pth --export-int8 --input-size 64
```

### 5. 生成验证数据集

```bash
python make_fruit_npz.py --dataset-dir data_set/fruit_data/val --input-size 96
```

## 模型说明

### MobileNetV2 架构

- **Inverted Residual Blocks**：倒残差结构，先升维再降维
- **Depthwise Separable Convolution**：深度可分离卷积，大幅减少参数量
- **Linear Bottleneck**：线性瓶颈层，避免 ReLU 信息丢失
- **宽度乘子 (Alpha)**：控制通道数缩放比例

### 量化说明

| 量化方式 | 说明 | 适用场景 |
|----------|------|----------|
| 动态量化 | 推理时动态计算量化参数 | 通用场景，无需校准数据 |
| 静态量化 | 使用校准图片预先计算量化参数 | 追求更高精度 |

## 支持的类别

```json
{
    "0": "apple",
    "1": "banana",
    "2": "grape",
    "3": "orange",
    "4": "pear"
}
```

## 许可证

本项目仅供学习和研究使用。
