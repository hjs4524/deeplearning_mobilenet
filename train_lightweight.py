"""
轻量级 MobileNetV2 训练脚本
- alpha=0.75: 通道数缩减为原来的75%
- input_size=64: 输入分辨率降低到64x64
- 预计推理时间: ~200ms (相比原版700ms)
"""
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from mobilenetv2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # ============ 轻量级参数配置 ============
    ALPHA = 0.75          # 宽度乘子: 0.75 (通道数变为原来的75%)
    INPUT_SIZE = 64       # 输入分辨率: 64x64 (原来是96x96)
    BATCH_SIZE = 64       # 小模型可以用更大的batch
    EPOCHS = 20           # 增加训练轮次补偿模型容量下降
    LR = 0.001            # 稍大的学习率
    # ========================================

    print(f"Config: alpha={ALPHA}, input_size={INPUT_SIZE}, batch_size={BATCH_SIZE}")

    # 数据预处理 - 使用64x64输入
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(INPUT_SIZE * 256 / 224)),  # 等比缩放
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "fruit_data")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    # 保存类别索引
    fruit_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in fruit_list.items())
    with open('class_indices_lightweight.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
        pin_memory=True
    )

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )

    print(f"Training: {train_num} images, Validation: {val_num} images")

    # 创建轻量级模型 (alpha=0.75)
    net = MobileNetV2(num_classes=5, alpha=ALPHA).to(device)

    # 统计模型参数量
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # 尝试加载预训练权重 (部分加载，因为alpha不同)
    pretrained_path = "./mobilenet_v2.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pre_weights = torch.load(pretrained_path, map_location='cpu')

        # 只加载形状匹配的权重
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained")

    # 冻结前几层特征提取器，只微调后面的层
    for i, param in enumerate(net.features.parameters()):
        if i < 10:  # 冻结前10个参数组
            param.requires_grad = False

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    optimizer = optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练
    best_acc = 0.0
    save_path = './MobileNetV2_lightweight.pth'

    for epoch in range(EPOCHS):
        # ========== Train ==========
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.desc = f"train epoch[{epoch+1}/{EPOCHS}] loss:{loss.item():.3f} acc:{100.*correct/total:.1f}%"

        scheduler.step()

        # ========== Validate ==========
        net.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                _, predicted = outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += predicted.eq(val_labels).sum().item()
                val_bar.desc = f"val epoch[{epoch+1}/{EPOCHS}]"

        val_acc = val_correct / val_total
        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        print(f'[Epoch {epoch+1}/{EPOCHS}] train_loss: {avg_loss:.3f} '
              f'train_acc: {train_acc:.3f} val_acc: {val_acc:.3f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f'  -> Saved best model (acc={val_acc:.3f})')

    print(f'\nTraining complete! Best val accuracy: {best_acc:.3f}')
    print(f'Model saved to: {save_path}')


if __name__ == '__main__':
    main()
