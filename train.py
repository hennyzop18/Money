import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    image_size: int
    batch_size: int
    feature_extract_epochs: int # Epochs cho giai đoạn 1
    fine_tune_epochs: int       # Epochs cho giai đoạn 2
    lr_feature_extract: float
    lr_fine_tune: float
    weight_decay: float
    val_ratio: float
    num_workers: int
    seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5), # Tăng xác suất lật
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def build_model(num_classes: int, freeze_backbone: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Đóng băng các lớp của phần thân mô hình (backbone)
    for param in model.features.parameters():
        param.requires_grad = not freeze_backbone
        
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * images.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * images.size(0)
        pred = torch.argmax(logits, dim=1)
        running_correct += int((pred == labels).sum().item())
        running_total += int(images.size(0))
    return running_loss / max(running_total, 1), running_correct / max(running_total, 1)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình EfficientNet")
    # ... Các args cũ vẫn giữ nguyên, nhưng thêm/sửa một vài cái ...
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="artifacts_efficientnet") # Đổi tên thư mục output
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-extract-epochs", type=int, default=10) # Epochs cho Giai đoạn 1
    parser.add_argument("--fine-tune-epochs", type=int, default=20) # Epochs cho Giai đoạn 2
    parser.add_argument("--lr-feature-extract", type=float, default=1e-3)
    parser.add_argument("--lr-fine-tune", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Đã bỏ --epochs và --lr, thay bằng các tham số cho từng giai đoạn
    cfg = TrainConfig(
        data_dir=args.data_dir, output_dir=args.output_dir, image_size=args.image_size,
        batch_size=args.batch_size, feature_extract_epochs=args.feature_extract_epochs,
        fine_tune_epochs=args.fine_tune_epochs, lr_feature_extract=args.lr_feature_extract,
        lr_fine_tune=args.lr_fine_tune, weight_decay=args.weight_decay,
        val_ratio=args.val_ratio, num_workers=args.num_workers, seed=args.seed
    )

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ... Phần chuẩn bị data loader giữ nguyên ...
    train_tf, val_tf = build_transforms(cfg.image_size)
    base_ds = datasets.ImageFolder(cfg.data_dir)
    idx_to_class = {v: k for k, v in base_ds.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    (output_dir / "money_class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")
    full_ds_train = datasets.ImageFolder(cfg.data_dir, transform=train_tf)
    full_ds_val = datasets.ImageFolder(cfg.data_dir, transform=val_tf)
    n_total = len(base_ds)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val
    train_ds, _ = random_split(full_ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    _, val_ds = random_split(full_ds_val, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # --- GIAI ĐOẠN 1: FEATURE EXTRACTION ---
    print("\n--- Bắt đầu Giai đoạn 1: Feature Extraction ---")
    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=cfg.lr_feature_extract, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    best_val_accuracy = -1.0
    best_model_path = output_dir / "efficientnet_best_model.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.feature_extract_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        scheduler.step(val_loss) # Cập nhật learning rate scheduler

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        print(f"Epoch [1/{epoch+1}/{cfg.feature_extract_epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({"model": model.state_dict(), "class_names": class_names, "image_size": cfg.image_size}, best_model_path)
            print(f"-> Đã lưu mô hình tốt nhất với Val Acc = {best_val_accuracy:.4f}")
    
    # --- GIAI ĐOẠN 2: FINE-TUNING ---
    print("\n--- Bắt đầu Giai đoạn 2: Fine-Tuning ---")
    # Mở băng các lớp để fine-tune
    for param in model.features.parameters():
        param.requires_grad = True
    
    # Tải lại trọng số tốt nhất từ giai đoạn 1 và tạo optimizer mới với LR thấp hơn
    model.load_state_dict(torch.load(best_model_path)["model"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_fine_tune, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    for epoch in range(cfg.fine_tune_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)
        
        print(f"Epoch [2/{epoch+1}/{cfg.fine_tune_epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({"model": model.state_dict(), "class_names": class_names, "image_size": cfg.image_size}, best_model_path)
            print(f"-> Đã lưu mô hình tốt nhất với Val Acc = {best_val_accuracy:.4f}")

    print("\nQuá trình huấn luyện hoàn tất!")
    print(f"Mô hình tốt nhất đã được lưu tại: {best_model_path}")
    print(f"Độ chính xác tốt nhất trên tập Validation: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()
