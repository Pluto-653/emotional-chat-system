"""训练 ResNet18 表情识别模型。

常用命令：
  粗糙版(先给B联调):  python train.py --data_root data/fer2013 --epochs 3 --freeze_backbone --out_dir runs/rough
  正式训练:            python train.py --data_root data/fer2013 --epochs 40 --out_dir runs/full
  对比实验 E1:         加 --freeze_backbone      (只训分类头)
  对比实验 E2:         加 --no_aug               (不做数据增强)
  对比实验 E3:         加 --class_weight         (类别权重)
输出(out_dir)：best.pt(纯state_dict, 给B) last.pt(断点) train_log.csv curves.png meta.json
"""
import argparse
import csv
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import MEAN, STD, build_datasets
from labels import LABELS, NUM_CLASSES
from net import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="fer2013.csv 所在目录，或含 train/ test/ 的目录")
    p.add_argument("--out_dir", default="runs/full")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_backbone", action="store_true", help="E1: 只训练分类头")
    p.add_argument("--no_aug", action="store_true", help="E2: 关闭数据增强")
    p.add_argument("--class_weight", action="store_true", help="E3: 使用类别权重")
    p.add_argument("--no_pretrained", action="store_true", help="不加载ImageNet权重(服务器无法联网时的应急/冒烟测试)")
    p.add_argument("--resume", default="", help="从 last.pt 断点继续训练")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, preds, gts = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        preds.append(out.argmax(1).cpu())
        gts.append(y.cpu())
    preds, gts = torch.cat(preds).numpy(), torch.cat(gts).numpy()
    return total_loss / len(gts), accuracy_score(gts, preds), f1_score(gts, preds, average="macro")


def plot_curves(rows, path):
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["train_loss"] for r in rows], label="train loss")
    ax[0].plot(ep, [r["val_loss"] for r in rows], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [r["train_acc"] for r in rows], label="train acc")
    ax[1].plot(ep, [r["val_acc"] for r in rows], label="val acc")
    ax[1].plot(ep, [r["val_f1"] for r in rows], label="val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[环境] device={device}  torch={torch.__version__}  torchvision={torchvision.__version__}")

    # ---------- 数据 ----------
    ds = build_datasets(args.data_root, img_size=args.img_size, aug=not args.no_aug, seed=args.seed)
    common = dict(num_workers=args.num_workers, pin_memory=use_amp)
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False, **common)

    # ---------- 模型 ----------
    model = build_model(pretrained=not args.no_pretrained).to(device)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    if args.freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False
        groups = [{"params": model.fc.parameters(), "lr": args.lr_head}]
    else:
        groups = [{"params": backbone_params, "lr": args.lr_backbone},
                  {"params": model.fc.parameters(), "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    weight = None
    if args.class_weight:
        counts = np.bincount(ds["train"].labels, minlength=NUM_CLASSES).astype(np.float32)
        weight = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32).to(device)
        print("[类别权重]", {k: round(float(v), 2) for k, v in zip(LABELS, weight.cpu())})
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------- 断点续训 ----------
    start_epoch, best_f1, rows = 1, -1.0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_f1, rows = ck["epoch"] + 1, ck["best_f1"], ck["rows"]
        print(f"[续训] 从第 {start_epoch} 轮继续")

    # ---------- 训练 ----------
    best_path = os.path.join(args.out_dir, "best.pt")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if args.freeze_backbone:  # 冻结时也固定 BN 统计量
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        scheduler.step()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": run_loss / n, "train_acc": correct / n,
               "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}
        rows.append(row)
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)  # 纯 state_dict，给 B 用
            flag = "  <-- best"
        print(f"epoch {epoch:02d}/{args.epochs}  train_loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} F1 {val_f1:.4f}{flag}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_f1": best_f1, "rows": rows},
                   os.path.join(args.out_dir, "last.pt"))
        with open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    plot_curves(rows, os.path.join(args.out_dir, "curves.png"))

    # ---------- 用最佳权重在测试集上评估 ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    best_row = max(rows, key=lambda r: r["val_f1"])
    print(f"\n[最终] best epoch={best_row['epoch']}  val_acc={best_row['val_acc']:.4f}  "
          f"test_acc={test_acc:.4f}  test_macro_F1={test_f1:.4f}")

    # ---------- meta.json：B 的推理预处理必须严格按此文件 ----------
    meta = {
        "arch": "resnet18", "labels": LABELS, "num_classes": NUM_CLASSES,
        "input_size": args.img_size, "input_channels": 3,
        "preprocess": "人脸裁剪(外扩约10%) -> 灰度 -> 复制成3通道 -> Resize(input_size,input_size) -> ToTensor(0-1) -> Normalize(mean,std)",
        "mean": MEAN, "std": STD,
        "fc": "Sequential(Dropout(0.3), Linear(512, 7))",
        "weights_format": "state_dict",
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "best_epoch": best_row["epoch"], "val_acc": round(best_row["val_acc"], 4),
        "test_acc": round(test_acc, 4), "test_macro_f1": round(test_f1, 4),
        "options": {"freeze_backbone": args.freeze_backbone, "aug": not args.no_aug,
                    "class_weight": args.class_weight, "pretrained": not args.no_pretrained},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[完成] 产物在 {args.out_dir}/ : best.pt  meta.json  train_log.csv  curves.png")


if __name__ == "__main__":
    main()
