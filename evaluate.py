"""在测试集(或验证集)上评测：准确率、每类 P/R/F1、macro-F1、归一化混淆矩阵。

用法： python evaluate.py --data_root data/fer2013 --weights runs/full/best.pt --out_dir runs/full/eval
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import build_datasets
from labels import LABELS
from net import build_model


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out_dir", default="eval")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_datasets(args.data_root, img_size=args.img_size)[args.split]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            gts.append(y)
    preds, gts = torch.cat(preds).numpy(), torch.cat(gts).numpy()

    acc = accuracy_score(gts, preds)
    macro_f1 = f1_score(gts, preds, average="macro")
    ids = list(range(len(LABELS)))
    report = classification_report(gts, preds, labels=ids, target_names=LABELS, digits=4, zero_division=0)
    cm = confusion_matrix(gts, preds, labels=ids, normalize="true")

    print(f"split={args.split}  accuracy={acc:.4f}  macro-F1={macro_f1:.4f}\n")
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"split={args.split}  accuracy={acc:.4f}  macro-F1={macro_f1:.4f}\n\n{report}")
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "accuracy": acc, "macro_f1": macro_f1,
                   "per_class": classification_report(gts, preds, labels=ids, target_names=LABELS,
                                                      output_dict=True, zero_division=0)}, f, indent=2)
    plot_confusion(cm, os.path.join(args.out_dir, "confusion_matrix.png"),
                   f"Normalized confusion matrix ({args.split}, acc={acc:.3f})")
    print(f"[完成] 结果已保存到 {args.out_dir}/")


if __name__ == "__main__":
    main()
