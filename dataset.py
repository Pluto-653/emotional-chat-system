"""FER2013 数据加载与预处理。

同时支持两种常见格式（自动识别）：
  1) CSV 版：data_root 下有 fer2013.csv（列：emotion, pixels, Usage）
     划分：Training 28709 / PublicTest 3589(验证) / PrivateTest 3589(测试)
  2) 图片文件夹版：data_root/train/<类名>/*.jpg 与 data_root/test/<类名>/*.jpg
     划分：train 中按 9:1 分层拆出验证集，test 目录作为测试集
两种格式最终标签都按 labels.py 的 LABELS 顺序映射。
"""
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from labels import LABELS, LABEL2ID

MEAN = [0.485, 0.456, 0.406]  # ImageNet 均值方差（预训练权重要求）
STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_transform(img_size: int = 112, train: bool = False, aug: bool = True, normalize: bool = True):
    """训练/验证的预处理。推理端（后端）必须复刻验证版本。

    normalize=False 仅用于 analysis.py 可视化增强效果。
    """
    tfs = [transforms.Grayscale(num_output_channels=3)]  # 灰度复制成 3 通道
    if train and aug:
        tfs += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
        if normalize:
            tfs.append(transforms.Normalize(MEAN, STD))
        tfs.append(transforms.RandomErasing(p=0.25))
    else:
        tfs += [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if normalize:
            tfs.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(tfs)


class FolderDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.labels = [s[1] for s in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def get_pil(self, i):
        return Image.open(self.samples[i][0]).convert("L")

    def __getitem__(self, i):
        img = self.get_pil(i)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


class CSVDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images  # (N, 48, 48) uint8
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def get_pil(self, i):
        return Image.fromarray(self.images[i])  # 二维 uint8 -> 灰度图 "L"

    def __getitem__(self, i):
        img = self.get_pil(i)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def _scan_folder(split_dir: str) -> List[Tuple[str, int]]:
    samples = []
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        if cls.lower() not in LABEL2ID:
            raise ValueError(f"未知类别文件夹 '{cls}'，应为 {LABELS} 之一")
        for f in sorted(os.listdir(cls_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                samples.append((os.path.join(cls_dir, f), LABEL2ID[cls.lower()]))
    return samples


def detect_format(data_root: str) -> Tuple[str, str]:
    if os.path.isfile(data_root) and data_root.lower().endswith(".csv"):
        return "csv", data_root
    csv_path = os.path.join(data_root, "fer2013.csv")
    if os.path.isfile(csv_path):
        return "csv", csv_path
    if os.path.isdir(os.path.join(data_root, "train")) and os.path.isdir(os.path.join(data_root, "test")):
        return "folder", data_root
    raise FileNotFoundError(
        f"在 {data_root} 下没找到 fer2013.csv，也没找到 train/ 与 test/ 文件夹，请检查 --data_root"
    )


def build_datasets(data_root: str, img_size: int = 112, aug: bool = True,
                   val_ratio: float = 0.1, seed: int = 42) -> Dict:
    """返回 {'train','val','test','fmt'}。"""
    fmt, path = detect_format(data_root)
    t_train = build_transform(img_size, train=True, aug=aug)
    t_eval = build_transform(img_size, train=False)

    if fmt == "csv":
        import pandas as pd
        df = pd.read_csv(path)
        pixels = np.stack([np.array(p.split(), dtype=np.uint8) for p in df["pixels"]]).reshape(-1, 48, 48)
        labels = df["emotion"].values

        def part(usage, tf):
            m = (df["Usage"] == usage).values
            return CSVDataset(pixels[m], labels[m], tf)

        train, val, test = part("Training", t_train), part("PublicTest", t_eval), part("PrivateTest", t_eval)
    else:
        all_train = _scan_folder(os.path.join(path, "train"))
        y = [s[1] for s in all_train]
        idx_tr, idx_val = train_test_split(range(len(all_train)), test_size=val_ratio,
                                           stratify=y, random_state=seed)
        train = FolderDataset([all_train[i] for i in idx_tr], t_train)
        val = FolderDataset([all_train[i] for i in idx_val], t_eval)
        test = FolderDataset(_scan_folder(os.path.join(path, "test")), t_eval)

    print(f"[数据] 格式={fmt}  train={len(train)}  val={len(val)}  test={len(test)}")
    return {"train": train, "val": val, "test": test, "fmt": fmt}
