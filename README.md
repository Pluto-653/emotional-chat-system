# model/ —— 成员 A：FER2013 + ResNet18

## 目录
labels.py 标签顺序（全项目唯一） · net.py 模型 · dataset.py 数据与预处理
analysis.py 数据分析 · train.py 训练 · evaluate.py 评测

## 步骤
```bash
pip install -r requirements.txt

# 1. 数据分析（生成 analysis/ 下 3 张图 + 终端统计）
python analysis.py --data_root data/fer2013 --out_dir analysis

# 2. 粗糙版（3轮，先发给 B 联调）
python train.py --data_root data/fer2013 --epochs 3 --freeze_backbone --out_dir runs/rough

# 3. 正式训练（建议在 tmux 中运行）
python train.py --data_root data/fer2013 --epochs 40 --out_dir runs/full

# 4. 对比实验（每次只改一个条件，输出目录不同）
python train.py --data_root data/fer2013 --epochs 40 --freeze_backbone --out_dir runs/e1_freeze
python train.py --data_root data/fer2013 --epochs 40 --no_aug          --out_dir runs/e2_noaug
python train.py --data_root data/fer2013 --epochs 40 --class_weight    --out_dir runs/e3_cw

# 5. 评测（准确率 / 每类 P R F1 / 混淆矩阵）
python evaluate.py --data_root data/fer2013 --weights runs/full/best.pt --out_dir runs/full/eval

# 断线续训
python train.py --data_root data/fer2013 --epochs 40 --out_dir runs/full --resume runs/full/last.pt
```

## 交付给 B（微信/网盘）
runs/full/best.pt + runs/full/meta.json（权重不要提交到 git）
