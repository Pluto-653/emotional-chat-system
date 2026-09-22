"""摄像头实时表情识别小demo——在本机(有摄像头的电脑)运行，不需要GPU。

用法：
  python webcam_demo.py --weights best.pt --meta meta.json

运行后弹出摄像头窗口，实时框出人脸并显示表情+置信度。
按 q 键退出。

依赖: pip install torch torchvision opencv-python pillow
（这几个都是CPU版即可，单帧推理在CPU上通常几十毫秒）
"""
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 每类一个颜色，方便区分 (BGR)
COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "neutral": (200, 200, 200),
}


def build_model(num_classes=7):
    from torchvision.models import resnet18
    import torch.nn as nn
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--camera", type=int, default=0, help="摄像头编号，通常0是默认摄像头")
    p.add_argument("--expand", type=float, default=0.1, help="人脸框外扩比例，需与训练/meta.json 保持一致的裁剪逻辑")
    args = p.parse_args()

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    img_size = meta["input_size"]
    mean, std = meta["mean"], meta["std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(LABELS)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    face_cascade = cv2.CascadeClassifier(
        r"D:\opencv_data\haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("级联分类器加载失败，检查 XML 路径")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("打不开摄像头，检查 --camera 编号或摄像头权限/占用情况")

    print("[提示] 摄像头窗口打开后，按 q 键退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            # 外扩一定比例，和训练时的预处理逻辑对齐
            ex, ey = int(w * args.expand), int(h * args.expand)
            x1, y1 = max(0, x - ex), max(0, y - ey)
            x2, y2 = min(frame.shape[1], x + w + ex), min(frame.shape[0], y + h + ey)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("L")
            inp = tf(pil_img).unsqueeze(0).to(device)
            with torch.inference_mode():
                probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label, conf = LABELS[idx], probs[idx]
            color = COLORS[label]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.0%}"
            cv2.putText(frame, text, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Expression Demo (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
