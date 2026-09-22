"""全项目唯一的标签顺序定义。
训练(model/)、推理(backend/app/labels.py)、前端显示必须使用完全相同的顺序。
此顺序与 fer2013.csv 的 emotion 数字一致：0 angry ... 6 neutral。
"""
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
