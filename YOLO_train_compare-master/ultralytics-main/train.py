# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/28 20:39
@Author  : Mjy
@Site    : 
@File    : train.py
@Software: PyCharm
"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    results = model.train(data="coco128.yaml", epochs=10, imgsz=640, batch=32)
