# YOLO Training and Evaluation Script

## 项目介绍

该项目包含了三个不同的目标检测模型：YOLOv8、YOLOv10 和 Hyper YOLO。项目的目的是通过训练这些模型，并从训练结果中提取最佳指标来进行比较和评估。它还提供了一套自动化脚本来执行模型训练、合并结果以及保存最终评估报告。

### 项目功能
- 自动运行YOLOv8、YOLOv10 和 Hyper YOLO的训练脚本。
- 自动提取每个模型的 `results.csv` 文件中的训练损失（如 `box_loss`、`cls_loss`）和评估指标（如 `precision`、`recall`、`mAP50` 等）。
- 将三个模型的结果合并为一个 CSV 文件，方便后续的对比和分析。
- 支持自动下载并训练 COCO128 数据集。

## 原作者声明

- **YOLOv8**：https://github.com/ultralytics/ultralytics
- **YOLOv10**：https://github.com/THU-MIG/yolov10
- **Hyper YOLO**：https://github.com/iMoonLab/Hyper-YOLO

感谢这些项目的开源和贡献，所有的代码和实现均来自这些开源库，原作者保留所有权利。

## 项目用途

本项目旨在为目标检测任务提供一种自动化、简便的方式来比较不同版本的 YOLO 模型。通过运行预训练的模型并从其训练过程中提取相关性能指标，您可以快速评估每个模型在给定数据集上的表现。

此外，项目也支持通过`COCO128`数据集进行训练和验证，`COCO128`是`COCO`数据集的一个小型子集，适用于快速实验和调试。

## COCO128 数据集介绍

`COCO128` 是 `COCO` 数据集的简化版本，由128张图片构成，包含80个类别。`COCO128` 提供了一个较小但完整的图像数据集，可以用于快速测试和评估目标检测模型。它非常适合用于模型的快速训练和验证，尤其是在实验阶段。

COCO 数据集是计算机视觉领域最广泛使用的标准数据集之一，包含多种不同的目标类别，适用于目标检测、实例分割和关键点检测等任务。COCO128 数据集作为其子集，保持了与 COCO 数据集相同的标注格式，但减少了样本数量，从而使得它成为了快速实验和开发的理想选择。

### 数据集链接
- [COCO数据集](http://cocodataset.org/)
- [COCO128 数据集下载链接](https://github.com/ultralytics/coco128)