import subprocess
import os
import cv2
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont

def run_training_script(script_name):
    """调用训练脚本并等待其完成"""
    try:
        print(f"正在执行 {script_name} ...")
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} 执行完成！")
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 时出错: {e}")

import os
import pandas as pd
import re

def get_latest_folders(base_path, num=1):
    """获取文件夹中的最新子文件夹"""
    # 获取所有文件夹的路径
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # 使用正则表达式提取文件夹中的数字，并按数字排序
    def extract_number(folder_name):
        """从文件夹名中提取数字，如果没有找到返回 -1"""
        match = re.search(r'\d+', folder_name)
        return int(match.group()) if match else -1  # 如果没找到数字，返回 -1
    
    # 使用改进的排序逻辑
    folders.sort(key=lambda x: extract_number(x), reverse=True)
    
    # 返回最新的 num 个文件夹
    return folders[:num]

def read_best_data_from_csv(model_name, base_path):
    """读取每个模型的 results.csv 文件并提取需要的列"""
    # 获取模型对应的最新文件夹
    if model_name in ["yolov8", "hyperyolo"]:
        # 对于 YOLO V8 和 Hyper YOLO，从 runs/detect/ 获取
        latest_folder = get_latest_folders(base_path, 1)[0]  # 获取最新的文件夹
    else:
        # 对于 YOLO V10，从 runs/ 获取
        latest_folder = get_latest_folders("runs", 1)[0]  # 获取最新的文件夹

    csv_path = os.path.join(base_path, latest_folder, "results.csv")
    
    if os.path.exists(csv_path):
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)
        
        # 根据模型类型提取不同的列
        if model_name == "yolov8":
            best_data = {
                "train/box_loss": df.iloc[:, 2].min(),  # train/box_loss 列的索引为 2
                "train/cls_loss": df.iloc[:, 3].min(),  # train/cls_loss 列的索引为 3
                "train/dfl_loss": df.iloc[:, 4].min(),  # train/dfl_loss 列的索引为 4
                "metrics/precision(B)": df.iloc[:, 5].max(),  # metrics/precision(B) 列的索引为 5
                "metrics/recall(B)": df.iloc[:, 6].max(),  # metrics/recall(B) 列的索引为 6
                "metrics/mAP50(B)": df.iloc[:, 7].max(),  # metrics/mAP50(B) 列的索引为 7
                "metrics/mAP50-95(B)": df.iloc[:, 8].max()  # metrics/mAP50-95(B) 列的索引为 8
            }
        elif model_name == "yolov10":
            # 对于 YOLO V10，假设文件结构类似，只需调整为正确的路径
            best_data = {
                "train/box_loss": df.iloc[:, 2].min(),
                "train/cls_loss": df.iloc[:, 3].min(),
                "train/dfl_loss": df.iloc[:, 4].min(),
                "metrics/precision(B)": df.iloc[:, 7].max(),
                "metrics/recall(B)": df.iloc[:, 8].max(),
                "metrics/mAP50(B)": df.iloc[:, 9].max(),
                "metrics/mAP50-95(B)": df.iloc[:, 10].max()
            }
        elif model_name == "hyperyolo":
            best_data = {
                "train/box_loss": df.iloc[:, 1].min(),
                "train/cls_loss": df.iloc[:, 2].min(),
                "train/dfl_loss": df.iloc[:, 3].min(),
                "metrics/precision(B)": df.iloc[:, 7].max(),
                "metrics/recall(B)": df.iloc[:, 8].max(),
                "metrics/mAP50(B)": df.iloc[:, 9].max(),
                "metrics/mAP50-95(B)": df.iloc[:, 10].max()
            }
        else:
            print(f"未知模型 {model_name}，无法处理该文件")
            return None
        
        return best_data
    else:
        print(f"未找到 {latest_folder} 目录中的 results.csv 文件")
        return None

def concatenate_results(base_path):
    """合并三个模型的 results.csv 文件，按相同属性对齐"""
    # 模型名称
    model_names = ['yolov8', 'yolov10', 'hyperyolo']
    
    # 用于存储每个模型的结果
    model_results = []
    
    # 读取每个模型的 results.csv 文件
    for model_name in model_names:
        if model_name == "yolov8" or model_name == "yolov10":
            model_base_path = os.path.join(base_path, "detect")
        else:
            model_base_path = "runs"
        
        best_data = read_best_data_from_csv(model_name, model_base_path)
        if best_data is not None:
            # 将每个模型的数据添加到模型结果列表中
            model_results.append(best_data)
    
    # 合并所有模型的结果（按列拼接）
    if len(model_results) == 3:
        # 转换为 DataFrame
        final_df = pd.DataFrame(model_results, columns=["train/box_loss", "train/cls_loss", "train/dfl_loss", 
                                                        "metrics/precision(B)", "metrics/recall(B)", 
                                                        "metrics/mAP50(B)", "metrics/mAP50-95(B)"], 
                                index=["YOLO V8", "YOLO V10", "Hyper YOLO"])
        
        # 返回合并后的 DataFrame
        return final_df
    else:
        print("未能读取到足够的 CSV 文件")
        return None

def save_concatenated_csv(final_df, output_path):
    """保存合并后的 CSV 文件"""
    if final_df is not None:
        final_df.to_csv(output_path)
        print(f"合并后的 CSV 文件已保存到 {output_path}")
    else:
        print("未能保存 CSV 文件")


if __name__ == "__main__":
    # 设置 base_path 为 runs/detect/ 目录
    base_path = "runs/detect/"

    # 训练文件名称列表
    training_scripts = [
        "ultralytics-main/train.py", 
        "yolov10-main/train.py", 
        "Hyper-YOLO-main/train.py"
    ]

    # 依次执行训练脚本
    for script in training_scripts:
        run_training_script(script)

    # 设置 base_path 为 runs/detect/ 目录
    base_path = "runs/"
    
    # 合并三个模型的结果
    final_df = concatenate_results(base_path)
    
    # 保存合并后的 CSV 文件
    if final_df is not None:
        save_concatenated_csv(final_df, "final_result.csv")
