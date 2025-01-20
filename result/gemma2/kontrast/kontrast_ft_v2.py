import os
import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_f1(tp, fn, fp):
    """计算 F1 分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def calculate_precision(tp, fp):
    """计算精度"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def calculate_recall(tp, fn):
    """计算召回率"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def compare_ft_csv(base_folder, target_folder):
    # 记录所有的 ft 类型
    ft_item = set()
    # 初始化矩阵
    ft_matrix = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == "ft.csv":
                # 获取 base 和 target 的对应路径
                base_path = os.path.join(root, file)
                relative_path = os.path.relpath(base_path, base_folder)
                target_path = os.path.join(target_folder, relative_path)

                if not os.path.exists(target_path):
                    print(f"Target file not found: {target_path}")
                    continue

                # 读取两个 CSV 文件
                base_df = pd.read_csv(base_path)
                target_df = pd.read_csv(target_path)

                # 合并文件以进行对比
                merged_df = pd.merge(
                    base_df, target_df, on="id", suffixes=("_base", "_target")
                )

                # 计算虚拟项 "Expected Software Changes"
                virtual_ft = "Expected Software Changes"
                virtual_rows = pd.DataFrame({
                    "id": [None] * len(merged_df),  # 无需 ID
                    "ft_base": [virtual_ft] * len(merged_df),
                    "ft_target": [virtual_ft] * len(merged_df),
                })

                # 为 merged_df 增加虚拟项
                merged_df = pd.concat([merged_df, virtual_rows], ignore_index=True)

                # 遍历每一行，更新矩阵并记录 ft 类型
                for _, row in merged_df.iterrows():
                    ft_base = row["ft_base"]
                    ft_target = row["ft_target"]
                    ft_item.add(ft_base)
                    ft_item.add(ft_target)
                    ft_matrix[ft_base][ft_target] += 1

    # 将 ft_item 转换为列表并排序，方便矩阵计算
    ft_item = sorted(ft_item)
    num_classes = len(ft_item)

    # 构建矩阵（行：true，列：predict）
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    ft_index = {ft: i for i, ft in enumerate(ft_item)}  # 类型到索引的映射

    for ft_true, pred_dict in ft_matrix.items():
        for ft_pred, count in pred_dict.items():
            i, j = ft_index[ft_true], ft_index[ft_pred]
            matrix[i, j] += count

    # 计算每个 ft 的 F1 分数、精度和召回率，并计算权重
    f1_scores = []
    precision_scores = []
    recall_scores = []
    weights = []
    total_true = matrix.sum(axis=1)  # 每个类型的总 true 数量

    for i, ft in enumerate(ft_item):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = matrix.sum() - (tp + fn + fp)

        # 计算 F1、精度、召回率
        f1 = calculate_f1(tp, fn, fp)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        
        # 计算权重
        weight = total_true[i] / total_true.sum() if total_true.sum() > 0 else 0

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        weights.append(weight)

        print(f"FT: {ft}, TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Weight: {weight:.4f}")

    # 计算加权 F1 分数、加权精度和加权召回率
    weighted_f1 = sum(w * f for w, f in zip(weights, f1_scores))
    weighted_precision = sum(w * p for w, p in zip(weights, precision_scores))
    weighted_recall = sum(w * r for w, r in zip(weights, recall_scores))

    # 输出结果
    print("\nOverall Results:")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")


# 使用示例
base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20250105_063317_baseline"  # 替换为您的根文件夹路径
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241124_164151"
target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20250105_063317_baseline"
compare_ft_csv(base_folder, target_folder)
