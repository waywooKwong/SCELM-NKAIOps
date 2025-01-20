import os
import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_f1_score(base_folder):
    total_count = 0
    TP = TN = FP = FN = 0

    # 遍历所有子文件夹
    for root, _, files in os.walk(base_folder):
        if "change_result.csv" in files:
            file_path = os.path.join(root, "change_result.csv")
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 遍历每一行，提取id和change_type
            for _, row in df.iterrows():
                id_value = str(row["id"]).zfill(5)  # 确保id是五位字符串
                change_type = row["change_type"]

                # 实际类型判定
                actual_type = "normal" if id_value > "20000" else "failure"

                # 统计总数
                total_count += 1

                # 比较实际类型与预测类型
                if actual_type == "normal":
                    if change_type == "normal":
                        TN += 1
                    else:
                        FP += 1
                elif actual_type == "failure":
                    if change_type == "failure":
                        TP += 1
                    else:
                        FN += 1

    # 计算 precision, recall 和 F1 分数
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # 打印统计结果
    print(f"总数: {total_count}")
    print(f"TP (True Positive): {TP}")
    print(f"TN (True Negative): {TN}")
    print(f"FP (False Positive): {FP}")
    print(f"FN (False Negative): {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def calculate_similarity_score(base_folder):
    reason_scores = []
    solution_scores = []

    # 遍历所有子文件夹
    for root, _, files in os.walk(base_folder):
        if "change_result.csv" in files:
            file_path = os.path.join(root, "change_result.csv")
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 遍历每一行，提取reason_score和solution_score
            for _, row in df.iterrows():
                # 提取 tensor 中的数字部分
                reason_score = extract_tensor_value(row["reason_score"])
                solution_score = extract_tensor_value(row["solution_score"])

                # 将数值添加到对应的列表
                reason_scores.append(reason_score)
                solution_scores.append(solution_score)

    # 计算均值
    reason_score_mean = sum(reason_scores) / len(reason_scores) if reason_scores else 0
    solution_score_mean = (
        sum(solution_scores) / len(solution_scores) if solution_scores else 0
    )

    print(f"reason similarity score: {reason_score_mean:.4f}")
    print(f"solution similarity score: {solution_score_mean:.4f}")


def extract_tensor_value(tensor_str):
    # 使用正则表达式提取 tensor() 中的数字部分
    match = re.search(r"tensor\((\d+\.\d*|\d+)\)", tensor_str)
    if match:
        return float(match.group(1))
    else:
        return 0.0  # 如果没有匹配到数值，返回默认值


# 使用示例
base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2:2b/yunzhanghu/20250106_095700_gemma2:2b"
# base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/yunzhanghu/20241206_143746_ablation"
# /home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/llama3/yunzhanghu/20241210_162755
calculate_f1_score(base_folder)
calculate_similarity_score(base_folder)
