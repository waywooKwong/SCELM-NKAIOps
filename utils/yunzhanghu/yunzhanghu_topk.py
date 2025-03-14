import os
import pandas as pd


def calculate_topk_scores(base_folder, target_folder):
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_items = 0

    for root, _, files in os.walk(target_folder):
        for file in files:
            if file == "top5_kpi.csv":
                # 获取 base 和 target 的对应路径
                target_path = os.path.join(root, file)
                relative_path = os.path.relpath(target_path, target_folder)
                base_path = os.path.join(base_folder, relative_path)

                if not os.path.exists(base_path):
                    print(f"Base file not found: {base_path}")
                    continue

                # 读取 base 和 target 的文件
                try:
                    target_df = pd.read_csv(target_path)
                    base_df = pd.read_csv(base_path)
                except Exception as e:
                    print(f"Error reading files: {e}")
                    continue

                if target_df.empty or base_df.empty:
                    print(f"One of the files is empty: {file}")
                    continue

                if "kpi1" not in target_df.columns:
                    print(f"kpi1 column missing in target dataframe: {file}")
                    continue

                # 提取 target 的 root 数据
                roots = target_df.set_index("id")["kpi1"].to_dict()

                # 遍历 base 数据
                for _, row in base_df.iterrows():
                    try:
                        base_id = int(row["id"])
                        if base_id not in roots:
                            continue

                        root = roots[base_id]
                        kpis = [row[f"kpi{i}"] for i in range(1, 6) if f"kpi{i}" in row]

                        if root in kpis:
                            index = kpis.index(root)
                            if index == 0:
                                top1, top3, top5 = 100, 100, 100
                            elif index in [1, 2]:
                                top1, top3, top5 = 0, 100, 100
                            elif index in [3, 4]:
                                top1, top3, top5 = 0, 0, 100
                        else:
                            top1, top3, top5 = 0, 0, 0

                        total_top1 += top1
                        total_top3 += top3
                        total_top5 += top5
                        total_items += 1
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

                print(f"Processed: {relative_path}")

    # 输出整体结果
    if total_items > 0:
        avg_top1 = total_top1 / total_items
        avg_top3 = total_top3 / total_items
        avg_top5 = total_top5 / total_items
        print("\nOverall Results:")
        print(f"Total Items: {total_items}")
        print(f"Average Top1: {avg_top1:.2f}%")
        print(f"Average Top3: {avg_top3:.2f}%")
        print(f"Average Top5: {avg_top5:.2f}%")
    else:
        print("No valid items were processed.")


# 使用示例
base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/yunzhanghu/20250102_142707"
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241124_164151"
target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2:2b/yunzhanghu/20250106_095700_gemma2:2b"
# base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/llama3.1/yunzhanghu/20241222_122759"
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/llama3.1/yunzhanghu/20241222_123505"
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241122_172528"

calculate_topk_scores(base_folder, target_folder)