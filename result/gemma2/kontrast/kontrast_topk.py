import os
import pandas as pd


def calculate_topk_scores(base_folder, target_folder):
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_items = 0

    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file == "top5_kpi.csv":
                target_path = os.path.join(root, file)
                relative_path = os.path.relpath(target_path, target_folder)
                base_path = os.path.join(base_folder, relative_path)

                if not os.path.exists(base_path):
                    print(f"Base file not found: {base_path}")
                    continue

                try:
                    target_df = pd.read_csv(target_path, on_bad_lines='skip')
                    base_df = pd.read_csv(base_path)
                except pd.errors.ParserError as e:
                    print(f"Error reading {target_path}: {e}")
                    continue

                target_df_filtered = target_df[target_df["id"] > 50000]
                roots = target_df_filtered.set_index("id")["kpi1"].to_dict()

                for _, row in base_df.iterrows():
                    base_id = row["id"]
                    if base_id not in roots:
                        continue

                    root = roots[base_id]
                    kpis = [row.get(f"kpi{i}", None) for i in range(1, 6)]

                    top1, top3, top5 = 0, 0, 0

                    if root in kpis:
                        index = kpis.index(root)
                        if index == 0:
                            top1, top3, top5 = 100, 100, 100
                        elif index in [1, 2]:
                            top1, top3, top5 = 0, 100, 100
                        elif index in [3, 4]:
                            top1, top3, top5 = 0, 0, 100

                    total_top1 += top1
                    total_top3 += top3
                    total_top5 += top5
                    total_items += 1

                print(f"Processed: {relative_path}")

    avg_top1 = total_top1 / total_items if total_items > 0 else 0
    avg_top3 = total_top3 / total_items if total_items > 0 else 0
    avg_top5 = total_top5 / total_items if total_items > 0 else 0

    print("\nOverall Results:")
    print(f"Total Items: {total_items}")
    print(f"Average Top1: {avg_top1:.2f}%")
    print(f"Average Top3: {avg_top3:.2f}%")
    print(f"Average Top5: {avg_top5:.2f}%")


# 使用示例
# base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/groundtruth-v4"
# base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241124_164151"
base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20250105_063317_baseline"
target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2:2b/kontrast/20250106_102200_gemma2:2b"
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241122_172528"

calculate_topk_scores(base_folder, target_folder)
