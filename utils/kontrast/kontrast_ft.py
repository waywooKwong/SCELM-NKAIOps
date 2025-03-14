import os
import pandas as pd


def compare_ft_csv(base_folder, target_folder):
    total_same = 0
    total_diff = 0
    total_ids = 0

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

                # 以 `id` 为键进行对比，限制 `id > 50000`
                merged_df = pd.merge(
                    base_df, target_df, on="id", suffixes=("_base", "_target")
                )
                #merged_df = merged_df[merged_df["id"] > 50000]  # 添加限制条件

                # 比较 `ft` 列的值
                same_count = (merged_df["ft_base"] == merged_df["ft_target"]).sum()
                diff_count = len(merged_df) - same_count

                # 更新总计数
                total_same += same_count
                total_diff += diff_count
                total_ids += len(merged_df)

                # 输出单文件结果
                print(f"Compared: {relative_path}")
                print(
                    f"Same: {same_count}, Different: {diff_count}, Total: {len(merged_df)}"
                )

    # 计算总准确率
    accuracy = total_same / total_ids * 100 if total_ids > 0 else 0

    # 输出总结果
    print("\nOverall Results:")
    print(f"Total Same: {total_same}")
    print(f"Total Different: {total_diff}")
    print(f"Total IDs Compared: {total_ids}")
    print(f"Accuracy: {accuracy:.2f}%")


# 使用示例
base_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241122_172528"
# target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241122_165445"
target_folder = "/home/zhengtinghua/kuangweihua/ollama/mydocuments/cluster/gemma2/kontrast/20241124_164151"

compare_ft_csv(base_folder, target_folder)
