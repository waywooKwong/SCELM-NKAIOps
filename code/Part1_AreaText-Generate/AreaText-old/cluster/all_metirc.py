import os
import csv

# 定义要遍历的文件夹路径
folder_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k1_csv_anomaly'  # 替换为你的文件夹路径
# 定义all_metric.csv的路径
output_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/cluster/all_metric.csv'

# 读取已经存在的指标名称（如果all_metric.csv存在的话）
all_metrics = set()
if os.path.exists(output_file):
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            all_metrics.add(row[0])

# 遍历文件夹中的所有csv文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row['指标名称']
                if metric_name not in all_metrics:
                    all_metrics.add(metric_name)

# 将新的指标名称写入all_metric.csv
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['全部指标名称'])  # 写入标题行
    for metric in sorted(all_metrics):
        writer.writerow([metric])

print(f"已更新 {output_file}")
