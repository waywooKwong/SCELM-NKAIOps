import re
import pandas as pd

def parse_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    metrics = {}
    all_indices = set()  # 用于存储所有出现的序号
    for line in lines:
        match = re.match(r'\s*(\d+), 指标(训练集|测试集)：(.+?); ([0-9a-f]+);最小值: (.+?); 最大值: (.+?); 均值: (.+?); 整体的趋势: (.+?);', line)
        if match:
            index = match.group(1)
            dataset_type = match.group(2)
            details = {
                'name': match.group(3),
                'min': match.group(5),
                'max': match.group(6),
                'mean': match.group(7),
                'trend': match.group(8)
            }

            all_indices.add(index)  # 记录所有序号

            if index not in metrics:
                metrics[index] = {}
            metrics[index][dataset_type] = details
    
    differing_trends = {}
    for index, details in metrics.items():
        if '训练集' in details and '测试集' in details:
            train_trend = details['训练集']['trend']
            test_trend = details['测试集']['trend']
            if train_trend != test_trend:
                differing_trends[index] = details
    
    return differing_trends, all_indices

def save_differing_trends_to_csv(differing_trends, file_path):
    data = []
    for index, details in differing_trends.items():
        train_details = details.get('训练集', {})
        test_details = details.get('测试集', {})
        data.append([
            train_details.get('name', ''),
            train_details.get('mean', ''),
            test_details.get('mean', ''),
            train_details.get('trend', ''),
            test_details.get('trend', '')
        ])
    
    df = pd.DataFrame(data, columns=[
        '指标名称', '训练集均值', '测试集均值', '训练集趋势', '测试集趋势'
    ])
    df.to_csv(file_path, index=False, encoding='utf-8')

input_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt'
output_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/get_diff_trend.csv'

differing_trends, all_indices = parse_metrics(input_file)
save_differing_trends_to_csv(differing_trends, output_file)

print(f"Saved differing trends to {output_file}")
