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

def save_differing_trends_to_csv(differing_trends, csv_anomaly_path, result_path):
    # 读取csv_anomaly.csv文件
    csv_anomaly_df = pd.read_csv(csv_anomaly_path)
    
    data = []
    for index, details in differing_trends.items():
        train_details = details.get('训练集', {})
        test_details = details.get('测试集', {})
        metric_name = train_details.get('name', '')

        # 在csv_anomaly_df中找到对应的异常情况
        anomaly_row = csv_anomaly_df[csv_anomaly_df['指标名称'] == metric_name]
        if not anomaly_row.empty:
            anomaly_status = anomaly_row['异常情况'].values[0]
        else:
            anomaly_status = '未找到对应的异常数据'

        data.append([
            metric_name,
            train_details.get('mean', ''),
            test_details.get('mean', ''),
            train_details.get('trend', ''),
            test_details.get('trend', ''),
            anomaly_status
        ])
    
    df = pd.DataFrame(data, columns=[
        '指标名称', '训练集均值', '测试集均值', '训练集趋势', '测试集趋势', '异常情况'
    ])
    df.to_csv(result_path, index=False, encoding='utf-8')

input_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt'
csv_anomaly_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/csv_anomaly.csv'
result_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/diff_and_anomaly.csv'

differing_trends, all_indices = parse_metrics(input_file)
save_differing_trends_to_csv(differing_trends, csv_anomaly_path, result_path)

print(f"Saved differing trends to {result_path}")
