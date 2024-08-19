import re

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
            details = match.group(0)  # 获取整行匹配内容

            all_indices.add(index)  # 记录所有序号

            if index not in metrics:
                metrics[index] = {}
            metrics[index][dataset_type] = details
    
    differing_trends = {}
    for index, details in metrics.items():
        if '训练集' in details and '测试集' in details:
            train_trend = re.search(r'整体的趋势: (.+?);', details['训练集']).group(1)
            test_trend = re.search(r'整体的趋势: (.+?);', details['测试集']).group(1)
            if train_trend != test_trend:
                differing_trends[index] = details
    
    return differing_trends, all_indices

def save_differing_trends(differing_trends, all_indices, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        differing_indices = list(differing_trends.keys())
        # 将列表中的数字转换为字符串，并使用逗号分隔
        differing_indices_str = ', '.join(differing_indices)
        
        file.write(f"全部序号总数: {len(all_indices)}\n")
        file.write(f"train_test趋势不同序号:[{differing_indices_str}], 总数: {len(differing_indices)}\n\n")
        file.write(f"详细信息：\n")
        for index, details in differing_trends.items():
            file.write(details['训练集'] + '\n')
            file.write(details['测试集'] + '\n')


input_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt'
output_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/get_diff_trend.txt'

differing_trends, all_indices = parse_metrics(input_file)
save_differing_trends(differing_trends, all_indices, output_file)

print(f"Saved differing trends to {output_file}")

