import re

def analyze_anomalies(diff_trend_path, csv_anomaly_path, result_path):
    indices = []
    # 读取get_diff_trend.txt文件中的序号数组
    with open(diff_trend_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) > 1:
            index_line = lines[1].strip()
            # 提取序号数组
            match = re.search(r'\[([0-9, ,]*)\]', index_line)
            if match:
                indices = [int(x) for x in match.group(1).split(',') if x.strip().isdigit()]

    if not indices:
        print("未能从 get_diff_trend.txt 中提取到有效的序号数组。")
        return

    # 初始化结果列表
    results = []

    # 检查csv_anomaly.txt文件中对应行的异常情况
    anomaly_count = 0
    no_anomaly_count = 0
    with open(csv_anomaly_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index in indices:
            if index < len(lines):
                line = lines[index].strip()
                if '出现异常' in line:
                    results.append(f"{index} 出现异常")
                    anomaly_count += 1
                else:
                    results.append(f"{index} 未出现异常")
                    no_anomaly_count += 1
            else:
                results.append(f"{index} 超出文件行数范围")

    # 生成result.txt文件
    with open(result_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')
        f.write(f"\n统计:\n出现异常的数量: {anomaly_count}\n未出现异常的数量: {no_anomaly_count}\n")

# 示例调用
diff_trend_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/get_diff_trend.txt'
csv_anomaly_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/csv_anomaly.txt'
result_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/has_anomaly.txt'

analyze_anomalies(diff_trend_path, csv_anomaly_path, result_path)
