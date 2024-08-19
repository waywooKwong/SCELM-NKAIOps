import pandas as pd

def analyze_anomalies(diff_trend_csv, csv_anomaly_csv, result_csv):
    # 读取get_diff_trend.csv文件
    diff_trend_df = pd.read_csv(diff_trend_csv)
    
    # 检查是否存在“指标名称”列
    if '指标名称' not in diff_trend_df.columns:
        print("get_diff_trend.csv 中没有找到 '训练集名称' 列")
        return

    # 提取指标名称列表
    indices = diff_trend_df['指标名称'].tolist()
    
    if not indices:
        print("未能从 get_diff_trend.csv 中提取到有效的指标名称数组。")
        return

    # 读取csv_anomaly.csv文件
    csv_anomaly_df = pd.read_csv(csv_anomaly_csv)
    
    # 检查是否存在“数据集名称”和“是否出现异常”列
    if '指标名称' not in csv_anomaly_df.columns or '异常情况' not in csv_anomaly_df.columns:
        print("csv_anomaly.csv 中没有找到 '指标名称' 或 '异常情况' 列")
        return

    # 初始化结果列表
    results = []

    # 遍历指标名称列表，检查异常情况
    for index in indices:
        anomaly_row = csv_anomaly_df[csv_anomaly_df['指标名称'] == index]
        if not anomaly_row.empty:
            anomaly_status = anomaly_row['异常情况'].values[0]
            results.append([index, anomaly_status])
        else:
            results.append([index, '未找到对应的异常数据'])

    # 将结果写入新的CSV文件
    result_df = pd.DataFrame(results, columns=['趋势变化的指标', '异常'])
    result_df.to_csv(result_csv, index=False, encoding='utf-8')
    print(f"Saved differing trends to {result_csv}")

# 示例调用
diff_trend_csv = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/get_diff_trend.csv'
csv_anomaly_csv = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/csv_anomaly.csv'
result_csv = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/has_anomaly.csv'

analyze_anomalies(diff_trend_csv, csv_anomaly_csv, result_csv)
