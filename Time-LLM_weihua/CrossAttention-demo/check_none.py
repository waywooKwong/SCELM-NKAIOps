import pandas as pd

def detect_missing_values(csv_file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 检查是否有缺省值
        missing_values = df.isnull() | df.isna()
        
        # 打印缺省值的统计信息
        missing_summary = missing_values.sum()
        total_missing = missing_summary.sum()
        
        if total_missing == 0:
            print(f"No missing values found in {csv_file_path}")
        else:
            print(f"Missing values detected in {csv_file_path}:")
            print(missing_summary)
        
        return df, missing_values
    except Exception as e:
        print(f"Failed to read {csv_file_path}: {e}")
        return None, None

# 示例用法
csv_file_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/anomaly_details11.csv'
df, missing_values = detect_missing_values(csv_file_path)

# 打印有缺省值的行
if missing_values is not None and missing_values.any().any():
    print("Rows with missing values:")
    print(df[missing_values.any(axis=1)])
