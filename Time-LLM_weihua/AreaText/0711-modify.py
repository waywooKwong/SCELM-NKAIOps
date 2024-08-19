import pandas as pd

# 示例调用
input_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k4_combine/combined_40001.csv'
output_file = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k5_finetune_modify/modify_40001.csv'

# 读取CSV文件
df = pd.read_csv(input_file)
df['true-label'] = df['prediction']

# 修改指定行数范围内的 'true-label' 列的值
df.loc[142:1716, 'true-label'] = 1
df.loc[1812:3386, 'true-label'] = 1

# 保存修改后的CSV文件
df.to_csv(output_file, index=False)

