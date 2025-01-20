import pandas as pd
import numpy as np
import random

# 读取CSV文件
df = pd.read_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/res2.csv')

# #将'value'列的值全部置为0
# df['value'] = 0

random_values = [random.random() for _ in range(len(df))]
df['value'] = random_values

# time_values = df['timestamp'].values
# sin_values = np.sin(2 * np.pi * time_values / 1000)
# df['value'] = sin_values

# 保存修改后的DataFrame到CSV文件
df.to_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/all-zero.csv', index=False)
