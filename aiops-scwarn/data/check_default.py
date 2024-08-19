import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/sunyongqian/liuheng/aiops-scwarn/data/daily/geass/32266/test_log_middle.csv')
# 找出每列中的缺省值数量
missing_values = df.isnull().sum()

# 过滤掉没有缺省值的列，只打印包含缺省值的列
missing_values = missing_values[missing_values > 0]

print(missing_values)
