import pandas as pd
import numpy as np

# 读取CSV文件并提取数值数据
def read_csv(filename, a, b):
    df = pd.read_csv(filename)
    data = df['value'].values[:a*b]  # 仅保留前a*b个数值
    return data

# 将数值数据转换为二维矩阵
def convert_to_matrix(data, a, b):
    if len(data) % (a * b) != 0:
        raise ValueError("数据量不符合指定的维度要求")
    
    matrix = np.array(data).reshape((-1, b))
    return matrix

# 示例用法
filename = '/home/sunyongqian/liuheng/Time-LLM/weihua/res.csv'  # 替换为你的CSV文件路径
a = 30  # 指定矩阵的行数
b = 20  # 指定矩阵的列数
data = read_csv(filename, a, b)
matrix = convert_to_matrix(data, a, b)
print(matrix.shape)
