import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

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

# 生成样本数据
def generate_data(n_samples, period):
    Y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        Y[i, 0] = period
    return Y

# LSTM模型
def build_model(time_steps):
    model = Sequential()
    model.add(LSTM(50, input_shape=(time_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 训练模型
def train_model(model, X_train, Y_train, epochs, batch_size):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# 生成数据
n_samples = 30
time_steps = 20
period = 30
Y_train = generate_data(n_samples, period)

filename = '/home/sunyongqian/liuheng/Time-LLM/weihua/all-zero.csv'  # 替换为你的CSV文件路径
data = read_csv(filename, n_samples, time_steps)
X_train = convert_to_matrix(data, n_samples, time_steps)

# 重塑输入数据形状
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
sample = X_train[0].reshape((1, time_steps, 1))

# 构建和训练模型
model = build_model(time_steps)
train_model(model, X_train, Y_train, epochs=50, batch_size=64)

# 使用模型预测周期
sample = X_train[0].reshape((1, time_steps, 1))
predicted_period = model.predict(sample)
print("Predicted period:", predicted_period[0][0])


'''
保存X_train数据
# 将DataFrame展平为一维Series
X_train_flat = X_train.flatten()
# 生成时间戳列表，假设一个数据对应一个时间戳
start_timestamp = 1692548400
timestamps = [start_timestamp + i * 100 for i in range(len(X_train_flat))]
# 创建包含展开数据和时间戳的DataFrame
data = {'timestamp': timestamps,'value': X_train_flat}
df = pd.DataFrame(data)
# 将DataFrame保存到CSV文件
df.to_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/X_train.csv', index=False)
'''