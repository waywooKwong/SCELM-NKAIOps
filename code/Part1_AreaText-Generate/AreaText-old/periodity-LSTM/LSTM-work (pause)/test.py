import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 准备数据
data = np.array([5, 7, 9, 7, 5, 7, 9, 7, 5, 7, 9, 7])
look_back = 3
X, y = [], []
for i in range(len(data)-look_back):
    X.append(data[i:i+look_back])
    y.append(data[i+look_back])
X, y = np.array(X), np.array(y)

# 数据预处理
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X = X / np.max(X)

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X, y, epochs=100, batch_size=1, verbose=2)

# 测试模型
test_input = np.array([[7, 9, 7]])
test_input = np.reshape(test_input, (1, look_back, 1))
test_input = test_input / np.max(X)
predicted_output = model.predict(test_input)
print("预测的周期性数据:", predicted_output)
