import numpy as np
# tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 生成样本数据
def generate_data(n_samples, time_steps, period):
    X = np.zeros((n_samples, time_steps))
    Y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        phase = np.random.rand() * 2 * np.pi
        for t in range(time_steps):
            X[i, t] = np.sin(2 * np.pi * t / period + phase)
        Y[i, 0] = period
    return X, Y

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
n_samples = 1000
time_steps = 100
period = 7.5
X_train, Y_train = generate_data(n_samples, time_steps, period)

# 重塑输入数据形状
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# 构建和训练模型
model = build_model(time_steps)
train_model(model, X_train, Y_train, epochs=25, batch_size=32)

# 使用模型预测周期
sample = X_train[0].reshape((1, time_steps, 1))
print(sample)
predicted_period = model.predict(sample)

print("Predicted period:", predicted_period[0][0])
