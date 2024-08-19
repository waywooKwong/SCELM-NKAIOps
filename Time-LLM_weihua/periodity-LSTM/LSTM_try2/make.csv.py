import pandas as pd
import math
import time

def generate_data(filename, num_rows):
    timestamps = []
    values = []
    timestamp = int(time.time())
    period = 132 # 周期
    for i in range(num_rows):
        timestamps.append(timestamp)
        value = math.sin(2 * math.pi * (i % period) / period) * 50  # 正弦函数
        values.append(value)
        timestamp += 1

    data = {'timestamp': timestamps, 'value': values}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    generate_data('/home/sunyongqian/liuheng/Time-LLM/weihua/LSTM_try2/data.csv', 500)  # 生成100行数据的CSV文件
