import pandas as pd
from datetime import datetime, timedelta

def change_timestamps(start_time, csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 将时间戳列转换为datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算时间间隔
    interval = timedelta(minutes=2)
    
    # 更改时间戳
    new_timestamps = []
    current_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    for i in range(len(df)):
        new_timestamps.append(current_time)
        current_time += interval
    
    # 更新时间戳列
    df['timestamp'] = new_timestamps
    
    # 将修改后的数据保存回CSV文件
    df.to_csv('test_kpi.csv', index=False)

change_timestamps('2023-11-24 23:00:00','test_kpi_old.csv')

