import time
from datetime import datetime

def convert_to_date(time_data):
    def is_timestamp(val):
        try:
            # 检查是否为整数，并且长度为10的时间戳
            return len(str(val)) == 10 and int(val)
        except ValueError:
            return False

    def timestamp_to_date(timestamp):
        # 将时间戳转换为日期格式
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

    converted_data = []
    for item in time_data:
        if is_timestamp(item):
            converted_data.append(timestamp_to_date(item))
        else:
            # 假设其他格式的数据是正确的日期格式，可以直接添加
            converted_data.append(item)
    
    return converted_data

# 示例数据
time_data = [1700548200, '2024-07-14', 1623072987, '2023-12-25']

# 处理数据
converted_data = convert_to_date(time_data)
print(converted_data)
