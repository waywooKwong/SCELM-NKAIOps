import pandas as pd

def get_time_bounds_true(base_path, kind_suffix, index_suffix):
    """
    获取训练和测试数据的开始和结束时间。

    参数:
    base_path (str): 数据文件的基础路径。
    kind_suffix (str): 数据文件的种类后缀。
    index_suffix (str): 数据文件的索引后缀。

    返回:
    tuple: 包含训练和测试数据的开始和结束时间的元组。
    """
    train_file_path = f"{base_path}/data/sc{kind_suffix}{index_suffix}/train_log.csv"
    test_file_path = f"{base_path}/data/daily{kind_suffix}{index_suffix}/test_log.csv"

    # 读取训练数据
    df_train = pd.read_csv(train_file_path)
    timestamp_train = df_train['timestamp']

    train_start_time = pd.to_datetime(timestamp_train.iloc[0], unit='s')
    train_end_time = pd.to_datetime(timestamp_train.iloc[-1], unit='s')

    # 读取测试数据
    df_test = pd.read_csv(test_file_path)
    timestamp_test = df_test['timestamp']
    test_start_time = pd.to_datetime(timestamp_test.iloc[0], unit='s')
    test_end_time = pd.to_datetime(timestamp_test.iloc[-1], unit='s')

    return train_start_time, train_end_time, test_start_time, test_end_time


def get_time_bounds_false(base_path, kind_suffix, index_suffix):
    """
    获取训练和测试数据的开始和结束时间。

    参数:
    base_path (str): 数据文件的基础路径。
    kind_suffix (str): 数据文件的种类后缀。
    index_suffix (str): 数据文件的索引后缀。

    返回:
    tuple: 包含训练和测试数据的开始和结束时间的元组。
    """
    train_file_path = f"{base_path}/data/sc{kind_suffix}{index_suffix}/train_log.csv"
    test_file_path = f"{base_path}/data/daily{kind_suffix}{index_suffix}/test_log.csv"

    # 读取训练数据
    df_train = pd.read_csv(train_file_path)
    timestamp_train = df_train['timestamp']
    train_start_time = timestamp_train.iloc[0]
    train_end_time = timestamp_train.iloc[-1]

    # 读取测试数据
    df_test = pd.read_csv(test_file_path)
    timestamp_test = df_test['timestamp']
    test_start_time = timestamp_test.iloc[0]
    test_end_time = timestamp_test.iloc[-1]

    return train_start_time, train_end_time, test_start_time, test_end_time

import pandas as pd

def get_time_bounds(base_path, kind_suffix, index_suffix, convert_to_datetime=True):
    """
    获取训练和测试数据的开始和结束时间。

    参数:
    base_path (str): 数据文件的基础路径。
    kind_suffix (str): 数据文件的种类后缀。
    index_suffix (str): 数据文件的索引后缀。
    convert_to_datetime (bool): 是否将时间戳转换为日期时间格式。默认值为True。

    返回:
    tuple: 包含训练和测试数据的开始和结束时间的元组。
    """
    train_file_path = f"{base_path}/data/sc{kind_suffix}{index_suffix}/train_log.csv"
    test_file_path = f"{base_path}/data/daily{kind_suffix}{index_suffix}/test_log.csv"

    # 读取训练数据
    df_train = pd.read_csv(train_file_path)
    timestamp_train = df_train['timestamp']

    # 清理无效数据
    timestamp_train = timestamp_train.dropna()
    timestamp_train = timestamp_train[timestamp_train.apply(lambda x: str(x).isdigit())]

    if convert_to_datetime:
        try:
            timestamp_train = timestamp_train.astype(int)
            train_start_time = pd.to_datetime(timestamp_train.iloc[0], unit='s')
            train_end_time = pd.to_datetime(timestamp_train.iloc[-1], unit='s')
        except Exception as e:
            print(f"Error converting train timestamps: {e}")
            return None, None, None, None
    else:
        train_start_time = timestamp_train.iloc[0]
        train_end_time = timestamp_train.iloc[-1]

    # 读取测试数据
    df_test = pd.read_csv(test_file_path)
    timestamp_test = df_test['timestamp']

    # 清理无效数据
    timestamp_test = timestamp_test.dropna()
    timestamp_test = timestamp_test[timestamp_test.apply(lambda x: str(x).isdigit())]

    if convert_to_datetime:
        try:
            timestamp_test = timestamp_test.astype(int)
            test_start_time = pd.to_datetime(timestamp_test.iloc[0], unit='s')
            test_end_time = pd.to_datetime(timestamp_test.iloc[-1], unit='s')
        except Exception as e:
            print(f"Error converting test timestamps: {e}")
            return None, None, None, None
    else:
        test_start_time = timestamp_test.iloc[0]
        test_end_time = timestamp_test.iloc[-1]

    return train_start_time, train_end_time, test_start_time, test_end_time


##   Origin Method
# index_suffix = '/29811'
# kind_suffix = '/bkverify'
# base_path = '/home/sunyongqian/liuheng/aiops-scwarn/data'

# # 获取时间边界
# train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds(base_path, kind_suffix, index_suffix)

# # 打印结果
# print(f"Train Start Time: {train_start_time}")
# print(f"Train End Time: {train_end_time}")
# print(f"Test Start Time: {test_start_time}")
# print(f"Test End Time: {test_end_time}")


def find_row_by_timestamp(csv_file_path, target_timestamp):
    """
    根据指定的timestamp查找CSV文件中对应的行号。

    参数:
    csv_file_path (str): CSV文件的路径。
    target_timestamp (str): 要查找的目标timestamp。

    返回:
    int: 对应的行号。如果未找到，返回-1。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 检查'timestamp'列是否存在
    if 'timestamp' not in df.columns:
        raise ValueError("CSV文件中没有'timestamp'列")

    # 查找目标timestamp所在的行号
    try:
        row_number = df.index[df['timestamp'] == target_timestamp].tolist()
        if len(row_number) == 0:
            return -1  # 未找到目标timestamp
        return row_number[0]
    except KeyError:
        return -1  # 未找到目标timestamp

# # 示例用法
# csv_file_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811/00a3dbac84ee086719f2b891079df682_train_origin.csv'
# target_timestamp = test_start_time

# row_number = find_row_by_timestamp(csv_file_path, target_timestamp)
# if row_number != -1:
#     print(f"目标timestamp所在的行号是: {row_number}")
# else:
#     print("未找到目标timestamp")