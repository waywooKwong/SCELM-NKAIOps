import torch
from model import Model
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, date
import json
from time_bounds import get_time_bounds_false,get_time_bounds_true,get_time_bounds

import time
from datetime import datetime

import time
from datetime import datetime


def convert_to_date(time_data):
    def is_timestamp(val):
        try:
            # 检查是否为10位的时间戳，无论是字符串还是整数
            val_str = str(val)
            return val_str.isdigit() and len(val_str) == 10
        except ValueError:
            return False

    def timestamp_to_date(timestamp):
        # 将时间戳转换为日期格式
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

    if isinstance(time_data, list):
        converted_data = []
        for item in time_data:
            if is_timestamp(item):
                converted_data.append(timestamp_to_date(item))
            else:
                # 假设其他格式的数据是正确的日期格式，可以直接添加
                converted_data.append(item)
        return converted_data
    else:
        # 如果是单个值
        if is_timestamp(time_data):
            return timestamp_to_date(time_data)
        else:
            return time_data
# # 处理数据
# converted_data = convert_to_date()
# print(converted_data)


def AreaText_Part2(target_index, suffix):
    index = target_index
    kind_suffix = suffix
    base_path = '/home/sunyongqian/liuheng/aiops-scwarn'
    text_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/areaText_v5/chinese/areaText_' + index + '_v4.txt'
    overall_csv_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv/result_json_and_csv_' + index + '/result_csv/' + index + '/overall.csv'
    result_csv_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv/result_json_and_csv_' + index + '/result_csv/' + index
    json_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv/result_json_and_csv_' + index + '/result_json/result_' + index + '.json'
    combine_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k4_combine/combined_' + index + '.csv'

    # 获取时间边界
    index_suffix = '/' + index
    train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds_false(base_path, kind_suffix, index_suffix)

    print("index:", index, ' ', "kind_suffix:", kind_suffix)
    print(f"Train Start Time: {train_start_time}")
    print(f"Train End Time: {train_end_time}")
    print(f"Test Start Time: {test_start_time}")
    print(f"Test End Time: {test_end_time}")

    configs = {
        'description': 'Sample dataset',
        'pred_len': 10,
        'seq_len': 8,
        'top_k': 3,
        'enc_in': 4
    }
    model = Model(configs)

    folder_path = result_csv_path  # 替换成你的文件夹路径
    prefix_set = set()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            prefix = file_name.split('_')[0]
            if prefix.endswith('.csv'):
                prefix_set.add(prefix[:-4])
            else:
                prefix_set.add(prefix)

    prefix_list = list(prefix_set)
    if "overall" in prefix_list:
        prefix_list.remove("overall")
    input_string = prefix_list

    type_dict = {
        1: "Sudden increase",
        2: "Sudden decrease",
        3: "Level shift up",
        4: "Level shift down",
        5: "Steady increase",
        6: "Steady decrease",
        7: "Single spike",
        8: "Single dip",
        9: "Transient level shift up",
        10: "Transient level shift down",
        11: "Multiple spikes",
        12: "Multiple dips",
        13: "Fluctuations"
    }

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dim_info = data['metadata']['dim_info']
    prefix_trans_map = {}

    for prefix in prefix_list:
        for dim_info_dict in dim_info:
            if prefix in dim_info_dict:
                trans_name = dim_info_dict[prefix]
                prefix_trans_map[prefix] = trans_name

    content = []
    content.append("异常变更领域文本：\n")
    content.append("编号:NO.{} \n".format(index))
    kind_suffix_0 = kind_suffix.replace("/", "")
    content.append("服务:{}\n".format(kind_suffix_0))
    content.append("提交开始时间:" + train_start_time + "\n")
    content.append("提交结束时间:" + train_end_time + "\n")
    content.append("分析开始时间:" + test_start_time + "\n")
    content.append("分析结束时间:" + test_end_time + "\n")
    content.append("与服务相关的指标变化分析：")

    df = pd.read_csv(overall_csv_path)
    fault_rows = df[df['model_label'] == 1]['timestamp']
    if not fault_rows.empty:
        result = fault_rows.to_string(index=False)
    else:
        result = "no fault"

    content.append("\n\nSCWARN 算法认为在下述时间戳有异常：\n")
    result_lines = result.splitlines()
    for idx, line in enumerate(result_lines, start=0):
        line = convert_to_date(line)
        content.append(f"    {idx + 1}, {line}\n")

    df = pd.read_csv(combine_csv_path)
    pattern_data = df['prediction']
    unique_values = pattern_data.unique()
    unique_count = len(unique_values)

    exception_count = 1
    anomaly_count = 0
    timestamp_count = 0
    pattern_count = 0
    pattern_list = []
    
    text = """\n这是对异常类型(pattern)的定义：
    异常描述形状分为两大类：Still in abnormal state和Recover to normal state,
    Still in abnormal state,这类异常在出现异常点后持续处于异常状态
    1. Sudden increase 突增
    2. Sudden decrease 突降
    3. Level shift up 整体层次上升
    4. Level shift down 整体层次下降
    5. Steady increase 持续上升
    6. Steady decrease 持续下降
    Recover to normal state,这类异常在出现异常点后回归正常状态
    7. Single spike 单一高峰
    8. Single dip 单一低谷
    9. Transient level shift up 瞬间整体层次上升
    10. Transient level shift down 瞬间整体层次下降
    11. Multiple spikes 连续多峰
    12. Multiple dips 连续多谷
    13. Fluctuations 持续波动\n"""
    # with open(text_path, 'a') as file:
    #     file.write(text)    
    content.append(text)

    content.append("\n与整体异常相关的单指标异常图形类型(与整体异常无关的单指标异常未输出):")
    current_csvname = None

    anomaly_info = []
    for idx, row in df.iterrows():
        anomaly_count += 1
        csvname = row['csvname']
        timestamp = convert_to_date(row['anomaly_timestamp'])
        prediction = row['prediction']
        prediction_type = type_dict.get(prediction, prediction)

        if csvname == current_csvname:
            anomaly_info[-1]['timestamps'].append((timestamp, prediction_type))
            timestamp_count += 1
            if prediction_type not in pattern_list:
                pattern_list.append(prediction_type)
                pattern_count += 1
        else:
            if current_csvname is not None:
                anomaly_info[-1]['summary'] = "\n      该指标共出现 {} 异常类型，类型为：{}".format(pattern_count, pattern_list)
                timestamp_count = 0
                pattern_count = 0
                pattern_list = []

            current_csvname = csvname
            if csvname in prefix_trans_map:
                anomaly_info.append({
                    'header': "\n    {}.指标名称: {}".format(exception_count, prefix_trans_map[csvname]),
                    'timestamps': [(timestamp, prediction_type)]
                })
                exception_count += 1
                timestamp_count = 1
                if prediction_type not in pattern_list:
                    pattern_list.append(prediction_type)
                    pattern_count = 1
                else:
                    pattern_count = 0

    if current_csvname is not None:
        anomaly_info[-1]['summary'] = "\n      该指标共出现 {} 异常类型，类型为：{}".format(pattern_count, pattern_list)

    for info in anomaly_info:
        pattern_timestamp_map = {}
        for timestamp, pattern in info['timestamps']:
            if pattern not in pattern_timestamp_map:
                pattern_timestamp_map[pattern] = []
            pattern_timestamp_map[pattern].append(timestamp)

        for pattern, timestamps in pattern_timestamp_map.items():
            timestamps = convert_to_date(timestamps)
            if pattern in {type_dict[7], type_dict[8], type_dict[9], type_dict[10], type_dict[11], type_dict[12], type_dict[13]}:
                overlapping_timestamps = [ts for ts in timestamps if str(ts) in result]

                if overlapping_timestamps:
                    content.append(info['header'])
                    if 'summary' in info:
                        content.append(info['summary'])    
                    overlapping_timestamps = convert_to_date(overlapping_timestamps)
                    content.append("\n      其中，类型[{}] 恢复正常，与整体同时出现异常的时间戳: {}".format(pattern, ', '.join(map(str, overlapping_timestamps))))
            else:
                content.append(info['header'])
                if 'summary' in info:
                    content.append(info['summary'])    
                timestamps = convert_to_date(timestamps)
                content.append("\n      其中，类型[{}] 持续异常，时间戳是: {}".format(pattern, ', '.join(map(str, timestamps))))

    for col_index in range(0, len(input_string)):
        csv_path = folder_path + '/' + prefix_list[col_index] + '_train_origin.csv'
        df = pd.read_csv(csv_path)
        input_data = df['origin_value']  

        input_data = torch.tensor(input_data.values).view(1, len(input_data), 1)
        content.append('    ' + str(col_index + 1) + '.指标名称: '+ prefix_trans_map[prefix_list[col_index]])
        prompt, num_before = model.forecast(input_data)
        content.append('\n        变更发生前')  
        for p in prompt:
            content.append(str(p))
        csv_path = folder_path + '/' + prefix_list[col_index] + '.csv'
        df = pd.read_csv(csv_path)
        input_data = df['origin_value']  

        input_data = torch.tensor(input_data.values).view(1, len(input_data), 1)
        prompt, num_after = model.forecast(input_data)
        content.append('\n        变更发生后')  
        for p in prompt:
            content.append(str(p))
        content.append('\n        变更前后数据范围对比：变更前范围：[{},{}], 变更后范围：[{},{}]\n'.format(num_before[0][0], num_before[0][1], num_after[0][0], num_after[0][1]))

    print('single csv data finish in to output.txt')
    
    with open(text_path, 'a') as file:
        file.writelines(content)

    with open(text_path, 'a') as file:
        file.write("总共发现 {} 个异常情况\n".format(anomaly_count-1))
        file.write("出现异常的指标总数：{}\n".format(exception_count - 1))
        file.write("异常描述形状编号总数：{}\n\n".format(unique_count))

    print('overall csv data finish in to output.txt')
    print(len(prefix_list))
