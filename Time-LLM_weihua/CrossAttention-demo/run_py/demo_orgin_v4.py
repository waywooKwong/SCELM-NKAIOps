import torch
from model import Model
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json
from time_bounds import get_time_bounds_false,get_time_bounds_true,get_time_bounds
# sum_weihua_part123.py 与 demo_orgin_v2.py
index = '29811' 
kind_suffix = '/bkverify'
base_path = '/home/sunyongqian/liuheng/aiops-scwarn'

text_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/areaText_v4/areaText_'+index+'_v4.txt'
overall_csv_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_'+index+'/result_csv/'+index+'/overall.csv'

result_csv_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_'+index+'/result_csv/'+index
json_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_'+index+'/result_json/result_'+index+'.json'
combine_csv_path = '/home/sunyongqian/liuheng/Time-Series-Library-main/AIOps_dataset/fluxrankplus/src/fluxrankplus/model/combined'+index+'.csv'

# 获取时间边界
index_suffix = '/'+ index
train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds_false(base_path, kind_suffix, index_suffix)
# 将时间转换为字符串格式
# train_start_time_str = str(train_start_time).strftime('%Y-%m-%d %H:%M:%S')
# train_end_time_str = str(train_end_time).strftime('%Y-%m-%d %H:%M:%S')
# test_start_time_str = str(test_start_time).strftime('%Y-%m-%d %H:%M:%S')
# test_end_time_str = str(test_end_time).strftime('%Y-%m-%d %H:%M:%S')

# 打印结果
print("index:",index,' ',"kind_suffix:",kind_suffix)
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

# 获得prefix名称
folder_path = result_csv_path  # 替换成你的文件夹路径
prefix_set = set() # 存储前缀的集合
for file_name in os.listdir(folder_path):# 遍历文件夹中的文件
    if file_name.endswith('.csv'): # 确保文件是以.csv结尾的
        prefix = file_name.split('_')[0]# 分割文件名，获取前缀部分
        if prefix.endswith('.csv'):
            prefix_set.add(prefix[:-4])
        else:
            prefix_set.add(prefix)# 否则，将前缀添加到集合中

prefix_list = list(prefix_set)# 将集合转换为列表
if "overall" in prefix_list:
    prefix_list.remove("overall")    # 如果存在，就删除它
input_string = prefix_list

# 类型映射字典
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

# 得到prefix 对应的 tran 名称
# 读取JSON文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

dim_info = data['metadata']['dim_info']
prefix_trans_map = {}

# 遍历prefix，获取对应的trans_name数组
for prefix in prefix_list:
    for dim_info_dict in dim_info:
        if prefix in dim_info_dict:
            trans_name = dim_info_dict[prefix]
            prefix_trans_map[prefix] = trans_name

# 初始化内容列表
content = []
# 写入初始内容
content.append("异常变更领域文本：\n")
content.append("编号:NO.{} {}\n".format(index,kind_suffix))
content.append("服务:E-TRAN\n")
content.append("提交开始时间:" + str(train_start_time) + "\n")
content.append("提交结束时间:" + str(train_end_time) + "\n")
content.append("分析开始时间:" + str(test_start_time) + "\n")
content.append("分析结束时间:" + str(test_end_time) + "\n")
content.append("与服务相关的指标变化分析：\n")

for col_index in range(0, len(input_string)):
    csv_path = folder_path + '/' + prefix_list[col_index] + '_train_origin.csv'
    df = pd.read_csv(csv_path)
    input_data = df['origin_value']  # 按需求读取数据

    input_data = torch.tensor(input_data.values).view(1, len(input_data), 1)  # 转换成默认的形状
    content.append('    ' + str(col_index + 1) + '.指标名称: '+ prefix_trans_map[prefix_list[col_index]])
    prompt, num_before = model.forecast(input_data)
    # print('num_before',num_before[0][0])
    content.append('\n        变更发生前')  # 源名称
    for p in prompt:
        content.append(str(p))
    csv_path = folder_path + '/' + prefix_list[col_index] + '.csv'
    df = pd.read_csv(csv_path)
    input_data = df['origin_value']  # 按需求读取数据

    input_data = torch.tensor(input_data.values).view(1, len(input_data), 1)  # 转换成默认的形状
    prompt, num_after = model.forecast(input_data)
    # print('num_after',num_after[0][1])
    content.append('\n        变更发生后')  # name
    for p in prompt:
        content.append(str(p))
    content.append('\n        变更前后数据范围对比：变更前范围：[{},{}], 变更后范围：[{},{}]\n'.format(num_before[0][0], num_before[0][1], num_after[0][0], num_after[0][1]))

print('single csv data finish in to output.txt')

df = pd.read_csv(overall_csv_path)
fault_rows = df[df['model_label'] == 1]['timestamp']
if not fault_rows.empty:
    result = fault_rows.to_string(index=False)
else:
    result = "no fault"

content.append("\nSCWARN 算法认为在下述时间戳有异常：\n")
result_lines = result.splitlines()
for idx, line in enumerate(result_lines, start=0):
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

content.append("\n发生异常的指标中图形对应的异常类型:")
current_csvname = None

# 用于存储每个指标的异常信息
anomaly_info = []

for idx, row in df.iterrows():
    anomaly_count += 1
    csvname = row['csvname']
    timestamp = row['anomaly_timestamp']
    prediction = row['prediction']

    # 使用映射字典将数字转换为英文名词
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

# 如果还有未写入的最后一个指标的异常信息
if current_csvname is not None:
    anomaly_info[-1]['summary'] = "\n      该指标共出现 {} 异常类型，类型为：{}".format(pattern_count, pattern_list)

# 写入具体异常指标的介绍
for info in anomaly_info:
    # content.append(info['header'])
    # 增加每个异常描述图形编号对应的时间戳
    pattern_timestamp_map = {}
    for timestamp, pattern in info['timestamps']:
        if pattern not in pattern_timestamp_map:
            pattern_timestamp_map[pattern] = []
        pattern_timestamp_map[pattern].append(timestamp)
    # if 'summary' in info:
    #                 content.append(info['summary'])
    # v2 的写法
    # for pattern, timestamps in pattern_timestamp_map.items():
    #     content.append("      其中，类型[{}]的时间戳是: {}".format(pattern, ', '.join(map(str, timestamps))))
    for pattern, timestamps in pattern_timestamp_map.items():
        if pattern in {type_dict[7], type_dict[8], type_dict[9], type_dict[10], type_dict[11], type_dict[12], type_dict[13]}:  # Check for patterns 7 to 13
            overlapping_timestamps = [ts for ts in timestamps if str(ts) in result]
            if overlapping_timestamps:
                # 添加 summary
                content.append(info['header'])
                if 'summary' in info:
                    content.append(info['summary'])       
                content.append("\n      其中，类型[{}]恢复正常，与整体同时出现异常的时间戳: {}".format(pattern, ', '.join(map(str, overlapping_timestamps))))
        else:
            # Print all timestamps for patterns 1 to 6
               # 添加 summary
            content.append(info['header'])
            if 'summary' in info:
                content.append(info['summary'])
            content.append("\n      其中，类型[{}]持续异常，时间戳是: {}".format(pattern, ', '.join(map(str, timestamps))))


# 将内容写入文件
with open(text_path, 'a') as file:
    file.writelines(content)


text = """这是对异常类型(pattern)的定义：
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

# 在文件末尾写入异常情况数量
with open(text_path, 'a') as file:
    file.write("\n总共发现 {} 个异常情况\n".format(anomaly_count-1))
    file.write("出现异常的指标总数：{}\n".format(exception_count - 1))
    file.write("异常描述形状编号总数：{}\n\n".format(unique_count))
    file.write(text)    
    file.write("\n请分析如上领域文本以及指标异常情况,并给出你的判断:预期变更or失败变更"+"\n")
    file.write("并且给出你的思考和推理原因,如果是失败变更,请给出你的建议解决方法"+"\n")
    file.write("请按如下格式进行回答:{("+"\n")
    file.write(" \"change_type\":\"请返回预期变更或是失败变更,\""+"\n")
    file.write(" \"reason\":\"你确定change_type字段的原因\","+"\n")
    file.write(" \"solution\":\"None if this is an expected change or solution\")}"+"\n")

print('overall csv data finish in to output.txt')
print(len(prefix_list))
