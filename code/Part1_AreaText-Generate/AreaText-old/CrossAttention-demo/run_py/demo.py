import torch
from model import Model
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json

# 获取 test 和 train 中的 timestamp
from time_bounds import get_time_bounds

# 定义基础路径和文件夹路径
# 定义文件路径相关的变量
index_suffix = '/29867'
kind_suffix = '/bkverify'

base_path = '/home/sunyongqian/liuheng/aiops-scwarn/'

result_folder = 'result_json_and_csv_29836' 
csv_folder = 'result_json_and_csv_29836/result_csv/29836'
json_folder = 'result_json'

# 定义各个文件的路径
folder_path = os.path.join(base_path, csv_folder) 
json_file_path = os.path.join(base_path, result_folder, json_folder, 'result_29836.json')
output_file_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/areatext_29836.txt'
overall_csv_path = os.path.join(folder_path, 'overall.csv')
combined_csv_path = '/home/sunyongqian/liuheng/Time-Series-Library-main/AIOps_dataset/fluxrankplus/src/fluxrankplus/model/combined29836.csv'

# 获取时间边界
train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds(base_path, kind_suffix, index_suffix)

# 配置模型
configs = {
    'description': 'Sample dataset',
    'pred_len': 10,
    'seq_len': 8,
    'top_k': 3,
    'enc_in': 4
}
model = Model(configs)

# 获取文件前缀集合
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

# 读取JSON文件并获取prefix对应的trans名称
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

dim_info = data['metadata']['dim_info']
prefix_trans_map = {}

for prefix in prefix_list:
    for dim_info_dict in dim_info:
        if prefix in dim_info_dict:
            trans_name = dim_info_dict[prefix]
            prefix_trans_map[prefix] = trans_name

# 初始化内容列表
content = []

# 写入初始内容
content.append("领域文本：\n")
content.append("编号:NO.29811\n")
content.append("服务:E-TRAN\n")
content.append("提交开始时间:" + train_start_time + "\n")
content.append("提交结束时间:" + train_end_time + "\n")
content.append("分析开始时间:" + test_start_time + "\n")
content.append("分析结束时间:" + test_end_time + "\n")
content.append("与服务相关的指标变化分析：\n")

# 处理每个prefix的CSV数据
for col_index in range(len(prefix_list)):
    prefix = prefix_list[col_index]

    # 处理 '_train_origin.csv'
    csv_train_origin_path = os.path.join(folder_path, f'{prefix}_train_origin.csv')
    if os.path.exists(csv_train_origin_path):
        df = pd.read_csv(csv_train_origin_path)
        input_data = df['origin_value'].values

        input_data = torch.tensor(input_data).view(1, len(input_data), 1)
        prompt = model.forecast(input_data)

        content.append(f'    {col_index + 1}. ')
        content.append(f'指标训练集：{prefix_trans_map.get(prefix, "未知")}；{prefix}；\n')  

        for p in prompt:
            content.append(str(p))
        content.append('\n')

    # 处理 '.csv'
    csv_path = os.path.join(folder_path, f'{prefix}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        input_data = df['origin_value'].values

        input_data = torch.tensor(input_data).view(1, len(input_data), 1)
        prompt = model.forecast(input_data)

        content.append(f'    {col_index + 1}. ')
        content.append(f'指标测试集：{prefix_trans_map.get(prefix, "未知")}；{prefix}；\n')  

        for p in prompt:
            content.append(str(p))
        content.append('\n')

# 写入SCWARN算法的异常时间戳
df = pd.read_csv(overall_csv_path)
fault_rows = df[df['model_label'] == 1]['timestamp']
result = fault_rows.to_string(index=False) if not fault_rows.empty else "no fault"

content.append("\nSCWARN 算法认为在下述时间戳有异常：\n")
result_lines = result.splitlines()
for idx, line in enumerate(result_lines, start=1):
    content.append(f"    {idx}, {line}\n")

# 读取combined.csv并处理异常信息
df = pd.read_csv(combined_csv_path)
anomaly_info = []
anomaly_count = 0

for csvname, group in df.groupby('csvname'):
    exception_count = len(group['anomaly_timestamp'].unique())
    pattern_count = len(group['prediction'].unique())

    anomaly_info.append({
        'header': f"\n    {exception_count}. 指标名称: {prefix_trans_map.get(csvname, '未知')}\n",
        'timestamps': group[['anomaly_timestamp', 'prediction']].values.tolist(),
        'summary': f"\n      该指标共出现 {pattern_count} 异常类型，类型为：{', '.join(map(str, group['prediction'].unique()))}\n"
    })

    anomaly_count += exception_count

# 写入具体异常指标的介绍
for info in anomaly_info:
    content.append(info['header'])
    content.append(info['summary'])

    for timestamp, prediction in info['timestamps']:
        content.append(f"      其中，类型[{type_dict.get(prediction, prediction)}]的时间戳是: {timestamp}")

# 写入文件末尾的文本信息
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

content.append(f"\n总共发现 {anomaly_count} 个异常情况\n")
content.append(f"出现异常的指标总数：{len(anomaly_info)}\n")
content.append(f"异常描述形状编号总数：{len(df['prediction'].unique())}\n\n")
content.append(text)
content.append("\n请分析如上领域文本以及指标异常情况,并给出你的判断:预期变更or失败变更\n")
content.append("并且给出你的思考和推理原因,如果是失败变更,请给出你的建议解决方法\n")
content.append("请按如下格式进行回答:{(\n")
content.append(" \"change_type\":\"请返回预期变更或是失败变更,\"\n")
content.append(" \"reason\":\"你确定change_type字段的原因\",\n")
content.append(" \"solution\":\"None if this is an expected change or solution\")}\n")

# 写入内容到文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.writelines(content)

print('写入完成')