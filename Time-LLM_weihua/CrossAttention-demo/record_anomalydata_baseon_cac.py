##### part1 record_anomalydata_baseon_cac.py

import pandas as pd
import os
import json

# 获得prefix名称
folder_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_30705/result_csv/30705'  # 替换成你的文件夹路径
csv_anomaly_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/csv_anomaly30705.csv'
anomaly_output_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/anomaly_details30705.csv'

prefix_set = set()  # 存储前缀的集合
for file_name in os.listdir(folder_path):  # 遍历文件夹中的文件
    if file_name.endswith('.csv'):  # 确保文件是以.csv结尾的
        prefix = file_name.split('_')[0]  # 分割文件名，获取前缀部分
        if prefix.endswith('.csv'):
            prefix_set.add(prefix[:-4])
        else:
            prefix_set.add(prefix)  # 否则，将前缀添加到集合中

prefix_list = list(prefix_set)  # 将集合转换为列表
print(f"Number of prefixes: {len(prefix_list)}")
if "overall" in prefix_list:
    prefix_list.remove("overall")  # 如果存在，就删除它

with open('/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_30705/result_json/result_30705.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dim_info = data['metadata']['dim_info']
prefix_trans_map = {}

# 遍历prefix，获取对应的trans_name数组
for prefix_1 in prefix_list:
    for dim_info_dict in dim_info:
        if prefix_1 in dim_info_dict:
            trans_name = dim_info_dict[prefix_1]
            prefix_trans_map[prefix_1] = trans_name

results = []
anomaly_results = []

# 初始化计数器
count_equal_1 = 0
count_not_equal_1 = 0
for idx, name in enumerate(prefix_list, start=0):
    csv_path = os.path.join(folder_path, name + '.csv')
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        continue
    
    if 'model_label' not in df.columns:
        print(f"model_label not in {csv_path}")
        continue

    X = df['model_label']

    if (X == 1).any():
        count_equal_1 += 1
        results.append([name, '出现异常', prefix_trans_map.get(name, '未知')])

        # 处理异常情况
        for index in X[X == 1].index:
            Y = df['timestamp'][index]
            Z_values = df['origin_value'].tolist()

            anomaly_row = [name, Y]

            # 获取前后各 15 个数据点，使用两个计数器交替
            forward_count = 0
            backward_count = 0
            Z_segment = []

            for i in range(1, 30):
                if backward_count <= forward_count and index - backward_count - 1 >= 0:
                    Z_segment.insert(0, Z_values[index - backward_count - 1])
                    backward_count += 1
                elif index + forward_count + 1 < len(Z_values):
                    Z_segment.append(Z_values[index + forward_count + 1])
                    forward_count += 1
                elif index - backward_count - 1 >= 0:
                    Z_segment.insert(0, Z_values[index - backward_count - 1])
                    backward_count += 1
                else:
                    Z_segment.append(None)

            # 添加中心点
            Z_segment.insert(15, Z_values[index])
            anomaly_row.extend(Z_segment)
            anomaly_results.append(anomaly_row)
    else:
        count_not_equal_1 += 1
        results.append([name, '正常', prefix_trans_map.get(name, '未知')])

# 将结果写入新的CSV文件
try:
    results_df = pd.DataFrame(results, columns=['数据集名称', '异常情况', '指标名称'])
    results_df.to_csv(csv_anomaly_path, index=False, encoding='utf-8')
    print(f"Info written into {csv_anomaly_path}")
except Exception as e:
    print(f"Failed to write results CSV: {e}")

# 写入包含异常详细信息的CSV文件
try:
    anomaly_columns = ['csvname', 'anomaly_timestamp'] + [f'Z_value_{i}' for i in range(1, 31)]
    anomaly_results_df = pd.DataFrame(anomaly_results, columns=anomaly_columns)
    anomaly_results_df.to_csv(anomaly_output_csv_path, index=False, encoding='utf-8')
    print(f"Info written into {anomaly_output_csv_path}")
except Exception as e:
    print(f"Failed to write anomaly details CSV: {e}")


##### part 2 from combine_datail_and_predict.py

# 读取两个CSV文件
df1 = pd.read_csv('Time-Series-Library-main/AIOps_dataset/fluxrankplus/src/fluxrankplus/model/predictions_30705.csv')
df2 = pd.read_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/anomaly_details30705.csv')

# 按列拼接
result = pd.concat([df2, df1], axis=1)

# 将结果保存到新的CSV文件中
result.to_csv('/home/sunyongqian/liuheng/Time-Series-Library-main/AIOps_dataset/fluxrankplus/src/fluxrankplus/model/combined30705.csv', index=False)