import pandas as pd
import os
import json

# 获得prefix名称
folder_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811'  # 替换成你的文件夹路径
prefix_set = set()  # 存储前缀的集合
for file_name in os.listdir(folder_path):  # 遍历文件夹中的文件
    if file_name.endswith('.csv'):  # 确保文件是以.csv结尾的
        prefix = file_name.split('_')[0]  # 分割文件名，获取前缀部分
        if prefix.endswith('.csv'):
            prefix_set.add(prefix[:-4])
        else:
            prefix_set.add(prefix)  # 否则，将前缀添加到集合中

prefix_list = list(prefix_set)  # 将集合转换为列表
print(len(prefix_list))
if "overall" in prefix_list:
    prefix_list.remove("overall")  # 如果存在，就删除它

with open('/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_json/result_29811.json', 'r', encoding='utf-8') as f:
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

# 初始化计数器
count_equal_1 = 0
count_not_equal_1 = 0
for idx, name in enumerate(prefix_list, start=0):
    csv_path = os.path.join(folder_path, name + '.csv')
    df = pd.read_csv(csv_path)
    X = df['model_label']

    if (X == 1).any():
        count_equal_1 += 1
        results.append([name, '出现异常', prefix_trans_map.get(name, '未知')])
    else:
        count_not_equal_1 += 1
        results.append([name, '正常', prefix_trans_map.get(name, '未知')])

# 将结果写入新的CSV文件
output_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/csv_anomaly.csv'
results_df = pd.DataFrame(results, columns=['数据集名称', '异常情况', '指标名称'])
results_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print('Info written into csv_anomaly.csv')
