import json
import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv('/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811/overall.csv')

folder_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811'  # 替换成你的文件夹路径
prefix_set = set()  # 存储前缀的集合
for file_name in os.listdir(folder_path):  # 遍历文件夹中的文件
    if file_name.endswith('.csv'):  # 确保文件是以.csv结尾的
        prefix = file_name.split('_')[0]  # 分割文件名，获取前缀部分
        if prefix.endswith('.csv'):
            prefix_set.add(prefix[:-4])
        else:
            prefix_set.add(prefix)  # 否则，将前缀添加到集合中

prefix_list = list(prefix_set)
if "overall" in prefix_list:
    prefix_list.remove("overall")  # 如果存在，就删除它

# 读取JSON文件
with open('/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_json/result_29811.json', 'r',
          encoding='utf-8') as f:
    data = json.load(f)

# 获取dim_info的内容
dim_info = data['metadata']['dim_info']

# 创建字典存储prefix和trans_name的对应关系
prefix_trans_map = {}

# 遍历prefix，获取对应的trans_name数组
for prefix in prefix_list:
    for dim_info_dict in dim_info:
        if prefix in dim_info_dict:
            trans_name = dim_info_dict[prefix]
            prefix_trans_map[prefix] = trans_name
            print(f'{prefix}: {trans_name}')

# 输出存储的prefix和trans_name对应关系
# print("Prefix and corresponding trans_name:")
# for prefix, trans_name in prefix_trans_map.items():
#     print(f'{prefix}: {trans_name}')
print(prefix_trans_map)
