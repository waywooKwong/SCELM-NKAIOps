import torch
from model import Model
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json

# 获取 test 和 train 中的 timestamp
from time_bounds import get_time_bounds

# 定义文件路径相关的变量
index_suffix = '/29811'
kind_suffix = '/bkverify'
base_path = '/home/sunyongqian/liuheng/aiops-scwarn/data'

# 获取时间边界
train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds(base_path, kind_suffix, index_suffix)

# 打印结果
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
folder_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811'  # 替换成你的文件夹路径
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

# 得到prefix 对应的 tran 名称
# 读取JSON文件
with open('/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_json/result_29811.json', 'r',
          encoding='utf-8') as f:
    data = json.load(f)
    
dim_info = data['metadata']['dim_info']
prefix_trans_map = {}

# 遍历prefix，获取对应的trans_name数组
for prefix in prefix_list:
    for dim_info_dict in dim_info:
        if prefix in dim_info_dict:
            trans_name = dim_info_dict[prefix]
            prefix_trans_map[prefix] = trans_name

with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
    file.write("领域文本：" + "\n")
    file.write("编号:NO.29811" + "\n")
    
    file.write("服务:E-TRAN" + "\n")
    file.write("提交开始时间:"+ train_start_time + "\n")
    file.write("提交结束时间:"+ train_end_time + "\n")
    file.write("分析开始时间:"+ test_start_time + "\n")
    file.write("分析结束时间:"+ test_end_time + "\n")
    file.write("与服务相关的指标变化分析：" + '\n')
    
for col_index in range(0, len(input_string)):
    csv_path = folder_path + '/' + prefix_list[col_index] +'_train_origin.csv'
    df = pd.read_csv(csv_path)
    input_data = df['origin_value']#按需求读取数据
    
    input_data = torch.tensor(input_data.values).view(1, len(input_data), 1) #转换成默认的形状
    prompt = model.forecast(input_data)
    with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
        file.write('    '+str(col_index) + ', ')
        file.write('指标训练集：'+prefix_trans_map[prefix_list[col_index]]+'; '+prefix_list[col_index]+';') # 源名称
        # file.write('指标训练集：'+input_string[col_index]+'; ') # 编码后名称
        
        for p in prompt:
            file.write(str(p))
        # if(col_index!=len(input_data)) :
        file.write('\n')
        
    csv_path = folder_path + '/' + prefix_list[col_index] +'.csv'
    df = pd.read_csv(csv_path)
    input_data = df['origin_value']#按需求读取数据
    
    input_data = torch.tensor(input_data.values).view(1, len(input_data), 1) #转换成默认的形状
    prompt = model.forecast(input_data)
    with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
        file.write('    '+str(col_index) + ', ')
        file.write('指标测试集：'+prefix_trans_map[prefix_list[col_index]]+'; '+prefix_list[col_index]+';') # name
        # file.write('指标测试集：'+input_string[col_index]+'; '+) # 编码后名称
        
        for p in prompt:
            file.write(str(p))
        # if(col_index!=len(input_data)) :
        file.write('\n')
        
        
print('single csv data finish in to output.txt')      

# 在文件末尾写入异常情况数量
with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
    file.write("以下是具体异常情况")
df = pd.read_csv('Time-LLM/weihua/CrossAttention-demo/combined.csv')
# 初始化异常情况计数器
exception_count = 1
with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
    file.write("异常情况：" + '\n')
    for idx, row in df.iterrows():
        csvname = row['csvname']
        timestamp = row['anomaly_timestamp']
        prediction = row['prediction']
        if csvname in prefix_trans_map:
            file.write("    {}. 指标: {}, 异常时间戳: {}, 异常模型: {}\n".format(exception_count,prefix_trans_map[csvname], timestamp, prediction))
            exception_count+=1

# 在文件末尾写入异常情况数量
with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output_read_folder.txt', 'a') as file:
    file.write("总共发现 {} 个异常情况。\n".format(exception_count))