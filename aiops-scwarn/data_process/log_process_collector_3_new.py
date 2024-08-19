from datetime import timedelta
import datetime, time
import logging
from aliyun.log import LogClient
from datetime import datetime, timedelta
import json
import sys
import time
import shutil
from os.path import dirname
from SCWarn.Drain3.drain3 import TemplateMiner
from SCWarn.Drain3.drain3.template_miner_config import TemplateMinerConfig
from SCWarn.Drain3.drain3.file_persistence import FilePersistence
import pandas as pd
import os
import threading
from multiprocessing import Pool, cpu_count


def drain_train_log(train_data_path):
    data=json_process_train(train_data_path)
    window_data=[]
    for item in data:
        window_data.append(item[1])
    drain_train(window_data)
    window_data.clear()
    return data[0][0],data[-1][0]




def json_process_train(file_path):
    data_time=[]
    data_msg=[]
    with open(file_path, "r") as f:
        data=json.load(f)
    for key , lines in data.items():
        for line in lines:
            # print(line)
            devide_time_data=line.split(" INFO ")[0]
            devide_useful_data=line.split(" INFO ")[1].split(' ')[3]
            data_time.append(devide_time_data)
            data_msg.append(devide_useful_data)
    return_data = [[col1, col2] for col1, col2 in zip(data_time, data_msg)]
    return return_data


def drain_train(input_data):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(f"drain3.bin")
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + f"/drain3.ini")
    config.profiling_enabled = False
    template_miner = TemplateMiner(persistence, config)

    lines = []

    for window in input_data:
        line = str(window).rstrip()
        lines.append(line)


    for line in lines:
        result = template_miner.add_log_message(line)

    print("Training done.")    


def template_to_table(train_data_path):
    pass


def drain_match(input_data,template_df):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(f"drain3.bin")
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + f"/drain3.ini")
    config.profiling_enabled = False
    template_miner = TemplateMiner(persistence, config)

    lines=[]
    for each in input_data:
        line=str(each).rstrip()
        lines.append(line)
    match_fail = []
    template_count = []
    for line in lines:
        cluster = template_miner.match(line)
        if cluster is None:
            print(f"No match found, log:{line}")
            match_fail.append(line)
        else:
            template = cluster.get_template()
            template_count.append(template)

    field_counts = {field: 0 for field in template_df.columns}
    for template_batch in template_count:
        for field in template_df.columns:
            field_counts[field] += template_batch.count(field)
    template_df = template_df.append(field_counts, ignore_index=True)

    return template_df, match_fail


#更新记录模版随时间变化的表
def update_record(new_table, record):
    if record.empty:
        # 如果记录表为空，则直接将新表作为记录表
        return new_table
    
    new_data = new_table.to_dict(orient='records')
    record = pd.concat([record, pd.DataFrame(new_data)], ignore_index=True)

    return record


def time_window_process(train_data_path,csv_file,start_time,end_time,minutes):

    train_data=json_process_train(train_data_path)
    section_start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f ')
    section_end_time = section_start_time + timedelta(minutes=minutes)

    start_time_datatime = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f ')
    end_time_datatime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S.%f ')    

    while start_time_datatime <= section_end_time and section_end_time <= end_time_datatime:

        template_df = pd.DataFrame()
        # window_data_temp = []
        # template_table = template_to_table(train_data_path)
        template_table = pd.DataFrame()
        window_data = []
        match_fail = []

        for each in train_data:
            each_time_datatime = datetime.strptime(each[0], '%Y-%m-%d %H:%M:%S.%f ')  
            if section_start_time <= each_time_datatime and each_time_datatime <= section_end_time:
                window_data.append(each[1])
            else:
                continue
        
        template_process_df, match_fail_tmp = drain_match(window_data, template_table)
        if match_fail_tmp:
                match_fail.append(match_fail_tmp)

        #将文件时间当作时刻
        template_process_df["timestamp"] = section_start_time
        template_df = update_record(template_process_df, template_df)
        window_data.clear()
        section_start_time_str = section_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')
        csv_save_file = csv_file + '/' + section_start_time_str + '.csv'
        template_df.to_csv(csv_save_file, index=False)

        if match_fail:
            print(f"没匹配上的日志为：\n{match_fail}")

        section_start_time = section_end_time
        section_end_time = section_start_time + timedelta(minutes=minutes)




#处理中间文件夹中的csv文件合并成训练集或者测试集
def csv_process(folder_path, output_file):
    # 获取文件夹中所有CSV文件的路径和文件名
    files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            files.append((file_name, file_path))

    # 按照文件名排序
    files.sort(key=lambda x: x[0])

    # 初始化记录表
    record_table = pd.DataFrame()

    # 读取、处理文件并更新记录表
    for _, file_path in files:
        df = pd.read_csv(file_path)
        #print(df)
        print(record_table)
        record_table = update_record(df, record_table)
        record_table = record_table.fillna(0)

    # 根据 timestamp 字段排序记录表
    sorted_records = record_table.sort_values(by="timestamp")

    # 保存排序后的结果到新的CSV文件
    sorted_records.to_csv(output_file, index=False)

    print("处理完成，并保存为", output_file)



def process_train_data(train_data_path, train_save_path):
    data = pd.read_csv(train_data_path, index_col=0)
    
    # 计算每个字段的0比重
    zero_percentage = (data == 0).mean()
    
    # 找到比重大于80%的字段
    fields_to_sum = zero_percentage[zero_percentage > 0.8].index
    
    # 将字段的数据相加，添加到新的列"UNKNOWN"
    data['unknown'] = data[fields_to_sum].sum(axis=1)
    
    # 删除比重大于80%的字段
    filtered_data = data.drop(columns=fields_to_sum)
    
    # 保存处理后的数据
    filtered_data.to_csv(train_save_path)



def get_train_data(train_data_path):


    
    start_time_1=time.time()
    time_head,time_tail=drain_train_log(train_data_path)
    end_time_1=time.time()
    preprocess_time = end_time_1 - start_time_1
    print("日志预处理时间为：", preprocess_time, "秒")


    start_time=time.time()

    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)
    middle_path = 'train_middle_data/'
    relative_path = 'data/datasc'
    csv_file = os.path.join(parent_dir, middle_path)
    # 创建目录（包括中间缺失的父目录）
    os.makedirs(csv_file, exist_ok=True)  

    start_time_2=time.time()
    minutes=1
    time_window_process(train_data_path,csv_file,time_head,time_tail,minutes)
    end_time_2=time.time()
    win_process_time=end_time_2-start_time_2

    output_middle_path = os.path.join(parent_dir, relative_path)
    output_path = output_middle_path
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + '/' + 'train_log_middle.csv'
    train_data_path = output_path + '/' + 'train_log.csv'


    start_time_3=time.time()
    csv_process(csv_file, output_file)
    process_train_data(output_file, train_data_path)
    end_time_3=time.time()
    csv_process_time=end_time_3-start_time_3

    end_time=time.time()
    process_time=end_time-start_time

    print("日志预处理时间为：", preprocess_time, "秒")
    print("时间窗口处理时间为：",win_process_time, "秒")
    print("csv文件处理时间为：", csv_process_time, "秒")
    print("日志处理输出时序数据的时间为：", process_time, "秒")