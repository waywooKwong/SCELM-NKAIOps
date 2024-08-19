from datetime import timedelta
import datetime, time
import logging
from aliyun.log import LogClient
from datetime import datetime, timedelta
import json
import sys
import time
from os.path import dirname
from SCWarn.Drain3.drain3 import TemplateMiner
from SCWarn.Drain3.drain3.template_miner_config import TemplateMinerConfig
from SCWarn.Drain3.drain3.file_persistence import FilePersistence
import pandas as pd
import os
import threading
from multiprocessing import Pool, cpu_count

# -*- coding: utf-8 -*-
#获取日志数据函数（根据新的阿里云日志SDK）
def get_history_data(project, logstore, start_time, query, minute, output_path):
    logger = logging.getLogger('log')
    ali_ak = ''# 阿里云日志的api
    ali_sk = ''
    client = LogClient("cn-zhangjiakou.log.aliyuncs.com", ali_ak,
                           ali_sk)
    start_time = start_time - timedelta(minutes=minute)
    end_time = start_time + timedelta(minutes=minute)

    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    file_date_path = os.path.join(output_path, end_time_str)
    os.makedirs(file_date_path, exist_ok=True)
    # it = client.pull_log_dump(project, logstore, from_time=start_time_str, to_time=end_time_str, file_path=file_date_path+"/dump_{}.data")
    resp = client.get_log(project, logstore, from_time=start_time_str, to_time=end_time_str, query=query, size=1000000).get_body()
    with open(file_date_path+"/data.data", "w") as f:
        json.dump(resp, f)


#最小时间窗口处理函数
def time_window_process(logstore,process_start_time, process_end_time, csv_file, query):
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)
    # 获取data文件夹下所有文件夹的路径
    if query == '* and _pod_name_: yid-develop-aisc-normal* and _container_name_: yid-develop-aisc-normal | select * from log limit 100000':
        data_folder_path = "/data/data/" + logstore + "/normal"
    else:
        data_folder_path = "/data/data/" + logstore + "/abnormal"
    data_folder = os.path.join(parent_dir, data_folder_path)
    folders = [os.path.join(data_folder, folder_name) for folder_name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder_name))]

    # 按文件夹名字中的时间进行排序
    sorted_folders = sorted(folders, key=lambda x: x.split('/')[-1])
    match_fail = []
    # 读取每个文件夹中的ndjson文件
    for time_folder in sorted_folders:
        start_time = os.path.basename(time_folder)
        start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if (process_start_time <= start_time_dt) and (start_time_dt<= process_end_time):
            template_df = pd.DataFrame()
            window_data_temp = []
            template_table = template_to_table(logstore)
            ndjson_files = [file for file in os.listdir(time_folder) if file.endswith(".data")]
            for ndjson_file in ndjson_files:
                file_path = os.path.join(time_folder, ndjson_file)
                data = json_process(file_path, logstore)
                window_data = []
                for item in data:
                    window_data.append(item['msg'])
                window_data_temp.append(window_data)
            template_process_df, match_fail_tmp = drain_match(window_data_temp, template_table, logstore)
            if match_fail_tmp:
                match_fail.append(match_fail_tmp)

            #将文件时间当作时刻
            template_process_df["timestamp"] = start_time
            template_df = update_record(template_process_df, template_df)

            window_data_temp.clear()
            csv_save_file = csv_file + '/' + start_time + '.csv'
            template_df.to_csv(csv_save_file, index=False)
        else:
            continue
    if match_fail:
        print(f"没匹配上的日志为：\n{match_fail}")

#日志预训练算法
def drain_train_log(logstore, query):
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)
    # 获取data文件夹下所有文件夹的路径
    if query == '* and _pod_name_: yid-develop-aisc-normal* and _container_name_: yid-develop-aisc-normal | select * from log limit 100000':
        data_folder_path = "/data/data/" + logstore + "/normal"
    else:
        data_folder_path = "/data/data/" + logstore + "/abnormal"
    data_folder = os.path.join(parent_dir, data_folder_path)
    folders = [os.path.join(data_folder, folder_name) for folder_name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder_name))]

    # 按文件夹名字中的时间进行排序
    sorted_folders = sorted(folders, key=lambda x: x.split('/')[-1])

    # 读取每个文件夹中的ndjson文件
    for folder in sorted_folders:
        ndjson_files = [file for file in os.listdir(folder) if file.endswith(".data")]
        for ndjson_file in ndjson_files:
            file_path = os.path.join(folder, ndjson_file)

            data = json_process(file_path, logstore)
            window_data = []
            for item in data:
                window_data.append(item['msg'])
            drain_train(window_data, logstore)
            window_data.clear()


def drain_train(input_data, logstore):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(f"{logstore}_drain3.bin")
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + f"/{logstore}_drain3.ini")
    config.profiling_enabled = False
    template_miner = TemplateMiner(persistence, config)

    lines = []

    for window in input_data:
        line = str(window).rstrip()
        lines.append(line)


    for line in lines:
        result = template_miner.add_log_message(line)

    print("Training done.")




#日志数据第一步预处理：筛选可用的数据
def json_process(json_file, logstore):
    # data = []
    data_return = []
    with open(json_file, "r") as f:
        if logstore == 'beyid':
            for line in f:
                item = json.loads(line)
                if 'level' in item and (item['level'] == 'debug' or item['level'] == 'info'):
                    continue
                elif 'msg' in item:
                    if ('fn' in item and item['fn'] !='null') or ('trace' in item and item['trace'] !='null') :
                        if item['level'] == 'error':
                            data_return.append(item)
                    else:
                        msg_data = item['msg']
                        if 'err=' in msg_data:
                            split_data = msg_data.split(',')
                            middle_values = split_data[1:3]
                            new_msg_data = ','.join(middle_values)
                            item['msg'] = new_msg_data
                        elif 'error=' in msg_data:
                            split_data = msg_data.split(',')
                            middle_values = split_data[1:3]
                            new_msg_data = ','.join(middle_values)
                            item['msg'] = new_msg_data
                        elif 'uri=/' in msg_data:
                            split_data = msg_data.split(',')
                            item['msg'] = split_data[2]
                        else:
                            split_data = msg_data.split(',')
                            if len(split_data) > 1:
                                item['msg'] = split_data[1]
                            else:
                                item['msg'] = split_data
                        data_return.append(item)
        elif logstore == 'ylint':
            for line in f:
                item = json.loads(line)
                if 'level' in item and (item['level'] == 'debug' or item['level'] == 'info' or item['level'] == 'warn'):
                    continue
                elif 'msg' in item:
                    msg_data = item['msg']
                    split_data = msg_data.split(',')
                    item['msg'] = split_data[0]
                    data_return.append(item)
        elif logstore == 'ymsg':
            for line in f:
                item1 = json.loads(line)
                item = json.loads(item1['content'])
                # print(item)
                if 'level' in item and (item['level'] == 'debug'):
                    continue
                elif 'msg' in item:
                    msg_data = item['msg']
                    split_data = msg_data.split(',')
                    item['msg'] = split_data[0]
                    data_return.append(item)
        elif logstore == 'ycard':
            for line in f:
                item = json.loads(line)
                if 'level' in item and (item['level'] == 'debug'):
                    continue
                elif 'msg' in item:
                    data_return.append(item)
        elif logstore == 'k8s-stdout':
            for line in f:
                item = json.loads(line)
                for item1 in item:
                    msg_data = item1['content']
                    split_data = msg_data.split('\t')
                    if 'DEBUG' in split_data:
                        continue
                    else:
                        if len(split_data) >=4:
                            split_data1 = split_data[3]
                            split_data2 = split_data1.split(',')
                            if len(split_data2) >=2:
                                item1['msg'] = split_data2[1]
                            else:
                                item1['msg'] = split_data2[0]
                        else:
                            # data_return.append(item1)
                            continue
                        data_return.append(item1)
    return data_return

#训练的模版转换成表格表头，用于匹配时往里面填充数据
def template_to_table(logstore):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(f"{logstore}_drain3.bin")
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + f"/{logstore}_drain3.ini")
    config.profiling_enabled = False
    template_miner = TemplateMiner(persistence, config)
    
    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

    headers= []
    for cluster_train in sorted_clusters:
        template_train = cluster_train.get_template()
        headers.append(template_train)
    df = pd.DataFrame(columns=headers)
    return df

#drain的匹配算法
def drain_match(input_data, template_df, logstore):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(f"{logstore}_drain3.bin")
    config = TemplateMinerConfig()
    config.load(dirname(__file__) + f"/{logstore}_drain3.ini")
    config.profiling_enabled = False
    template_miner = TemplateMiner(persistence, config)
    

    lines = []
    for window in input_data:
        for item in window:
            line = str(item).rstrip()
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


def get_all_data(project, logstore, start_time, end_time, minute, query):
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    if query == '* and _pod_name_: yid-develop-aisc-normal* and _container_name_: yid-develop-aisc-normal | select * from log limit 100000':
        path = "/data/data/" + logstore + "/normal"
    else:
        path = "/data/data/" + logstore + "/abnormal"
    output_path = os.path.join(parent_dir, path)
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    i=1
    while start_time <= end_time:
        print(f"第{i}次拿数据")
        get_history_data(project, logstore, start_time, query, minute, output_path)
        print(f"已获取{start_time}的数据")
        start_time += timedelta(minutes=minute)
        i = i + 1
    
    print("数据全部获取成功")


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
        # print(record_table)
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


def process_predict_data(train_path, predict_path, predict_save_path):
    df_a = pd.read_csv(train_path)
    df_b = pd.read_csv(predict_path)

    # 获取文件A的所有字段（除了"unknown"）
    fields_a = [field for field in df_a.columns if field != "unknown"]

    # 根据文件A的字段，筛选文件B的数据
    df_b_filtered = df_b[fields_a]

    # 对文件B中除了文件A的字段以外的数据进行相加
    df_b_filtered['unknown'] = df_b.drop(columns=fields_a).sum(axis=1)

    # 将筛选后的数据保存为新的CSV文件
    df_b_filtered.to_csv(predict_save_path, index=False)



#获取训练数据
def get_train_data(project, logstore, query, start_time, end_time, minute, id):
    s_time = time.time()
    get_all_data(project, logstore, start_time, end_time, minute, query)
    e_time = time.time()
    get_data_time = e_time - s_time
    print("数据获取时间为：", get_data_time, "秒")

    st_time = time.time()
    drain_train_log(logstore, query)
    en_time = time.time()
    preprocess_time = en_time - st_time
    print("日志预处理时间为：", preprocess_time, "秒")

    sta_time = time.time()

    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)

    middle_path = 'train_middle_data/' + logstore + '/' + start_time + '_' + end_time
    if logstore == 'beyid':
        relative_path = 'data/' + 'sc/yid/' + str(id)
    else:
        relative_path = 'data/' + 'sc/' + logstore + '/' + str(id)
    csv_file = os.path.join(parent_dir, middle_path)
    # 创建目录（包括中间缺失的父目录）
    os.makedirs(csv_file, exist_ok=True)  
    
    star_time = time.time()
    p = Pool(14)
    thread_start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    if_end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    for i in range(14):
        thread_start_time = thread_start_time + timedelta(hours=i*12)
        thread_end_time = thread_start_time + timedelta(hours=(i+1)*12)
        if thread_end_time <= if_end_time:
            p.apply_async(time_window_process, args=(logstore, thread_start_time, thread_end_time, csv_file, query))
        else:
            p.apply_async(time_window_process, args=(logstore, thread_start_time, if_end_time, csv_file, query))
    p.close()
    p.join()
    enb_time = time.time()
    win_process_time = enb_time-star_time

    output_path = os.path.join(parent_dir, relative_path)
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + '/' + 'train_log_middle.csv'
    train_data_path = output_path + '/' + 'train_log.csv'

    stara_time = time.time()
    csv_process(csv_file, output_file)
    process_train_data(output_file, train_data_path)
    enc_time = time.time()
    csv_process_time = enc_time - stara_time

    ena_time = time.time()
    process_time = ena_time - sta_time
    print("数据获取时间为：", get_data_time, "秒")
    print("日志预处理时间为：", preprocess_time, "秒")
    print("时间窗口处理时间为：",win_process_time, "秒")
    print("csv文件处理时间为：", csv_process_time, "秒")
    print("日志处理输出时序数据的时间为：", process_time, "秒")


#获取变更检测测试数据
def get_predict_datasc_data(project, logstore, query, start_time, end_time, minute):
    s_time = time.time()
    #get_all_data(project, logstore, start_time, end_time, minute, query)
    e_time = time.time()
    get_data_time = e_time - s_time
    print("数据获取时间为：", get_data_time, "秒")

    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)

    middle_path = 'predict_middle_data/datasc/'  + logstore + '/' + start_time + '_' + end_time
    relative_path = 'data/' + logstore + '/datasc'
    csv_file = os.path.join(parent_dir, middle_path)
    # 创建目录（包括中间缺失的父目录）
    os.makedirs(csv_file, exist_ok=True) 
    os.makedirs(relative_path, exist_ok=True)   

    star_time = time.time()
    p = Pool(4)
    thread_start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    for i in range(4):
        thread_start_time = thread_start_time + timedelta(hours=i*6)
        thread_end_time = thread_start_time + timedelta(hours=(i+1)*6)
        p.apply_async(time_window_process, args=(logstore, thread_start_time, thread_end_time, csv_file, query))
    p.close()
    p.join()
    enb_time = time.time()
    win_process_time = enb_time-star_time


    output_middle_path = os.path.join(parent_dir, relative_path)
    output_path = output_middle_path + '/' + start_time + '_' + end_time
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + '/' + 'predict_log_middle.csv'
    train_data_path = output_path + '/' + 'train_log.csv'
    predict_data_path = output_path + '/' + 'predict_log.csv'
    stara_time = time.time()
    csv_process(csv_file, output_file)
    process_predict_data(train_data_path, output_file, predict_data_path)
    enc_time = time.time()
    csv_process_time = enc_time - stara_time

    print("数据获取时间为：", get_data_time, "秒")
    print("时间窗口处理时间为：",win_process_time, "秒")
    print("csv文件处理时间为：", csv_process_time, "秒")

def get_predict_datadaily_data(project, logstore, query, start_time, end_time, minute, id):
    s_time = time.time()
    get_all_data(project, logstore, start_time, end_time, minute, query)
    e_time = time.time()
    get_data_time = e_time - s_time
    print("数据获取时间为：", get_data_time, "秒")

    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(__file__)
    # 回退到上一级目录
    parent_dir = os.path.dirname(current_dir)

    # 将日期时间字符串解析为 datetime 对象
    daily_test_time_obj = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    # 提取日期部分
    daily_test_time = daily_test_time_obj.strftime('%Y-%m-%d')

    middle_path = 'predict_middle_data/datadaily/'  + logstore + '/' + start_time + '_' + end_time
    if logstore == 'beyid':
        relative_path = 'data/daily/yid/' + daily_test_time
        train_relative_path = 'data/sc/yid/' + str(id)
    else:
        relative_path = 'data/daily/' + logstore + '/' + str(id)
        train_relative_path = 'data/sc/' + logstore + '/' + str(id)
    csv_file = os.path.join(parent_dir, middle_path)
    # 创建目录（包括中间缺失的父目录）
    os.makedirs(csv_file, exist_ok=True) 
    os.makedirs(relative_path, exist_ok=True)   

    star_time = time.time()
    p = Pool(4)
    thread_start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    if_end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    for i in range(4):
        thread_start_time = thread_start_time + timedelta(hours=i*6)
        thread_end_time = thread_start_time + timedelta(hours=(i+1)*6)
        if thread_end_time <= if_end_time:
            p.apply_async(time_window_process, args=(logstore, thread_start_time, thread_end_time, csv_file, query))
        else:
            p.apply_async(time_window_process, args=(logstore, thread_start_time, if_end_time, csv_file, query))
    p.close()
    p.join()
    enb_time = time.time()
    win_process_time = enb_time-star_time


    output_path = os.path.join(parent_dir, relative_path)
    train_path = os.path.join(parent_dir, train_relative_path)
    #print('output_path:', output_path)
    #print(' train_path',  train_path)

    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + '/' + 'test_log_middle.csv'
    train_data_path = train_path + '/' + 'train_log.csv'
    predict_data_path = output_path + '/' + 'test_log.csv'

    stara_time = time.time()
    csv_process(csv_file, output_file)
    process_predict_data(train_data_path, output_file, predict_data_path)
    enc_time = time.time()
    csv_process_time = enc_time - stara_time

    print("数据获取时间为：", get_data_time, "秒")
    print("时间窗口处理时间为：",win_process_time, "秒")
    print("csv文件处理时间为：", csv_process_time, "秒")
