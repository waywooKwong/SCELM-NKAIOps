from data_process.collector import*
from data_process.data_collector import*
from data_process.log_process_collector_3_k8s_yid import *
from module_yunzhanghu import *
from software_change_event.get_software_change_event import *
import click
import os
import numpy as np
import pandas
import json
import datetime
import time
import shutil
import data_process.log_process_collector_3_k8s_yid as log_process
from datetime import datetime, timedelta
from plot_data import *

class K8s_params:
    INSTANCE_NAME = "yid-develop-aisc-normal-6b888bb79-w7hcc"
    NODE_NAME = "10.200.67.42"
    POD_NAME = INSTANCE_NAME

class Log_params:
    project = 'yzh-cce-rc-log'
    logstore = 'k8s-stdout'
    minute = 2

# 跟据train_date计算训练的开始时间和检测的结束时间
def transfer_time(train_date, duration):
    time_stamp = int(time.mktime(time.strptime(train_date, "%Y-%m-%d %H:%M:%S")))

    if duration < 1200:
        # 减去一个整数得到另一个时间戳
        new_time_stamp = time_stamp + duration  
    else:
        new_time_stamp = time_stamp - duration

    # 将时间戳转为日期字符串
    new_date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(new_time_stamp))

    return new_date_str

#获取train_log
def get_train_log(train_date, train_duration, str_id):
    print("获取训练日志")
    start_train_time = transfer_time(train_date, train_duration)
    train_query = "* and _pod_name_: yid-develop-aisc-normal* and _container_name_: yid-develop-aisc-normal | select * from log limit 100000"
    print(start_train_time, train_date)
    log_process.get_train_data(Log_params.project, Log_params.logstore, train_query, start_train_time, train_date, Log_params.minute, str_id)

#获取test_log
def get_test_log(train_date, detection_duration, str_id):
    print("获取测试日志")
    end_test_time = transfer_time(train_date, detection_duration)
    #test_query =  "* and _pod_name_: yid-develop-aiops-abnormal* and _container_name_: yid-develop-aiops-abnormal | select * from log limit 100000"
    test_query = "* and _pod_name_: yid-develop-aisc-normal* and _container_name_: yid-develop-aisc-normal | select * from log limit 100000"
    log_process.get_predict_datadaily_data(Log_params.project, Log_params.logstore, test_query, train_date, end_test_time, Log_params.minute, str_id)

def pinjie_data(kpi_origin, log_origin):
    with open('datalog/{}.csv'.format(log_origin), 'r') as file:
        csv_reader = csv.reader(file)
        # next(csv_reader)
        count = 0
        for line in csv_reader:
            # 将 line[-1] 添加到 line[:-2] 构建新的列表
            new_line = line[:-2]
            new_line.append(line[-1])

            # 然后将新列表连接到 kpi_origin[count]
            kpi_origin[count] = kpi_origin[count] + new_line

            count = count + 1

    return kpi_origin

def montage_kpi(train_data_path):
    data_list1 = []
    with open(train_data_path + 'train_kpi.csv', 'r', newline='') as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            data_list1.append(row)

    # 读取第二个 CSV 文件
    data_list2 = []
    with open(train_data_path + 'train_log.csv', newline='') as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
            data_list2.append(row[:-2]+[row[-1]])

    kpi_origin = []
    for l1, l2 in zip(data_list1, data_list2):
        kpi_origin.append(l1+l2)
    # 合并两个列表
    return kpi_origin

def montage_kpi_test(test_data_path):
    data_list1 = []
    with open(test_data_path + 'test_kpi.csv', 'r', newline='') as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            data_list1.append(row)

    # 读取第二个 CSV 文件
    data_list2 = []
    with open(test_data_path + 'test_log.csv', newline='') as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
            data_list2.append(row[:-2]+[row[-1]])

    print(data_list2)
    kpi_origin = []
    for l1, l2 in zip(data_list1, data_list2):
        kpi_origin.append(l1+l2)
    # 合并两个列表

    return kpi_origin

def k8snet_process(service, sc_id, train_date):
    with open('./software_change_event/service_promql/yid_k8s/yid_k8s.json', 'r') as file:
        json_data = json.load(file)
    json_data['service'] = service
    json_data['id'] = sc_id
    json_data['train_end_date'] = train_date
    json_data['sc_end_date'] = train_date
    json_data['promql'] = []
    json_data['machine_bussniess'] = []
    k8s_line_count = 0
    SLA_line_count = 0
    # 读取k8s文本文件
    with open('./software_change_event/service_promql/yid_k8s/machine.txt', 'r') as file1:
        for line in file1:
            json_data['promql'].append(line.strip().format(K8s_params.INSTANCE_NAME, K8s_params.NODE_NAME))
            k8s_line_count += 1
        json_data['machine_bussniess'].append(k8s_line_count)

    # 读取SLA文本文件
    with open('./software_change_event/service_promql/yid_k8s/SLA.txt', 'r') as file2:
        lines = file2.readlines()
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                json_data['promql'].append(line.strip().format(K8s_params.POD_NAME))
            else:
                json_data['promql'].append(line.strip().format(K8s_params.POD_NAME))
            SLA_line_count += 1
        json_data['machine_bussniess'].append(SLA_line_count)
    return json_data

class log_params:
    project = 'yzh-log-service'
    minute = 2

# 命令行执行
# python3 -m run --publish_date '20230817' --prometheus_address '172.16.17.252:19192' --service 'ylint' --task_count 5 --step 120 --timeout 10000 --train_duration 604800 --detection_duration 86400 --predict_interval '30s'

# 输入变更日期返回所需要的变更事件
@click.command()
@click.option('--publish_date', '-pd', help='publish date', default='20230330', type=str)
@click.option('--prometheus_address', '-p', help='prometheus address', default='127.0.0.1:9091')
@click.option('--service', '-s', help='service', default='yid')
@click.option('--sc_id', '-c', help='sc_id', default=None)
@click.option('--train_date', '-t', help='the end time of train_time', default='2023-10-23 12:00:00')
@click.option('--task_count', '-task_count', help='prometheus metrics to query', type=int,default=5)
@click.option('--step', '-step', help='time interval', default=60, type=int)
@click.option('--timeout', '-timeout', help='train end time', default=30, type=int)
@click.option('--train_duration', '-td', help='train duration', default=259200, type=int)
@click.option('--detection_duration', '-dd', help='online detection duration', default=3600, type=int)
@click.option('--predict_interval', '-predict_interval', help='predict interval', default='30s')
def main(publish_date, prometheus_address, service, sc_id, train_date,task_count, step, timeout, train_duration, detection_duration,predict_interval):
    '''
    获取到变更的详细信息
        1.变更时间
        2.变更的指标
        3.在线检测区间
        4.变更id
    '''
    if sc_id == None:
        all_sc_info = get_change_event(publish_date, service)
    else:
        all_sc_info = [k8snet_process(service, sc_id, train_date)]
    for sc_info in all_sc_info:
        #print(sc_info)
        machine_busniess_count = sc_info['machine_bussniess']
        train_data_dir = "data/" + 'sc'+ '/' + service + '/' + sc_id + '/'
        test_data_dir =  "data/" + 'daily'+ '/' + service + '/' + sc_id + '/'                                                                                                                                                       
        nulljson = 0
        train_data_origin = [] 

        # 如果已经存在训练数据，则直接训练，不需要获取数据
        if not os.path.exists(train_data_dir) :
            os.makedirs(train_data_dir)       
            print('训练数据不存在，需要重新获取数据')
            #创建训练对象
            train_data_collector = TrainCollector(prometheus_address, train_data_dir, sc_info, task_count, step, timeout, train_duration)
            train_data_collector.get_proms_request()
            #返回指标的指标名、值
            kpinames, multiple_kpi_1 = asyncio.run(train_data_collector.run_async())
            multiple_kpi_2 = filter_NAN(multiple_kpi_1)
            #训练集的单指标合并为多指标
            multiple_kpi_3 = fix_data_null(multiple_kpi_2, sc_info['train_end_date'] ,train_duration, step)
            multiple_kpi_4 = fix_data_not_end_date(multiple_kpi_3, sc_info['train_end_date'])     

            #训练集的单指标合并为多指标 
            train_merge = Merge(kpinames, multiple_kpi_4)
            train_data_origin = train_merge.merge_kpi()
            savefile(train_data_origin, train_data_dir, 'train_kpi.csv')
            nulljson =  train_data_collector.nulljson

            get_train_log(train_date, train_duration, sc_id)
            shutil.copy("data/" + 'sc'+ '/' + 'k8s-stdout' + '/' + sc_id + '/' + 'train_log.csv', train_data_dir + 'train_log.csv')
        else:
            train_data_origin = []

        if not os.path.exists(test_data_dir) or service == 'yid_':
            print('测试数据不存在需要重新获取数据')
            os.makedirs(test_data_dir) 
            #创建在线检测对象
            test_data_collector = TestCollector(prometheus_address, sc_info, task_count, step, timeout, detection_duration, predict_interval)
            #执行在线检测
            test_data_iterator= test_data_collector.get_proms_request()
             #模型获取测试数据
            kpinames, multiple_kpi = next(test_data_iterator)
            multiple_kpi = filter_NAN(multiple_kpi)
            multiple_kpi = fix_data_not_end_date(multiple_kpi, sc_info['train_end_date'], detection_duration)
            multiple_kpi = fix_data_null(multiple_kpi, sc_info['train_end_date'], detection_duration, step)                          
            test_merge = Merge(kpinames, multiple_kpi)
            test_data_origin = test_merge.merge_kpi()
            test_time_stamp = [arr[0] for arr in test_data_origin[1:]]
            savefile(test_data_origin, test_data_dir, 'test_kpi.csv')
            get_test_log(train_date, detection_duration, sc_id)
            if service == 'yid_k8s':
                shutil.copy("data/" + 'daily'+ '/' + 'k8s-stdout' + '/' + sc_id + '/' + 'test_log.csv', test_data_dir + 'test_log.csv')

        kpi_columns, log_columns, test_length,train_data, test_data = load_data_no_sklearn_addlog(train_data_dir, test_data_dir, train_duration/step)
        #print(test_length)
        # 获取指标数量
        model_train(sc_id, train_data, [kpi_columns, log_columns])

        # 检测流程
        # 将指标数据与日志数据合并  
        train_data_origin = montage_kpi(train_data_dir)
        test_data_origin = montage_kpi_test(test_data_dir)

        df_train, train_score, train_dim_scores = detect_train_data(train_data, sc_id, [kpi_columns, log_columns])
        train_score = np.array([float(i) for i in train_score])
        df_test, test_score, test_dim_score = online_detect_no_sklearn(train_data, test_data, test_data_origin, sc_id, [kpi_columns, log_columns], sc_id)

        #模型开始在线检测
        test_score = np.array([float(i) for i in test_score['MLSTM']])

        # 获取异常点的个数
        _, _, zong_threshold = spot(train_score, sc_id)
        dim_thresholds = dim_spot(train_dim_scores, sc_id)

        output_SC_json_and_csv(sc_id, sc_info, train_data_origin, test_data_origin, test_score, zong_threshold, test_dim_score, dim_thresholds)
        result_plot(service, test_length, sc_id)
        print('end!')
if __name__ == '__main__':
    main()