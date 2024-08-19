import os
os.chdir("/home/devops/intelligent-change-nku-liuheng-yid-log-modified")
#os.system("sudo -su root")
#os.system("source /data/.bashrc")
#os.system("conda activate scwarn")
from data_process.collector import*
from data_process.data_collector import*
from data_process.log_process_collector_3 import *
from module import *
from software_change_event.get_software_change_event import *
from plot_data import *
import click
import os
import numpy as np
import datetime
import time
import pandas as pd


PROMETHEUS_ADDRESS = '172.16.17.252:19192'
TESTDURATION = 86400
STEP = 120
TIMEOUT = 1000
TASKCOUNT = 4
TRAINDURATION = 604800
ROOTDIR = '/home/devops/intelligent-change-nku-liuheng-yid-log-modified/data'
TRAINDIR = '/home/devops/intelligent-change-nku-liuheng-yid-log-modified/data/sc/{}/{}/'

def get_test_date():
    before_timestamp = int(time.time())  - 86400
    datetime_object = datetime.datetime.fromtimestamp(before_timestamp)
    test_date  = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
    return test_date

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


def get_infomation(test_date, service_name):
    info = dict()
    info['id'] = test_date.split(' ')[0]
    info['sc_end_date'] = test_date
    info['service'] = service_name
    info['promql'] = get_machine_promql(service_name) + get_Service_promql(service_name)
    info['test_dir'] = ROOTDIR  + '/' + 'daily' +'/'+ service_name + '/' + test_date.split(' ')[0] + '/'
    return info

def get_daily_log(test_date, service_name, logstore, sc_id):
    endtimestamp = int(datetime.datetime.strptime(test_date, '%Y-%m-%d %H:%M:%S').timestamp()) + 86400
    datetime_object = datetime.datetime.fromtimestamp(endtimestamp)
    end_time  = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
    get_predict_datadaily_data(log_params.project, logstore, test_date, end_time, log_params.minute, sc_id)

class log_params:
    project = 'yzh-log-service'
    minute = 2
 
@click.command()
@click.option('--service_name', '-s', help='service', default='yid', type=str)
@click.option('--logstore', '-l', default='beyid', type=str)
@click.option('--sc_id', '-c', help='case', default=None)
# 输入变更日期返回所需要的变更事件
def main(service_name, logstore, sc_id):  
  
    test_date = "2023-10-26 17:09:18"
    end_time = "2023-10-27 17:09:18"
    train_dir = TRAINDIR.format(service_name, sc_id)
    
    info = get_infomation(test_date, service_name)
    #创建在线检测数据获取对象
    test_data_collector = TestCollector(PROMETHEUS_ADDRESS, info, TASKCOUNT , STEP, TIMEOUT, TESTDURATION, 30)
    #执行在线检测
    test_data_iterator= test_data_collector.get_proms_request()
    train_data_origin = montage_kpi(train_dir)
    
    while True:
        try:
            #模型获取测试数据
            kpinames, multiple_kpi = next(test_data_iterator)
            multiple_kpi = filter_NAN(multiple_kpi)
            #print([len(i) for i in multiple_kpi])
            multiple_kpi = fix_data_not_end_date(multiple_kpi, info['sc_end_date'], TESTDURATION)
            multiple_kpi = fix_data_null(multiple_kpi, info['sc_end_date'], TESTDURATION, STEP)
            # multiple_kpi = fix_data_not_end_date(multiple_kpi, info['sc_end_date'], TESTDURATION) 
            test_merge = Merge(kpinames, multiple_kpi)
            test_data_origin = test_merge.merge_kpi()
            _ = [arr[0]  for arr in test_data_origin[1:]]
            savefile(test_data_origin, info['test_dir'], 'test_kpi.csv')
            
            # 获取日志
            #get_daily_log(test_date, service_name, logstore, sc_id)
            
            train_data, test_data = load_data_no_sklearn_addlog(train_dir, info['test_dir'], TRAINDURATION/STEP)
            
            # 获取指标数量
            machine = len(test_data_origin[0])-1
            busniess = len(train_data[0])-machine
            
            _, train_score, train_dim_scores = detect_train_data(train_data, sc_id, [machine, busniess])
            train_score = np.array([float(i) for i in train_score])
            #模型开始在线检测
            _, test_score, test_dim_score = online_detect_no_sklearn(train_data, test_data, test_data_origin, sc_id, [machine, busniess], info['id'])
            test_score = np.array([float(i) for i in test_score['MLSTM']])
            
            # # 获取异常点的个数
            _, _, zong_threshold = spot(train_score, info['id'])

            dim_thresholds = dim_spot(train_dim_scores, info['id'])
            
            # 输出检测结果
            train_data_origin = montage_kpi(train_dir)
            test_data_origin = montage_kpi_test(info['test_dir'])
            output_SC_json_and_csv(info['id'], info, train_data_origin, test_data_origin, test_score, zong_threshold, test_dim_score, dim_thresholds)
            result_plot(service_name)
        except StopIteration:
                break 
        
if __name__ == '__main__':
    main()   
 #main(run_params.test_date, run_params.service_name, run_params.train_dir, run_params.sc_id)
    

