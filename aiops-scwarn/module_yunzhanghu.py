import pandas as pd
import yaml
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
from SCWarn.approach.LSTM.lstm import *
from SCWarn.approach.LSTM.MLSTM import *
from SCWarn.approach.AutoEncoder.AE import *
from SCWarn.approach.AutoEncoder.VAE import *
from SCWarn.approach.AutoEncoder.MMAE import *
from SCWarn.approach.Metrics.ISST import ISST_predict
from SCWarn.approach.GRU.GRU import *
from SPOT.spot import dSPOT, SPOT
from utils import *
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import json
import os
import time
import json 
import csv
import hashlib
import math
from datetime import datetime

class Args:
    train_path = "data/{0}/train.csv"
    test_path = "data/{0}/abnormal.csv"
    output_path = "result/{0}/res.csv"
    output_dim_scores_path = "result/{0}/dim_scores.csv"
    model_path = "model/{0}"
    result_path = 'result/{0}.png'
    #scaler = 'standard'
    scaler = 'minmax'

# 将日志数据加入原始数据中，返回相应值
def add_log_origin(train_origin_data, test_origin_data):
    df_train_log = pd.read_csv('datalog/train2.csv')
    df_test_log = pd.read_csv('datalog/predict2.csv')
    df_train_log = df_train_log.set_index('timestamp')
    df_test_log = df_test_log.set_index('timestamp')
    
    train_origin_data[0] = train_origin_data[0] + list(df_train_log.columns)
    count = 1
    for index, row in df_train_log.iterrows():
        train_origin_data[count] = train_origin_data[count] + list(row)

# Global Configuration
with open("configs.yml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 输出为json
def output_SC_json_and_csv(sc_id, sc_info, train_data, test_data:list, test_zong_score:dict, zong_threshold:dict, test_dim_score, fen_threshold: list):
    output_dict = {
        "metadata":{
        "id": sc_id,
        #"sc_start_time": sc_info['train_end_date'],
        #"kpi_counter": sc_info['machine_bussniess'], 
        "promql": sc_info['promql'],
        "service": sc_info['service'], 
        #"rd_admin": sc_info['rd_admin'], 
        #"hosts": sc_info['hosts'],
        #"title":sc_info['title'],
        #"tag":sc_info['tag'],
        #"status":sc_info['status'],
        #"deploy_type":sc_info['deploy_type'],
        #"fix_version":sc_info['fix_version'],
        #"k8s_shark_deploy":sc_info['k8s_shark_deploy'],
        #"train_status": "",
        "dim_info": []
        },
    "online_detection_data": [],
    "anomaly_score":[],
                   }
    #print(len(train_data))
    max_distance = 1 
    if max_distance != 0:
        csv_save_dir = 'result_json_and_csv_{0}/result_csv/{0}/'.format(sc_id)
        if not os.path.exists(csv_save_dir):
            os.makedirs(csv_save_dir)
        
        # 获取promql语句
        promql = train_data[0][1:]
        # 创建promql到md5的映射
        promql_md5 = dict()
        
        for pql in promql:
            md5 = hashlib.md5()
            md5.update(pql.encode('utf-8'))
            promql_md5[pql] = md5.hexdigest()
        
        # 获取训练时间戳
        train_timestamps = []
        for data in train_data[-3600:]:
            # train_timestamps.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(data[0])))) 
            # train_timestamps.append(int(datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S').timestamp()))
            train_timestamps.append(int(data[0]))
            # train_timestamps.append(data[0])
        train_data_header = ['timestamp', 'origin_value']
        for i in range(len(promql)):
            csv_train_dim_save_path = csv_save_dir + promql_md5[promql[i]] + '_train_origin' + '.csv'
            f_train_dim = open(csv_train_dim_save_path, 'w')
            csv_train_dim_writer = csv.writer(f_train_dim)
            csv_train_dim_writer.writerow(train_data_header)
            train_data_column = [float(j[i+1]) for j in train_data[-3600:]]
            for time_stamp, origin_value in zip(train_timestamps, train_data_column):
                csv_train_dim_writer.writerow([time_stamp, origin_value])
            f_train_dim.close()
        
        # 获取测试时间戳
        test_timestamps = []
        for data in test_data[1:]:
            # test_timestamps.append(int(datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S').timestamp()))
            # test_timestamps.append(data[0])     
            #test_timestamps.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(data[0]))))
            test_timestamps.append(data[0])       
 
        csv_dim_header = ['timestamp', 'origin_value', 'anomaly_score', 'threshold', 'model_label']
        for i in range(len(promql)):
            # 获取在线检测的结果
            online_detection_data_dict = dict()
            online_detection_data_dict[promql[i]] = []
            test_data_column = [float(j[i+1])  for j in test_data[1:]]
            '''
            保存每个测试指标的csv文件
            '''
            csv_dim_save_path = csv_save_dir + promql_md5[promql[i]] + '.csv'
            f_dim = open(csv_dim_save_path, 'w')
            csv_dim_writer = csv.writer(f_dim)
            csv_dim_writer.writerow(csv_dim_header)
            '''
            保存每个指标的promql和文件名id
            '''
            id_promql = dict()
            id_promql[promql_md5[promql[i]]] = promql[i]
            output_dict['metadata']['dim_info'].append(id_promql)
             
            for time_stamp, anomaly_score, threshold, origin_value in zip(test_timestamps, test_dim_score[i], fen_threshold[i]['thresholds'], test_data_column[:]):
                dot_dict = dict()
                # 将检测结果保存为csv文件
                csv_dim_row_origin_score_threshold = []
                dot_dict['timestamp'] = time_stamp
                dot_dict['score'] = anomaly_score
                dot_dict['threshold'] = threshold
                dot_dict['origin'] = origin_value
                csv_dim_row_origin_score_threshold.append(time_stamp)
                csv_dim_row_origin_score_threshold.append(origin_value)
                csv_dim_row_origin_score_threshold.append(anomaly_score)
                csv_dim_row_origin_score_threshold.append(threshold)
                if anomaly_score > threshold:
                    csv_dim_row_origin_score_threshold.append(1)
                else:
                    csv_dim_row_origin_score_threshold.append(0)

                if anomaly_score > threshold:
                    dot_dict['is_anomaly'] = True
                else:
                    dot_dict['is_anomaly'] = False
                online_detection_data_dict[promql[i]].append(dot_dict)
                csv_dim_writer.writerow(csv_dim_row_origin_score_threshold)
            
            f_dim.close()
            output_dict['online_detection_data'].append(online_detection_data_dict)
             
        # 测试数据的数量
        overall_save_name = 'overall.csv'
        csv_overall_header = ['timestamp', 'anomaly_score', 'threshold', 'model_label']    
        f_overall = open(csv_save_dir+overall_save_name, 'w')
        csv_overall_write = csv.writer(f_overall)
        csv_overall_write.writerow(csv_overall_header) 
        online_count = len(test_timestamps)
        for i in range(online_count):
            csv_overall_score_threshold = []
            # 补全json数据中的anomaly字段
            anomaly_score_dict = dict()
            anomaly_score_dict['timestamp'] = test_timestamps[i]
            # if i<10:
            #     anomaly_score_dict['threshold'] = 0
            #     anomaly_score_dict['anomaly_score'] = 0
            # else:
            anomaly_score_dict['threshold'] = zong_threshold['thresholds'][i]
            anomaly_score_dict['anomaly_score']  = test_zong_score[i]
            
            if anomaly_score_dict['anomaly_score'] > anomaly_score_dict['threshold']:
                anomaly_score_dict['is_anomaly'] = True
            else:
                anomaly_score_dict['is_anomaly'] = False
            output_dict['anomaly_score'].append(anomaly_score_dict)
            
            csv_overall_score_threshold.append(anomaly_score_dict['timestamp'])
            csv_overall_score_threshold.append(anomaly_score_dict['anomaly_score'])
            csv_overall_score_threshold.append(anomaly_score_dict['threshold'])
            if anomaly_score_dict['anomaly_score'] > anomaly_score_dict['threshold']:
                csv_overall_score_threshold.append(1)
            else:
                csv_overall_score_threshold.append(0)
            csv_overall_write.writerow(csv_overall_score_threshold)
        
        f_overall.close()
            
        output_dict['train_status'] = "train success!"
    else:
        output_dict['train_status'] = "train failed"
    
    result_json_save_name = f'result_json_and_csv_{sc_id}/result_json/'
    if not os.path.exists(result_json_save_name):
        os.makedirs(result_json_save_name)
    result_json_save_path = f'result_json_and_csv_{sc_id}/result_json/' + 'result_{0}.json'.format(sc_id)
    with open(result_json_save_path, 'w') as f:
        json.dump(output_dict, f)

def add_log(csv_file, data):
    log_kpi = []
    df = pd.read_csv(csv_file)
    log_kpi.append(df.columns[1])
    for value in df['logs_kpi']:
        log_kpi.append(value)
    
    for i in range(len(log_kpi)):
        data[i].append(log_kpi[i])
    
    return data

def fix_data_null(mul_kpi, train_end_date, duration, step):
    temp_mul_kpi = []
    timestamp =  int(time.mktime(time.strptime(train_end_date, '%Y-%m-%d %H:%M:%S')))
    start_stamp = timestamp-duration
    for kpi in mul_kpi:
        temp_kpi = []
        temp_kpi.append(kpi[0])
        temp_point = 0
        for i in range(1, len(kpi)):
            if kpi[i][0] - temp_kpi[temp_point][0] != step:
                multiple = int((kpi[i][0] - temp_kpi[temp_point][0]) /step)
                for j in range(multiple):
                    temp_time = temp_kpi[temp_point][0] + step
                    temp_value = temp_kpi[temp_point][1]
                    temp_kpi.append([temp_time, temp_value])
                    temp_point += 1
            else:
                temp_kpi.append(kpi[i])
                temp_point += 1
        temp_mul_kpi.append(temp_kpi)
    
    # for kpi in temp_mul_kpi:
    #     print(kpi)
    return temp_mul_kpi
 
def fix_data_not_end_date(mul_kpi, train_end_date, duration=0):
    timestamp =  int(time.mktime(time.strptime(train_end_date, '%Y-%m-%d %H:%M:%S')))    
    temp_mul_kpi = []
    for kpi in mul_kpi:
        temp_kpi = kpi
        stamps = [i[0] for i in kpi]
        if timestamp not in stamps:
            temp_kpi.append([timestamp, kpi[len(kpi)-1][1]])
        temp_mul_kpi.append(temp_kpi)
    
    return temp_mul_kpi

def filter_NAN(multiple_kpi):
    for single_kpi in multiple_kpi:
        for i in range(len(single_kpi)):
            if single_kpi[i][1]=='NaN':
                single_kpi[i][1] = single_kpi[i-1][1]
    return multiple_kpi 

'''
# 补数据
    1.对于非irate数据 补前一个值
    2.对于irate数据 补0
'''
def fix_data(train_end_date, kpinames, multiple_kpi, step):
    stamp =  int(time.mktime(time.strptime(train_end_date, '%Y-%m-%d %H:%M:%S')))
    temp_multiple_kpi = []
    for i in range(len(kpinames)):
        temp = []
        slide1 = 0
        slide2 = 0
        if 'irate' in kpinames[i]:
            if multiple_kpi[i][0][0] != stamp:
                temp.append([stamp, 0])
                slide2 = 0
            else:
                temp.append(multiple_kpi[i][0])
                slide2 = 1
            while slide2 < len(multiple_kpi[i]):
                if int(multiple_kpi[i][slide2][0]) - int(temp[slide1][0]) == step:
                    temp.append(multiple_kpi[i][slide2])
                    slide1 += 1
                else:
                    for l in range (int((multiple_kpi[i][slide2][0] - temp[slide1][0])/step)):
                        temp.append([temp[slide1][0]+step, 0])
                        slide1 += 1
                slide2 += 1
            temp_multiple_kpi.append(temp)
        else:
            temp_multiple_kpi.append(multiple_kpi[i])
    return temp_multiple_kpi

       

def load_data_no_sklearn(train_data_path, test_data_path, train_size) ->list:
    mul_kpi = []
    df_train = pd.read_csv(train_data_path+'train.csv')
    #print(df_train)
    df_train = df_train.set_index('timestamp')
    df_test = pd.read_csv(test_data_path+'test.csv')
    df_test = df_test.set_index('timestamp')
    
    kpi_names = []
    for kpi_name in df_train.columns:
        kpi_names.append(kpi_name)
        mul_kpi.append( list(df_train[kpi_name])+list(df_test[kpi_name]))
    
    data = run_norm(kpi_names, mul_kpi, Args.scaler)
    return data[0: int(train_size)+1], data[int(train_size)+1:] 

def load_data_no_sklearn_addlog(train_data_path, test_data_path, train_size) ->list:
    with open(test_data_path+'test_kpi.csv', 'r') as file:
        # 使用逗号作为分隔符创建CSV阅读器
        reader = csv.reader(file, delimiter=',')
        # 读取文件的第一行，通常是列标题行
        first_row = next(reader)
        # 计算列数
        kpi_columns = len(first_row)-1
    
    with open(test_data_path+'test_log.csv', 'r') as file:
        # 使用逗号作为分隔符创建CSV阅读器
        reader = csv.reader(file, delimiter=',')
        # 读取文件的第一行，通常是列标题行
        first_row = next(reader)
        # 计算列数
        log_columns = len(first_row)-1
    
    
    mul_kpi = []
    df_train = pd.read_csv(train_data_path+'train_kpi.csv')
    df_train = df_train.set_index('timestamp')
    df_test = pd.read_csv(test_data_path+'test_kpi.csv')
    df_test = df_test.set_index('timestamp')
 
    #process log
    df_train_log = pd.read_csv(train_data_path+'train_log.csv')
    df_test_log = pd.read_csv(test_data_path+'test_log.csv')
    df_train_log = df_train_log.set_index('timestamp')
    df_test_log = df_test_log.set_index('timestamp')
    
   
    kpi_names = []
    for (kpi_name1,kpi_name2) in zip(df_train.columns,df_test.columns):
        kpi_names.append(kpi_name1)
        mul_kpi.append( list(df_train[kpi_name1])+list(df_test[kpi_name2]))
   
    for kpi_name in df_train_log.columns:
        kpi_names.append(kpi_name)
        mul_kpi.append( list(df_train_log[kpi_name])+list(df_test_log[kpi_name]))

    data = run_norm(kpi_names, mul_kpi, Args.scaler)
    return kpi_columns, log_columns, len(data)-int(train_size), data[0: int(train_size)+1], data[int(train_size)+1:]

    

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.set_index('timestamp')

    # normalize
    if Args.scaler == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    if Args.scaler == 'standard':
        scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df, scaler

#训练模型，系统里调用这个
def train_model(algorithm, train_data: np.ndarray, params, sc_id:str, modal:list):
    if algorithm == "MLSTM":
        model = get_model_MLSTM(train_data, modal,
                                params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
    if not os.path.exists(Args.model_path.format(sc_id)):
        os.makedirs(Args.model_path.format(sc_id))
    torch.save(model, Args.model_path.format(sc_id)+f"/{algorithm}_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

#跑mlstm模型，判断是否有模型，如果没有模型再训练一次，并得到预测的异常分数
def run_mlstm(train_data: np.ndarray, test_data: np.ndarray, params, sc_id:str, modal:list):
    if not os.path.exists(Args.model_path.format(sc_id)):
        os.makedirs(Args.model_path.format(sc_id))
        model = get_model_MLSTM(train_data, modal,
                            params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
        torch.save(model, Args.model_path.format(sc_id) + f"/MLSTM_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")
    else:
        print('model already exists!')
        model = torch.load(Args.model_path.format(sc_id)+f"/MLSTM_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

     
    print(len(test_data))
    scores, dim_scores = get_prediction_MLSTM_test(model, train_data, test_data, params['seq_len'], modal)
    return scores, dim_scores

#跑算法，每个算法都能跑，其他算法已经删掉了，直接调这个就成
def run_algorithms(algorithms, train_data: np.ndarray, test_data: np.ndarray, sc_id: str, modal:list):
    results = {}
    for i in algorithms:
        run = None
        if i == "MLSTM":
            run = run_mlstm

        if run is not None:
            #异常分数结果输出
            scores, dim_scores = run(train_data, test_data, algorithms[i], sc_id, modal)
            results[i] = [float(i) for i in scores]

            # if 'seq_len' in algorithms[i]:
            #     seq_len = algorithms[i]['seq_len']
            #     results[i] = [np.nan] * seq_len + results[i]

        # incorrect configuration
        else:
            print(f"{i} isn't included in SCWarn. Please check the config.yml.")

    return results, dim_scores

#异常检测，输出异常分数，系统里调用这个
def output_score(df_test_no_transform, df_train: pd.DataFrame, df_test: pd.DataFrame, sc_id: str, modal:list):
    df_train, df_test = df_train.copy(), df_test.copy()
    # run algorithms
    train_data, test_data = df_train, df_test
    test_data = test_data.astype(float)
    results, dim_results = run_algorithms(config['algorithms'], train_data, test_data, sc_id, modal)
    # save results
    df_results = pd.DataFrame(results)
    df_results['timestamp'] = df_test_no_transform.index
    df_results = df_results.set_index('timestamp')
    shape = dim_results.shape
    ndim = dim_results.ndim
    new_shape = (shape[0] * shape[1], shape[2]) if ndim > 2 else (shape[0], shape[1])
    dim_results_2d = dim_results.reshape(new_shape)
    dim_res = pd.DataFrame(dim_results_2d)
    dim_res['timestamp'] = df_test_no_transform.index[0:shape[0]]
    dim_res = dim_res.set_index('timestamp')
    return df_results, results, dim_res

#异常检测，输出异常分数，系统里调用这个
def output_score_no_sklearn(train_data, test_data, test_timestamp, sc_id: str, modal:list):
    results, dim_results = run_algorithms(config['algorithms'], train_data, test_data, sc_id, modal)
    # save results
    df_results = pd.DataFrame(results)
    # print(df_results)
    df_results['timestamp'] = test_timestamp
    df_results = df_results.set_index('timestamp')
    shape = dim_results.shape
    ndim = dim_results.ndim
    new_shape = (shape[0] * shape[1], shape[2]) if ndim > 2 else (shape[0], shape[1])
    dim_results_2d = dim_results.reshape(new_shape)
    dim_res = pd.DataFrame(dim_results_2d)
    dim_res['timestamp'] = test_timestamp


    dim_res = dim_res.set_index('timestamp')
    return df_results, results, dim_res

#异常检测可视化
def anomaly_pic(df_results:pd.DataFrame):
    df_results = df_results.copy()

    x = df_results['timestamp']
    y = df_results['MLSTM']
    plt.plot(x, y)
    plt.xlabel('timestamp')
    plt.ylabel('MLSTM')
    plt.savefig('result/results.png')
    
def model_train(sc_id, train_data, modal:list):
    os.makedirs("model", exist_ok=True)
    #根据文件路径训练模型
    # df_train, _ = load_data(Args.train_path.format(sc_id))
    for algorithm in config['algorithms']:
        train_model(algorithm, train_data, config['algorithms'][algorithm], sc_id, modal)

#test_data用于接收在线数据输出的列表
def online_detect(test_data, sc_id:str, modal:list):
    #列表转成dataframe对象
    df_train, scaler = load_data(Args.train_path.format(sc_id))
    df_test = pd.DataFrame(test_data[1:], columns=test_data[0])
    #数据标准化
    df_test = df_test.set_index('timestamp')
    df_test_no_transform = df_test
    df_test = scaler.transform(df_test)
    #异常检测输出异常分数并输出保存成csv文件
    df_results,results, dim_res = output_score(df_test_no_transform, df_train, df_test, sc_id, modal)
    os.makedirs(os.path.dirname(Args.output_path.format(sc_id)), exist_ok=True)
    df_results.to_csv(Args.output_path.format(sc_id))
    dim_res.to_csv(Args.output_dim_scores_path.format(sc_id))
    
    return df_test, results, dim_res

#test_data用于接收在线数据输出的列表
def online_detect_no_sklearn(train_data, test_data, test_data_origin,sc_id:str, modal:list, if_id: str):
    #异常检测输出异常分数并输出保存成csv文件
    test_timestamp = [i[0] for i in test_data_origin[1:]]
    df_results,results, dim_res = output_score_no_sklearn(train_data, test_data, test_timestamp, sc_id, modal)
    os.makedirs(os.path.dirname(Args.output_path.format(if_id)), exist_ok=True)
    df_results.to_csv(Args.output_path.format(if_id))
    dim_res.to_csv(Args.output_dim_scores_path.format(if_id))
    
    return test_data, results, dim_res


#检测train data,用于spot的训练
def detect_train_data(train_data, sc_id,modal:list, params=config['algorithms']['MLSTM']):
    model = torch.load(Args.model_path.format(sc_id)+f"/MLSTM_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")
    scores, dim_score = get_prediction_MLSTM(model, train_data[-720:], params['seq_len'], modal)
    shape = dim_score.shape
    ndim = dim_score.ndim
    new_shape = (shape[0] * shape[1], shape[2]) if ndim > 2 else (shape[0], shape[1])
    dim_score_2d = dim_score.reshape(new_shape)
    dim_scores = pd.DataFrame(dim_score_2d)
    return train_data, scores, dim_score
    
def spot(train_score, sc_id:str) -> int:
    #检测结果的异常分数处理
    test_score = pd.read_csv(Args.output_path.format(sc_id))
    test_data_np = np.array(test_score['MLSTM'])
    data = test_data_np[:]
    #训练数据的异常分数处理
    init_data = train_score
    q = 0.03			# risk parameter
    d = 10				# depth parameter
    s = dSPOT(q,d)     	# biDSPOT object
    #s = SPOT(q)
    s.fit(init_data,data) 	# data import
    s.initialize() 	  		# initialization step
    results = s.run()    	# run
    # counter返回异常点的数量，alarm返回异常点的位置
    _, counter, alarm = s.plot(results, Args.result_path, sc_id)
    return counter, alarm, results

def dim_spot(train_dim_score, sc_id:str):
    test_score_pd = pd.read_csv(Args.output_dim_scores_path.format(sc_id))
    results = []
    for i in range(1, len(test_score_pd.columns)):
        test_data_np = np.array(test_score_pd[test_score_pd.columns[i]])
        init_data = np.array(train_dim_score[:, 0, i-1])
        q = 0.03	 			# risk parameter
        d = 10
        #s = SPOT(q)  				# depth parameter
        s = dSPOT(q,d)     	# biDSPOT object
        s.fit(init_data,test_data_np)
        s.initialize()
        result = s.run()
        results.append(result)
    return results

#找异常分数最大值和其所在位置
def find_max_value(sc_id:str):
    df = pd.read_csv(Args.output_path.format(sc_id))
    max_value = df['MLSTM'].max()
    max_index = df['MLSTM'].idxmax()
    return max_value, max_index

#根据最大值位置找阈值
def get_data_by_row(results, row):
    df = pd.DataFrame(results['thresholds'])
    print(row-10)
    return df.iloc[row-10]

#计算异常分数最大值和阈值之间的距离
def get_max_distance(sc_id:str, results):
    test_score, row = find_max_value(sc_id)
    threshold = get_data_by_row(results, row)
    max_distance = abs(test_score-threshold[0])
    return max_distance

#计算异常分数与阈值的平均距离
def get_average_difference(sc_id:str, results):
    df1 = pd.read_csv(Args.output_path.format(sc_id))
    df2 = pd.DataFrame(results['thresholds'])
    df1['column1'] = df1.loc[:, 'MLSTM'].astype('float64')
    df2['column2'] = df2.loc[:].astype('float64')
    df2['difference'] = abs(df1['column1'].sub(df2['column2']))
    average_distance = df2['difference'].mean()
    return average_distance




