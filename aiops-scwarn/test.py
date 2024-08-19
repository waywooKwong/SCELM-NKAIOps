from data_process.log_process_collector_3_new import *
import pandas as pd

train_data_path='/home/sunyongqian/liuheng/multi-model_dataset/data-eadro/TT Dataset/no fault/TT.2022-04-21T105158D2022-04-21T130705/logs.json'
ini_path='/home/sunyongqian/liuheng/aiops-scwarn/data_process/drain3.ini'
test_data_path='/home/sunyongqian/liuheng/multi-model_dataset/data-eadro/SN Dataset/data/SN.2022-04-17T181245D2022-04-17T183616/logs.json'

def run():
    #训练
    #1、把参与训练的数据，全部用drain跑一遍，文件模式，产出持久化文件.bin（.ini是改规则的地方）
    #2、时间窗口的训练，文件模式读取本地的持久化文件.bin，根据时间窗口，进行模板计数，转成训练的时序数据
    #3、检测，文件模式读取.bin文件，根据时间窗口，进行模板计数，转成测试的时序数据，拿一个列表存产生的新的日志
    # get_train_data(train_data_path)
    drain_train(train_data_path)



if __name__ == '__main__':
    run()