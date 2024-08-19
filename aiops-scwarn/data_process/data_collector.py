'''
获取机器指标的脚本
'''
import re
import click
import json
import os
import requests
import csv
import asyncio

from data_process.collector import*

def savefile(data, data_dir, dataname):
    # 判断文件夹是否存在，不存在创建新文件夹
    if not os.path.exists(data_dir+dataname):
        os.makedirs(data_dir, exist_ok=True)
        filename = dataname
        save_data_dir = data_dir+filename
        f = open(save_data_dir, 'w')
        csv_write = csv.writer(f)
        for row in data:
            csv_write.writerow(row)
    
        f.close()
    else:
        print("数据已经存在!")

class Merge:
    def __init__(self, kpinames, multiple_kpi) -> None:
        self.kpinames = kpinames
        self.multiple_kpi = multiple_kpi
        #聚合指标后的最终结果
        self.data = []
    
    '''
    [kpiname1,kpiname2......] [[timestamp1, value1], timestamp2, value2],.....] ------>[[timestamp, kpiname1, kpiname2.....],[timestamp1,value1,value2,...]..]
    '''
    def merge_kpi(self) -> List:
        csv_row = ['timestamp']
        for name in self.kpinames:
            csv_row.append(name)
        
        self.data.append(csv_row)    
        kpi_dict = dict()
        timestamps = []
        for i in self.multiple_kpi:
            #print(len(i))
            for j in i:
                timestamp, _ = j
                if timestamp not in kpi_dict.keys():
                    kpi_dict[timestamp]  = []
                    timestamps.append(timestamp)
                    kpi_dict[timestamp].append(timestamp)
                #kpi_dict[timestamp].append(value)
        
        for kpi in self.multiple_kpi:
            if len(kpi) <len(timestamps):
               print("len kpi", len(kpi))
               value = kpi[0][1]
               print(value)
               for timestamp in timestamps:
                   #print(timestamp)
                   kpi_dict[timestamp].append(value)
            else:
                for j in kpi:
                    tp, value = j
                    kpi_dict[tp].append(value)
          
        for row in kpi_dict.values():
            self.data.append(row)
        
        return self.data
