'''
获取机器指标的脚本
'''
import re
import click
import json
import requests
import csv
import asyncio
import time
import os
from typing import *

curl = 'http://{0}/api/v1/query_range?query={1}&start={2}&end={3}&step={4}&timeout={5}'

class Config:
    seq_len = 10
         
#数据收集器
class Collector:    
    def __init__(self, prometheus_address, sc_info, task_count, step, timeout):
        self.prometheus_address = prometheus_address
        self.sc_info = sc_info
        self.task_count = task_count
        self.step = step
        self.timeout = timeout
        # 收集所有的request请求的curl
        self.request_urls = []
        # promql对应的指标名 
        self.promql_kpi = dict()
        #self.noah_aipos_promql_name = sc_info['noah_kpi_name']
        #当前任务的数量
        self.current_task = 0
        self.multiple_kpi = []
        self.kpinames = []
        self.nulljson = 0 
    
    #并行睡眠时长
    async def delay_sleep(self, sleep_time):
        await asyncio.sleep(sleep_time)
        return sleep_time
    
    #并行请求
    async def request_url(self, curl):
        json_data = requests.get(curl)
        print(json_data.json())
        try:
            if json_data.json()['data']['result'] != []:
                try:
                    # load dict {'status':'success', 'data':{'resultType': 'matrix', 'result':[{'metric':{}, 'values':[[1668250800, '0.91'],[1668250810,'0.92']]]}}    
                    # kpi_name = self.promql_kpi[curl]
                    kpi_name = self.promql_kpi[curl]
                    # kpi_name
                    #if '__name__' in json_data.json()['data']['result'][0]['metric'].keys():
                        #kpi_name = json_data.json()['data']['result'][0]['metric']['__name__']
                    # hostname
                    # [timestamp, kpi]
                    timestamp_kpi = json_data.json()['data']['result'][0]['values']
                    self.multiple_kpi.append(timestamp_kpi)
                    self.kpinames.append(kpi_name)
                except Exception:
                    print("错误类型",json_data.json()) 
            else:
                #print('[] nulljson')
                lock = asyncio.Lock()
                async with lock:
                    self.nulljson += 1
        except Exception:
            print('exception nulljson:  '+curl)
            print('error type:  ', json_data.json())
            lock = asyncio.Lock()
            async with lock:
                self.nulljson += 1
                    
        self.current_task -= 1
    
    #并行任务初始化   
    async def init_task(self):
        temp = self.request_urls[:self.task_count]
        for each in temp:
            self.request_urls.remove(each)
            asyncio.create_task(self.request_url(each))
            self.current_task += 1
    
    #并行执行程序的主入口
    async def run_async(self) -> int:
        await self.init_task()
        while True:
            print("系统当前正在执行的任务数量:{}".format(self.current_task))
            if self.current_task<self.task_count:
                if self.request_urls:
                    each = self.request_urls[0]
                    del self.request_urls[0]
                    asyncio.create_task(self.request_url(each))
                    self.current_task += 1
                else:
                    if not self.current_task:
                        print("所有任务单元执行完成...终止程序")
                        break
                    else:
                        #  休眠等待所有任务执行完成
                        await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.5)
        return self.kpinames, self.multiple_kpi


class TrainCollector(Collector):
    def __init__(self, prometheus_address, data_dir, sc_info, task_count, step, timeout, train_duration):
        super().__init__(prometheus_address, sc_info, task_count, step, timeout)
        self.data_dir = data_dir
        self.train_duration = train_duration
        self.train_end_time = int(time.mktime(time.strptime(self.sc_info['train_end_date'], "%Y-%m-%d %H:%M:%S")))
        self.train_start_time = self.train_end_time - self.train_duration

    
    def get_proms_request(self):
        for metric in self.sc_info['promql']:
            self.request_urls.append(curl.format(self.prometheus_address,metric,self.train_start_time,self.train_end_time,self.step,self.timeout))
            self.promql_kpi[curl.format(self.prometheus_address,metric,self.train_start_time,self.train_end_time,self.step,self.timeout)] = metric

class TestCollector(Collector):
    def __init__(self, prometheus_address, sc_info, task_count, step, timeout, detection_duration, predict_interval):
        super().__init__(prometheus_address, sc_info, task_count, step, timeout)
        # 怎么改
        self.predict_start_time = int(time.mktime(time.strptime(self.sc_info['sc_end_date'], "%Y-%m-%d %H:%M:%S")))
        self.detection_duration = detection_duration
        self.predict_end_time = self.predict_start_time + self.detection_duration
        self.predict_interval = predict_interval
    
    '''
    获取当前时间戳
    当前时间戳ct 开始时间戳st 结束时间戳et
    1.如果当前时间戳在检测结束时间之外，则可以直接调用
    2.如果当前时间戳在检测结束时间之内，则需要分别考虑 st~ct ct~et
    '''
    def get_proms_request(self):
        current_timestamp = int(time.time())
        if current_timestamp>=self.predict_end_time:
            #print('变更完成后获取数据......')
            for metric in self.sc_info['promql']:
                self.request_urls.append(curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout))
                self.promql_kpi[curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout)] = metric 
        elif current_timestamp<self.predict_end_time:
            for metric in self.sc_info['promql']:
                self.request_urls.append(curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout))
                self.promql_kpi[curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout)] = metric 
            windows_time = Config.seq_len*self.step
            while True:
                print('获取实时数据......')
                end_time = self.predict_start_time + windows_time 
                if end_time > self.predict_end_time:
                    time.sleep(self.predict_end_time-self.predict_start_time)
                    self.request_urls = []
                    self.promql_kpi.clear()
                    for metric in self.sc_info['promql']:
                        self.request_urls.append(curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout))
                        self.promql_kpi[curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout)] = metric  
                    '''
                    如果超过了给定的时间戳 则需要结束在线检测
                    '''
                    break
                else:
                    time.sleep(windows_time)
                    self.request_urls = []
                    for metric in self.sc_info['promql']:
                        self.request_urls.append(curl.format(self.prometheus_address,metric,self.predict_start_time, self.predict_start_time+windows_time,self.step,self.timeout))
                        self.promql_kpi[curl.format(self.prometheus_address,metric,self.predict_start_time,self.predict_end_time,self.step,self.timeout)] = metric 
                self.predict_start_time += windows_time  
        yield asyncio.run(self.run_async()) 
                
