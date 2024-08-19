'''
判断promql语句能否获取数据
'''
import requests

filepath = 'yid_promql/SLA_Prometheus.txt'
hosts = 'bj1-rd-yos-prod-01'
curl = 'http://172.16.17.252:19192/api/v1/query_range?query={0}&start=1685189600&end=1685211200&step=120&timeout=100'

def func(filepath):
    with open(filepath, 'r') as f:
        for promql in f:
            resp = requests.get(curl.format(promql))
            if len(resp.json()['data']['result']) == 0:
               print(promql)
        
func(filepath)
