'''
获取VM和k8s的变更信息:
    1.tag
    2.hosts
    3.duation
    4.create_at
'''
from software_change_event.noah import *
from software_change_event.curl_info import *
import requests
import json

yid_chongqi_url = 'https://noah.yunzhanghu.net/api/v1/deploy/orders?page=30&page_size=100'

def vm_release(interface_url, noah_headers:dict):
    # 获取noah发布中符合变更条件的变更的信息
    json_noah, k8s_id = parse_noah_json(interface_url, noah_headers)
    for json_noah_data in json_noah: 
        # 获取相关order id对应的变更详细信息
        vm_release_info = request_url(Interface.vm_interface_name.format(json_noah_data['id']), Header.header)
        for vmri in vm_release_info.json()['data'][1:]:
            # 获取生产环境的变更
            if vmri['env'] == 'prod' and vmri['record'] != None:
                for record in json.loads(vmri['record']):
                    if record['status'] == 'DONE' and 'duration' in record.keys():
                        json_noah_data['record'] = record;
    
    return json_noah, k8s_id
        
def k8s_release():
    return
