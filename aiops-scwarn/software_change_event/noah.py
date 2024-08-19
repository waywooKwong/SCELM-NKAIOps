'''
访问noah获取变更工单
'''
import requests

def request_url(interface_url, noah_headers:dict):
    return requests.get(interface_url, headers = noah_headers)

def parse_noah_json(interface_url, noah_headers:dict):
    # 获取json对象
    json_ob = request_url(interface_url, noah_headers)
    json_noah = []
    k8s_id = []
    # 解析json文件
    for each_sc in json_ob.json()['data']['deploy_orders']:
        # 过滤条件 {status: '发布完成'/'已完成', stage: 'Prod', "k8s_shark_deploy": false} 虚拟机部署
        if each_sc['status'] in ["发布完成" ,"已完成"] and each_sc['stage'] == 'Prod':
            if each_sc['k8s_shark_deploy'] == True:
               k8s_id.append(each_sc['id']) 
            # 提取有用的信息包括 id service deploy_app_stages
            json_each_noah = dict()
            json_each_noah['id'] = each_sc['id']
            json_each_noah['service'] = each_sc['service']
            json_each_noah['title'] = each_sc['title']
            json_each_noah['tag'] = each_sc['tag']
            json_each_noah['status'] = each_sc['status']
            json_each_noah['deploy_type'] = each_sc['deploy_type']
            json_each_noah['fix_version'] = each_sc['fix_version']
            json_each_noah['k8s_shark_deploy'] = each_sc['k8s_shark_deploy']
            json_noah.append(json_each_noah)
            
    return json_noah, k8s_id
