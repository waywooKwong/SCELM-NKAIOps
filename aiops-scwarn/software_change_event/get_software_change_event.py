from software_change_event.noah import *
from software_change_event.vm_k8s import *
from software_change_event.curl_info import *
import time
import pandas as pd

yid_hosts = ['bj1-rd-yos-prod-01', 'bj3-rd-yos-prod-03', 'bj3-rd-yos-prod-02']
ymsg_hosts = ['bj1-rd-yos-prod-01', 'bj3-rd-yos-prod-03', 'bj3-rd-yos-prod-02']
ycard_hosts = ['bj1-rd-ycard-prod-01', 'bj2-rd-ycard-prod-02', 'bj3-rd-ycard-prod-03']
ylint_hosts = ['bj3-rd-ylint-prod-01']
bkverify_hosts = ['bj1-rd-bkverify-prod-01', 'bj3-rd-bkverify-prod-02', 'bj1-rd-bkverify-prod-03']

def get_hosts(service):
    hosts = []
    if service == 'yid':
        hosts = yid_hosts
    elif service == 'ymsg':
        hosts = ymsg_hosts
    elif service == 'ylint':
        hosts = ylint_hosts
    elif service == 'ycard':
        hosts = ycard_hosts
    elif service == 'bkverify':
        hosts = bkverify_hosts   
 
    return hosts

def get_machine_promql(service)->list:
    hosts = get_hosts(service)
    machine_promql = []
    promqls = []
    with open('software_change_event/service_promql/{0}/machine.txt'.format(service), 'r') as f:
        for promql in f:
            promqls.append(promql.strip())
    
    for host in hosts:
        for promql in promqls:
            machine_promql.append(promql.format(host))    
    
    
    return machine_promql

def get_Service_promql(service)->list:
    hosts = get_hosts(service)
    service_promql = []
    try:
        with open('software_change_event/service_promql/{0}/SLA_Prometheus.txt'.format(service), 'r') as f:
            for host in hosts:    
                for promql in f:
                    service_promql.append(promql.strip())
    except Exception:
        return service_promql
    
            
    return service_promql

class MachinePromQL():
    machine_PromQL = ['1-avg(irate(node_cpu_seconds_total{{hostname=~"{0}", mode=~"idle"}}[1m])) by (mode, hostname)','node_load1{{hostname=~"{0}"}}',\
        'node_load5{{hostname=~"{0}"}}','node_load15{{hostname=~"{0}"}}',\
        '(1-(node_memory_MemAvailable_bytes{{hostname=~"{0}"}}/node_memory_MemTotal_bytes{{hostname=~"{0}"}}))*100',\
        'node_memory_Buffers_bytes{{hostname=~"{0}"}}','node_memory_MemFree_bytes{{hostname=~"{0}"}}',
        '1-(node_filesystem_free_bytes{{hostname=~"{0}",fstype=~"ext4|xfs",mountpoint="/data"}} / node_filesystem_size_bytes{{hostname=~"{0}",fstype=~"ext4|xfs",mountpoint="/data"}})',\
        'irate(node_disk_reads_completed_total{{hostname=~"{0}"}}[1m])','irate(node_disk_writes_completed_total{{hostname=~"{0}"}}[1m])',\
        'node_netstat_Tcp_CurrEstab{{hostname=~"{0}"}}',\
        'irate(node_network_transmit_bytes_total{{hostname=~"{0}",device!~"tap.*|veth.*|br.*|docker.*|virbr*|lo*"}}[5m])*8']

class SLAPromQL():
    http_and_grpc_PromQL = ['sum(increase(std_requests_total{{job="{0}",environment="prod"}}[1m])) by(job)', 'sum(increase(std_grpc_server_handled_total{{job="{0}",environment="prod"}}[1m])) by(job)']

'''
遍历服务树，获取服务节点到负责人的映射
'''

def format_noah_json():
    kpi_json = requests.get(Interface.noah_aiops_kpi).json()
    kpi_promql = dict()
    promql_name = dict()
    for kpi in kpi_json["data"]:
        if kpi['kpi_skey'] not in kpi_promql.keys():
            kpi_promql[kpi['kpi_skey']] = []
        kpi_promql[kpi['kpi_skey']].append(kpi['kpi_query'])
        promql_name[kpi['kpi_query']] = kpi['kpi_name']
        
    return kpi_promql, promql_name

def traverse_the_service_tree() -> dict:
    service_rdadmin = dict()
    for service_node in requests.get(Interface.noah_service_tree_interface_name, headers=Header.header).json()['data']:
        service_rdadmin[service_node['skey']] = service_node['rd_admin']
    
    return service_rdadmin

def k8s_sc_start_time() -> dict:
    # 对于k8s变更，需要获取I_ORDER_ID和D_CREATED_AT
    df = pd.read_csv('software_change_event/k8s.csv')
    I_ORDER_ID = df['I_ORDER_ID'].values
    D_CREATED_AT = df['D_CREATED_AT'].values
    ID_CREATED = dict()
    for i in range(len(I_ORDER_ID)):
        ID_CREATED[I_ORDER_ID[i]] = D_CREATED_AT[i]
    
    return ID_CREATED
'''
获取PromQL指标
    1.获取机器指标
获取变更时间
    1.获取变更开始时间
    2.获取变更结束时间
获取http请求
'''
def get_mertics_and_time(json_noah:list, service_rdadmin:dict(), k8s_id: list, sim_service)->dict():
    sc_info = []
    # {'id': 20542, 'service': 'yzh.infra.engineering2.yid', 'record': {'tag': 'v0.18.15', 'hosts': ['hb3a-rd-yos-prod-01', 'hb3a-rd-yos-prod-02'], 'operator': 'zongming.jin', 'create_at': '2023-04-01 02:08:36', 'status': 'DONE', 'category': 'deploy', 'duration': 122}}
    ID_CREATED = k8s_sc_start_time()
    for each_json_noah in json_noah:
        #print(json_noah)
        if each_json_noah['id'] == 27479: 
            #print(each_json_noah)
            id_time_mertics = dict()
            # 获取service
            id_time_mertics['service'] = each_json_noah['service']
            service = id_time_mertics['service']
            id_time_mertics['id'] = each_json_noah['id']
            id_time_mertics['title'] = each_json_noah['title']
            id_time_mertics['tag'] = each_json_noah['tag']
            id_time_mertics['status'] = each_json_noah['status']
            id_time_mertics['deploy_type'] = each_json_noah['deploy_type']
            id_time_mertics['fix_version'] = each_json_noah['fix_version']
            id_time_mertics['k8s_shark_deploy'] = each_json_noah['k8s_shark_deploy']
            # 获取服务负责人
            id_time_mertics['rd_admin'] = service_rdadmin[service]
            # 获取机器名
            #hosts = each_json_noah['record']['hosts']
            id_time_mertics['hosts'] = yid_hosts
            # 获取训练截止时间
            create_at = each_json_noah['record']['create_at']
            id_time_mertics['train_end_date'] = create_at
            create_at_timestamp = int(time.mktime(time.strptime(each_json_noah['record']['create_at'], "%Y-%m-%d %H:%M:%S")))
            duration = each_json_noah['record']['duration']
            # 获取在线检测开始时间
            sc_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_at_timestamp))
            id_time_mertics['sc_end_date'] = sc_end
        
            #获取机器指标
            machine_promql = get_machine_promql(sim_service)
            id_time_mertics['machine_promql'] = machine_promql
        
            #获取SLA指标
            service_promql = get_Service_promql(sim_service)
            id_time_mertics['service_promql'] = service_promql
        
            #指标相加
            id_time_mertics['promql'] = machine_promql+service_promql
            id_time_mertics['machine_bussniess'] = [len(machine_promql), len(service_promql)]
            print(id_time_mertics['machine_bussniess'] ) 
            sc_info.append(id_time_mertics)
            break
    return sc_info
                
def get_change_event(publish_date, service)->dict():
    service_rdadmin = traverse_the_service_tree()
    json_noah, k8s_id = vm_release(Interface.deploy_order_interface_name.format(publish_date), Header.header)
    
    return get_mertics_and_time(json_noah, service_rdadmin, k8s_id, service)

if __name__ == '__main__':
    #curl --location --request GET 'https://noah.yunzhanghu.net/api/v1/deploy/reports/20230330' --header 'Authorization: Basic YWlvcHNfdXNlcjpuR1hFI0tOMjlibVJxU0ElUg=='
    get_change_event('20230406')

