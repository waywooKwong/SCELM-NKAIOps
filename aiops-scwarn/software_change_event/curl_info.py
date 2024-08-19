'''
保存需要访问的接口
保存需要的header
'''
class Interface:
    deploy_order_interface_name = 'https://noah.yunzhanghu.net/api/v1/deploy/reports/{0}'
    vm_interface_name = 'https://noah.yunzhanghu.net/api/v1/deploy/orders/{0}/steps'
    noah_sla = 'https://noah.yunzhanghu.net/api/v1/sla/rule/{0}'
    noah_service_tree_interface_name = 'https://noah.yunzhanghu.net/api/v1/service/unique/list/'
    noah_aiops_kpi = 'https://noah.yunzhanghu.net/aiops/api/v1/kpi_anamoly/list'
    gitlab_sla = 'https://noah.yunzhanghu.net/api/v1/service/srvs/{0}'

class Header:
    header = {'Authorization': 'Basic YWlvcHNfdXNlcjpuR1hFI0tOMjlibVJxU0ElUg=='} 