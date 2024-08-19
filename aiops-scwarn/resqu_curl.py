
import requests

query = 'probe_success{env="prod", instance="172.16.87.142:9430", job="port_probe"} == 0'

curl = 'http://172.16.17.252:19192/api/v1/query_range?query={0}&start=1685189600&end=1685211200&step=120&timeout=100'

resp = requests.get(curl.format(query))
print(resp.json())
