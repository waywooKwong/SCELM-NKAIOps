from weihua_PART1 import AreaText_Part1
from weihua_PART2_tune_v2 import AreaText_Part2
from weihua_PART2_English_v2 import AreaText_Part2_English

kind_suffix = '/yid_k8s'
for index in range(10079,10083):
    index = str(index)
    AreaText_Part1(index)
    AreaText_Part2(index,kind_suffix)
    AreaText_Part2_English(index,kind_suffix)