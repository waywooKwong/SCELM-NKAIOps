from weihua_PART1_kontrast import AreaText_Part1
from weihua_PART2_tune_v2_kontrast import AreaText_Part2
from weihua_PART2_English_v2_kontrast import AreaText_Part2_English


kind_suffix = '/emailservice'
index = '40003'
AreaText_Part1(target_index=index, suffix=kind_suffix)
#AreaText_Part2(index,kind_suffix)
AreaText_Part2_English(target_index=index, suffix=kind_suffix)