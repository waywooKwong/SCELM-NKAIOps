from weihua_PART1_kontrast import AreaText_Part1
from weihua_PART2_tune_v2_kontrast import AreaText_Part2
from weihua_PART2_English_v2_kontrast import AreaText_Part2_English


def get_all_child_folder(base_folder_path):

    import os

    # 设置大文件夹路径
    new_res_folder = base_folder_path

    # 用于记录所有的 kpi 和 xxxxx
    kpi_list = []
    xxxxx_list = []

    # 遍历 `new_res` 文件夹，获取所有子文件夹
    for kpi_folder in os.listdir(new_res_folder):
        kpi_folder_path = os.path.join(new_res_folder, kpi_folder)

        # 检查是否是文件夹，并且以 KPI 命名
        if os.path.isdir(kpi_folder_path):
            kpi_list.append(kpi_folder)

            # 进入每个 kpi 文件夹，查找符合格式的子文件夹
            for sub_folder in os.listdir(kpi_folder_path):
                sub_folder_path = os.path.join(kpi_folder_path, sub_folder)

                # 检查子文件夹名称是否符合 `result_json_and_csv_xxxxx` 格式
                if os.path.isdir(sub_folder_path) and sub_folder.startswith(
                    "result_json_and_csv_"
                ):
                    xxxxx = sub_folder[len("result_json_and_csv_") :]  # 提取 xxxxx 部分
                    xxxxx_list.append(xxxxx)
    # 去重
    kpi_list = list(set(kpi_list))
    xxxxx_list = list(set(xxxxx_list))
    return kpi_list, xxxxx_list


base_folder_path = (
    "/home/zhengtinghua/shenchao/aiops-scwarn/new_res"  # 替换为你的文件夹路径
)
kpi_list, index_list = get_all_child_folder(base_folder_path)

# for kpi in kpi_list :
#     if kpi!="emailservice" and kpi!="adservice" and kpi!="productcatalogservice" and kpi!="system" and kpi!="view_cartpage"and kpi!="checkoutpage" and kpi!="shippingservice"and kpi!="recommendationservice":
#         for index in index_list:
#             AreaText_Part1(target_index=index, suffix=kpi)
#             #AreaText_Part2(index,kind_suffix)
#             AreaText_Part2_English(target_index=index, suffix=kpi)


kpi = "checkoutservice"
for index in index_list:
    AreaText_Part1(target_index=index, suffix=kpi)
    # AreaText_Part2(index,kind_suffix)
    AreaText_Part2_English(target_index=index, suffix=kpi)
