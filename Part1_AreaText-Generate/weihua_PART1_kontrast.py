"""
这是生成领域文本第一部分的代码 by weihua
2024/11/01 从 219 服务器移动过来
尝试SCWarn运行kontrast数据集后的结果数据
进一步生成领域文本 
"""

from classification import CNNClassifier
import numpy as np
import pandas as pd
import json
import os


def AreaText_Part1(target_index, suffix):
    index = target_index
    # 文件夹路径
    base_folder_path = "/home/zhengtinghua/shenchao/aiops-scwarn/new_res/" + suffix

    model_pt_path = "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/weihua_areaText_generate_219/weihua_train2.pt"
    ## 1,2 同路径, 3,4同路径
    # 1. k1_csv_anomaly
    os.makedirs(
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k1_csv_anomaly/"
        + suffix, exist_ok=True
    )
    csv_anomaly_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k1_csv_anomaly/"
        + suffix
        + "/csv_anomaly_"
        + index
        + ".csv"
    )
    # 2. k2_anomaly_detail
    os.makedirs(
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k2_anomaly_detail/"
        + suffix, exist_ok=True
    )
    anomaly_output_csv_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k2_anomaly_detail/"
        + suffix
        + "/anomaly_details_"
        + index
        + ".csv"
    )
    # 3. k3_prediction
    os.makedirs(
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k3_prediction/"
        + suffix, exist_ok=True
    )
    prediction_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k3_prediction/"
        + suffix
        + "/predictions_"
        + index
        + ".csv"
    )
    """
    2024/11/01 注意这里的 result_json_and_csv 需要重新定义 1/2/3 ...
    """
    # 公用变量
    result_csv_path = (
        base_folder_path + "/result_json_and_csv_" + index + "/result_csv/" + index
    )
    json_path = (
        base_folder_path
        + "/result_json_and_csv_"
        + index
        + "/result_json/result_"
        + index
        + ".json"
    )
    # 4. k4_combine
    os.makedirs(
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k4_combine/"
        + suffix, exist_ok=True
    )
    combine_csv_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k4_combine/"
        + suffix
        + "/combined_"
        + index
        + ".csv"
    )

    os.makedirs(
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k6_cluster/"
        + suffix, exist_ok=True
    )
    # 6. k6_cluster
    cluster_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k6_cluster/"
        + suffix
        + "/cluster_"
        + index
        + ".json"
    )

    ##### part1 record_anomalydata_baseon_cac.py
    folder_path = result_csv_path
    prefix_set = set()  # 存储前缀的集合
    for file_name in os.listdir(folder_path):  # 遍历文件夹中的文件
        if file_name.endswith(".csv"):  # 确保文件是以.csv结尾的
            prefix = file_name.split("_")[0]  # 分割文件名，获取前缀部分
            if prefix.endswith(".csv"):
                prefix_set.add(prefix[:-4])
            else:
                prefix_set.add(prefix)  # 否则，将前缀添加到集合中

    prefix_list = list(prefix_set)  # 将集合转换为列表
    print(f"Number of prefixes: {len(prefix_list)}")
    if "overall" in prefix_list:
        prefix_list.remove("overall")  # 如果存在，就删除它

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dim_info = data["metadata"]["dim_info"]
    prefix_trans_map = {}
    # 定义保存的文件名

    # 将 dim_info 保存为新的 JSON 文件
    with open(cluster_path, "w") as file:
        json.dump(dim_info, file, indent=4)

    # 遍历prefix，获取对应的trans_name数组
    for prefix_1 in prefix_list:
        for dim_info_dict in dim_info:
            if prefix_1 in dim_info_dict:
                trans_name = dim_info_dict[prefix_1]
                prefix_trans_map[prefix_1] = trans_name

    results = []
    anomaly_results = []

    # 初始化计数器
    count_equal_1 = 0
    count_not_equal_1 = 0
    for idx, name in enumerate(prefix_list, start=0):
        csv_path = os.path.join(folder_path, name + ".csv")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        if "model_label" not in df.columns:
            print(f"model_label not in {csv_path}")
            continue

        X = df["model_label"]

        if (X == 1).any():
            count_equal_1 += 1
            results.append([name, "出现异常", prefix_trans_map.get(name, "未知")])

            # 处理异常情况
            for index in X[X == 1].index:
                Y = df["timestamp"][index]
                Z_values = df["origin_value"].tolist()

                anomaly_row = [name, Y]

                # 获取前后各 15 个数据点，使用两个计数器交替
                forward_count = 0
                backward_count = 0
                Z_segment = []

                for i in range(1, 30):
                    if (
                        backward_count <= forward_count
                        and index - backward_count - 1 >= 0
                    ):
                        Z_segment.insert(0, Z_values[index - backward_count - 1])
                        backward_count += 1
                    elif index + forward_count + 1 < len(Z_values):
                        Z_segment.append(Z_values[index + forward_count + 1])
                        forward_count += 1
                    elif index - backward_count - 1 >= 0:
                        Z_segment.insert(0, Z_values[index - backward_count - 1])
                        backward_count += 1
                    else:
                        Z_segment.append(None)

                # 添加中心点
                Z_segment.insert(15, Z_values[index])
                anomaly_row.extend(Z_segment)
                anomaly_results.append(anomaly_row)
        else:
            count_not_equal_1 += 1
            results.append([name, "正常", prefix_trans_map.get(name, "未知")])

    # 将结果写入新的CSV文件
    try:
        results_df = pd.DataFrame(
            results, columns=["数据集名称", "异常情况", "指标名称"]
        )
        results_df.to_csv(csv_anomaly_path, index=False, encoding="utf-8")
        print(f"Info written into {csv_anomaly_path}")
    except Exception as e:
        print(f"Failed to write results CSV: {e}")

    # 写入包含异常详细信息的CSV文件
    try:
        anomaly_columns = ["csvname", "anomaly_timestamp"] + [
            f"Z_value_{i}" for i in range(1, 31)
        ]
        anomaly_results_df = pd.DataFrame(anomaly_results, columns=anomaly_columns)
        anomaly_results_df.to_csv(
            anomaly_output_csv_path, index=False, encoding="utf-8"
        )
        print(f"Info written into {anomaly_output_csv_path}")
    except Exception as e:
        print(f"Failed to write anomaly details CSV: {e}")

    ##### PART 2 : MAKE PREDICTION

    clf = CNNClassifier(class_num=15)
    clf.load_model(model_pt_path)

    input_csv = anomaly_output_csv_path
    df = pd.read_csv(input_csv)

    # 列索引从 1 到 31，含第 32 列
    X = df.iloc[:, 2:32].values

    # 错误判断：检查 X 是否为空
    if X is None or X.size == 0:
        # 如果 X 为空，创建一个空的 DataFrame 并保存为空的 prediction_path 文件
        result_df = pd.DataFrame({"prediction": []})
    else:
        predictions = clf.predict(X)

        result_df = pd.DataFrame(
            {
                "prediction": predictions,
            }
        )
    result_df.to_csv(prediction_path, index=False)

    ##### PART 3 : MAKE COMBINE.CSV
    df1 = pd.read_csv(prediction_path)
    df2 = pd.read_csv(anomaly_output_csv_path)
    result = pd.concat([df2, df1], axis=1)

    # 获取 combine.csv
    result.to_csv(combine_csv_path, index=False)
