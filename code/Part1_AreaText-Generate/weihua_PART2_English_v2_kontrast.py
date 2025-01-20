import torch
from model_English import Model_English
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json
from time_bounds import get_time_bounds_false, get_time_bounds_true, get_time_bounds

# sum_weihua_part123.py and demo_orgin_v2.py


def convert_to_date(time_data):
    def is_timestamp(val):
        try:
            # 检查是否为10位的时间戳，无论是字符串还是整数
            val_str = str(val)
            return val_str.isdigit() and len(val_str) == 10
        except ValueError:
            return False

    def timestamp_to_date(timestamp):
        # 将时间戳转换为日期格式
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(time_data, list):
        converted_data = []
        for item in time_data:
            if is_timestamp(item):
                converted_data.append(timestamp_to_date(item))
            else:
                # 假设其他格式的数据是正确的日期格式，可以直接添加
                converted_data.append(item)
        return converted_data
    else:
        # 如果是单个值
        if is_timestamp(time_data):
            return timestamp_to_date(time_data)
        else:
            return time_data


data_index = 3


def AreaText_Part2_English(target_index, suffix):

    index = target_index
    kind_suffix = suffix
    base_folder_path = "/home/zhengtinghua/shenchao/aiops-scwarn/new_res/" + suffix
    # 5.
    text_folder = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/areaText/English/" + suffix
    )
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    text_path = text_folder + "/areaText_" + index + "_v4_English.txt"
    overall_csv_path = (
        base_folder_path
        + "/result_json_and_csv_"
        + index
        + "/result_csv/"
        + index
        + "/overall.csv"
    )

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

    # 4.
    combine_csv_path = (
        "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/process_mid_data/k4_combine/"
        + suffix
        + "/combined_"
        + index
        + ".csv"
    )

    # Obtain time boundaries
    index_suffix = "/" + index
    # train_start_time, train_end_time, test_start_time, test_end_time = get_time_bounds_false(base_path, kind_suffix, index_suffix)
    train_start_time, train_end_time, test_start_time, test_end_time = (
        "2024-10-07 00:00:00",
        "2024-10-07 23:58:00",
        "2024-10-08 00:00:00",
        "2024-10-08 23:36:00",
    )
    # Convert time to string format
    # train_start_time_str = str(train_start_time).strftime('%Y-%m-%d %H:%M:%S')
    # train_end_time_str = str(train_end_time).strftime('%Y-%m-%d %H:%M:%S')
    # test_start_time_str = str(test_start_time).strftime('%Y-%m-%d %H:%M:%S')
    # test_end_time_str = str(test_end_time).strftime('%Y-%m-%d %H:%M:%S')
    # Print results
    print("index:", index, " ", "kind_suffix:", kind_suffix)
    print(f"Train Start Time: {train_start_time}")
    print(f"Train End Time: {train_end_time}")
    print(f"Test Start Time: {test_start_time}")
    print(f"Test End Time: {test_end_time}")

    configs = {
        "description": "Sample dataset",
        "pred_len": 10,
        "seq_len": 8,
        "top_k": 3,
        "enc_in": 4,
    }
    model = Model_English(configs)

    # Get prefix names
    folder_path = result_csv_path  # Replace with your folder path
    prefix_set = set()  # Set to store prefixes
    for file_name in os.listdir(folder_path):  # Iterate through files in the folder
        if file_name.endswith(".csv"):  # Ensure file ends with .csv
            prefix = file_name.split("_")[0]  # Split filename to get prefix
            if prefix.endswith(".csv"):
                prefix_set.add(prefix[:-4])
            else:
                prefix_set.add(prefix)

    prefix_list = list(prefix_set)  # Convert set to list
    if "overall" in prefix_list:
        prefix_list.remove("overall")  # Remove if exists
    input_string = prefix_list

    # Type mapping dictionary
    type_dict = {
        1: "Sudden increase",
        2: "Sudden decrease",
        3: "Level shift up",
        4: "Level shift down",
        5: "Steady increase",
        6: "Steady decrease",
        7: "Single spike",
        8: "Single dip",
        9: "Transient level shift up",
        10: "Transient level shift down",
        11: "Multiple spikes",
        12: "Multiple dips",
        13: "Fluctuations",
    }

    # Get prefix corresponding trans names
    # Read JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dim_info = data["metadata"]["dim_info"]
    prefix_trans_map = {}

    # Iterate through prefixes to get corresponding trans_name arrays
    for prefix in prefix_list:
        for dim_info_dict in dim_info:
            if prefix in dim_info_dict:
                trans_name = dim_info_dict[prefix]
                prefix_trans_map[prefix] = trans_name

    # Initialize content list
    content = []
    # Write initial content
    # content.append("Anomaly change domain text:\n")
    kind_suffix_0 = kind_suffix.replace("/", "")
    content.append("id:NO.{} \n".format(index))
    content.append("Service:{}\n".format(kind_suffix_0))
    content.append("Submission start time:" + train_start_time + "\n")
    content.append("Submission end time:" + train_end_time + "\n")
    content.append("Analysis start time:" + test_start_time + "\n")
    content.append("Analysis end time:" + test_end_time + "\n")
    content.append("Analysis of kpi changes related to the service:\n")

    df = pd.read_csv(overall_csv_path)
    fault_rows = df[df["model_label"] == 1]["timestamp"]
    if not fault_rows.empty:
        result = fault_rows.to_string(index=False)
    else:
        result = "no fault"

    content.append(
        "\nSCWARN algorithm identifies anomalies at the following timestamps:\n"
    )
    result_lines = result.splitlines()
    result_lines = convert_to_date(result_lines)
    for idx, line in enumerate(result_lines, start=0):
        content.append(f"    {idx + 1}, {line}\n")

    df = pd.read_csv(combine_csv_path)
    pattern_data = df["prediction"]
    unique_values = pattern_data.unique()
    unique_count = len(unique_values)

    exception_count = 1
    anomaly_count = 0
    timestamp_count = 0
    pattern_count = 0
    pattern_list = []

    text = """\nThese are definitions for anomaly types (patterns):
    Anomaly description shapes are divided into two major categories: Still in abnormal state and Recover to normal state,
    Still in abnormal state, these anomalies remain in an abnormal state after appearing at the anomaly point
    1. Sudden increase
    2. Sudden decrease
    3. Level shift up
    4. Level shift down
    5. Steady increase
    6. Steady decrease
    Recover to normal state, these anomalies return to normal state after appearing at the anomaly point
    7. Single spike
    8. Single dip
    9. Transient level shift up
    10. Transient level shift down
    11. Multiple spikes
    12. Multiple dips
    13. Fluctuations\n"""
    content.append(text)

    content.append(
        "\nTypes of single kpi anomalies related to overall anomalies (single kpi anomalies not related to overall anomalies are not output):"
    )
    current_csvname = None

    # Store anomaly information for each metric
    anomaly_info = []
    # The abnormal kpi overlaps with the timestamp in overall
    kpi_concering_overall = []
    for idx, row in df.iterrows():
        anomaly_count += 1
        csvname = row["csvname"]
        timestamp = row["anomaly_timestamp"]
        timestamp = convert_to_date(row["anomaly_timestamp"])
        prediction = row["prediction"]

        # Use mapping dictionary to convert numbers to English nouns
        prediction_type = type_dict.get(prediction, prediction)

        if csvname == current_csvname:
            anomaly_info[-1]["timestamps"].append((timestamp, prediction_type))
            timestamp_count += 1
            if prediction_type not in pattern_list:
                pattern_list.append(prediction_type)
                pattern_count += 1
        else:
            if current_csvname is not None:
                if anomaly_info:
                    anomaly_info[-1]["summary"] = (
                        "\n      This kpi has {} anomaly types in total, with types: {}".format(
                            pattern_count, pattern_list
                        )
                    )
                    timestamp_count = 0
                    pattern_count = 0
                    pattern_list = []

            current_csvname = csvname
            if csvname in prefix_trans_map:
                anomaly_info.append(
                    {
                        "header": "\n    {}.kpi name: {}".format(
                            exception_count, prefix_trans_map[csvname]
                        ),
                        "timestamps": [(timestamp, prediction_type)],
                    }
                )
                if prefix_trans_map[csvname] not in kpi_concering_overall:
                    kpi_concering_overall.append(prefix_trans_map[csvname])
                exception_count += 1
                timestamp_count = 1
                if prediction_type not in pattern_list:
                    pattern_list.append(prediction_type)
                    pattern_count = 1
                else:
                    pattern_count = 0

    # If there are still anomaly information of the last metric not written
    if current_csvname is not None:
        if anomaly_info:
            anomaly_info[-1]["summary"] = (
                "\n      This kpi has {} anomaly types in total, with types: {}".format(
                    pattern_count, pattern_list
                )
            )

    # Write introduction of specific anomaly metrics
    for info in anomaly_info:
        pattern_timestamp_map = {}
        for timestamp, pattern in info["timestamps"]:
            if pattern not in pattern_timestamp_map:
                pattern_timestamp_map[pattern] = []
            pattern_timestamp_map[pattern].append(timestamp)

        for pattern, timestamps in pattern_timestamp_map.items():
            if pattern in {
                type_dict[7],
                type_dict[8],
                type_dict[9],
                type_dict[10],
                type_dict[11],
                type_dict[12],
                type_dict[13],
            }:  # Check for patterns 7 to 13
                overlapping_timestamps = [ts for ts in timestamps if str(ts) in result]
                if overlapping_timestamps:
                    content.append(info["header"])
                    if "summary" in info:
                        content.append(info["summary"])
                    content.append(
                        "\n      Among them, type [{}] recovered to normal, timestamps coinciding with overall anomalies: {}".format(
                            pattern, ", ".join(map(str, overlapping_timestamps))
                        )
                    )
            else:
                # Print all timestamps for patterns 1 to 6
                content.append(info["header"])
                if "summary" in info:
                    content.append(info["summary"])
                content.append(
                    "\n      Among them, type [{}] remained abnormal, timestamps are: {}".format(
                        pattern, ", ".join(map(str, timestamps))
                    )
                )

    content.append("\n\nThe overall data of kpi before and after the change\n")

    for col_index in range(0, len(input_string)):

        csv_path = folder_path + "/" + prefix_list[col_index] + "_train_origin.csv"
        df = pd.read_csv(csv_path)
        input_data = df["origin_value"]  # Read data as required

        input_data = torch.tensor(input_data.values).view(
            1, len(input_data), 1
        )  # Convert to default shape

        prefix_value = prefix_trans_map.get(prefix_list[col_index], None)
        if prefix_value != None:
            content.append(
                "    "
                + str(col_index + 1)
                + ".kpi name: "
                + prefix_trans_map[prefix_list[col_index]]
            )
        prompt, num_before = model.forecast(input_data)
        # print('num_before',num_before[0][0])
        content.append("\n        Before change: ")  # Source name
        for p in prompt:
            content.append(str(p))

        csv_path = folder_path + "/" + prefix_list[col_index] + ".csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            input_data = df["origin_value"]  # Read data as required
            # print("input_data:",input_data)

            input_data = torch.tensor(input_data.values).view(
                1, len(input_data), 1
            )  # Convert to default shape
            prompt, num_after = model.forecast(input_data)
            # print('num_after',num_after[0][1])
            content.append("\n        After change：")  # name
            for p in prompt:
                content.append(str(p))

            content.append(
                "\n        Comparison of data ranges before and after change: Before change range: [{},{}], After change range: [{},{}]\n".format(
                    num_before[0][0], num_before[0][1], num_after[0][0], num_after[0][1]
                )
            )

    print("single csv data finish in to output.txt")

    # Write content to file
    with open(text_path, "a") as file:
        file.writelines(content)

    # Write number of anomalies at the end of the file
    with open(text_path, "a") as file:
        file.write("Total anomalies found: {}\n".format(anomaly_count - 1))
        file.write("Total number of anomalous kpis: {}\n".format(exception_count - 1))
        file.write(
            "Total number of anomaly description shapes: {}\n\n".format(unique_count)
        )

    print("overall csv data finish in to output.txt")
    # print(len(prefix_list))
    # print("anomaly_info:", anomaly_info)
