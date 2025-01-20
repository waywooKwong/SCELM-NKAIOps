from classification import CNNClassifier
import numpy as np
import pandas as pd
import json
import os

def AreaText_Part1(target_index):
    index = target_index
    model_pt_path = '/home/sunyongqian/liuheng/Time-Series-Library-main/AIOps_dataset/fluxrankplus/src/fluxrankplus/model/weihua_train2.pt'
    ## 1,2 同路径, 3,4同路径
    # 1. k1_csv_anomaly
    csv_anomaly_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k1_csv_anomaly/csv_anomaly_'+index+'.csv'
    # 2. k2_anomaly_detail
    anomaly_output_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k2_anomaly_detail/anomaly_details_'+index+'.csv'
    # 3. k3_prediction
    prediction_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k3_prediction/predictions_'+index+'.csv'

    # 公用变量
    result_csv_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_'+index+'/result_csv/'+index  
    json_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv_'+index+'/result_json/result_'+index+'.json'
    # 4. k4_combine
    combine_csv_path = '/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k4_combine/combined_'+index+'.csv'
    
    ##### part1 record_anomalydata_baseon_cac.py
    folder_path = result_csv_path
    prefix_set = set()  # 存储前缀的集合
    for file_name in os.listdir(folder_path):  # 遍历文件夹中的文件
        if file_name.endswith('.csv'):  # 确保文件是以.csv结尾的
            prefix = file_name.split('_')[0]  # 分割文件名，获取前缀部分
            if prefix.endswith('.csv'):
                prefix_set.add(prefix[:-4])
            else:
                prefix_set.add(prefix)  # 否则，将前缀添加到集合中

    prefix_list = list(prefix_set)  # 将集合转换为列表
    print(f"Number of prefixes: {len(prefix_list)}")
    if "overall" in prefix_list:
        prefix_list.remove("overall")  # 如果存在，就删除它

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dim_info = data['metadata']['dim_info']
    prefix_trans_map = {}

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
        csv_path = os.path.join(folder_path, name + '.csv')
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue
        
        if 'model_label' not in df.columns:
            print(f"model_label not in {csv_path}")
            continue

        X = df['model_label']

        if (X == 1).any():
            count_equal_1 += 1
            results.append([name, '出现异常', prefix_trans_map.get(name, '未知')])

            # 处理异常情况
            for index in X[X == 1].index:
                Y = df['timestamp'][index]
                Z_values = df['origin_value'].tolist()

                anomaly_row = [name, Y]

                # 获取前后各 15 个数据点，使用两个计数器交替
                forward_count = 0
                backward_count = 0
                Z_segment = []

                for i in range(1, 30):
                    if backward_count <= forward_count and index - backward_count - 1 >= 0:
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
            results.append([name, '正常', prefix_trans_map.get(name, '未知')])

    # 将结果写入新的CSV文件
    try:
        results_df = pd.DataFrame(results, columns=['数据集名称', '异常情况', '指标名称'])
        results_df.to_csv(csv_anomaly_path, index=False, encoding='utf-8')
        print(f"Info written into {csv_anomaly_path}")
    except Exception as e:
        print(f"Failed to write results CSV: {e}")

    # 写入包含异常详细信息的CSV文件
    try:
        anomaly_columns = ['csvname', 'anomaly_timestamp'] + [f'Z_value_{i}' for i in range(1, 31)]
        anomaly_results_df = pd.DataFrame(anomaly_results, columns=anomaly_columns)
        anomaly_results_df.to_csv(anomaly_output_csv_path, index=False, encoding='utf-8')
        print(f"Info written into {anomaly_output_csv_path}")
    except Exception as e:
        print(f"Failed to write anomaly details CSV: {e}")


    ##### PART 2 : MAKE PREDICTION

    clf = CNNClassifier(class_num =15)
    clf.load_model(model_pt_path)

    input_csv = anomaly_output_csv_path
    df = pd.read_csv(input_csv)

    # 列索引从 1 到 31，含第 32 列
    X = df.iloc[:, 2:32].values 
    predictions = clf.predict(X)

    result_df = pd.DataFrame({
        'prediction': predictions,
    })
    result_df.to_csv(prediction_path, index=False)


    ##### PART 3 : MAKE COMBINE.CSV
    df1 = pd.read_csv(prediction_path)
    df2 = pd.read_csv(anomaly_output_csv_path)
    result = pd.concat([df2, df1], axis=1)

    # 获取 combine.csv
    result.to_csv(combine_csv_path, index=False)