import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
import json
import csv
import os
import time
import re

HUITU = 2160
SUBHUITU = 0


# 修改了run_one和run_all的版本
# def run_one(dir, meta_file, output_dir):
#     if os.path.exists(output_dir) is False:
#         os.mkdir(output_dir)
#     f = open(meta_file)
#     jdata = json.load(f)
#     metadata = jdata['metadata']
#     id = metadata['id']
#     dim_infos = metadata['dim_info']

#     metrics_cnt = len(dim_infos)
#     # fig, fig_a = plt.subplots(metrics_cnt, 1)

#     i = 0
#     for info in dim_infos:
#         key = ''
#         promql = ''
#         for k, v in info.items():
#             key = k
#             promql = v
#             break
#         if(promql == 'timestamp'):
#             continue
#         # print('key: ' + key + ', promql: ' + promql)

#         data_file = dir + '/' + key + '.csv'
#         origin_data_file = dir + '/' + key + '_train_origin.csv'
#         # 读取训练集和测试集的原始值文件路径
#         data, OVERALLTIMESTAMP  = get_csv_plot_data(data_file, origin_data_file)
#         output_file = output_dir + '/' + str(int(time.time()))  + key + '.png'
#         score_data = get_overall_csv_plot_data(dir + '/overall.csv', HUITU, OVERALLTIMESTAMP)
#         # print(f"promql:{promql}")
#         # print(f"data:{data[1]}")
#         generate_plot_image(output_file, data, score_data, key, promql)
#         # plot_data(fig_a, i, data, key, promql)
#         i += 1

# 修改位置1：修改了保存结果的路径
def run_one(dir, meta_file, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    with open(meta_file) as f:
        jdata = json.load(f)
    
    metadata = jdata['metadata']
    id = metadata['id']
    dim_infos = metadata['dim_info']

    metrics_cnt = len(dim_infos)

    for info in dim_infos:
        key = next(iter(info.keys()))  # 获取指标 key
        promql = info[key]

        if promql == 'timestamp':
            continue
        
        data_file = os.path.join(dir, f'{key}.csv')
        origin_data_file = os.path.join(dir, f'{key}_train_origin.csv')

        data, OVERALLTIMESTAMP = get_csv_plot_data(data_file, origin_data_file)
        output_file = os.path.join(output_dir, f'{int(time.time())}_{key}.png')

        score_data = get_overall_csv_plot_data(os.path.join(dir, 'overall.csv'), HUITU, OVERALLTIMESTAMP)
        generate_plot_image(output_file, data, score_data, key, promql)



def get_overall_csv_plot_data(data_file, null_cnt, sc_date):
    timestamps = [ sc_date ]
    anomaly_scores = ['0'] * null_cnt
    thresholds = ['0'] * null_cnt
    abnormal_x = []
    abnormal_y = []

    line_cnt = 0
    with open(data_file) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if line_cnt == 0:
                line_cnt += 1
                continue
            timestamps.append(str(datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")+timedelta(minutes=2)))
            anomaly_scores.append(row[1])
            thresholds.append(row[2])
            # modified
            if row[3] == '1':
                abnormal_x.append(timestamps[-1])
                abnormal_y.append(row[1])
            line_cnt += 1
    # print(anomaly_scores)
    return (timestamps, anomaly_scores, abnormal_x, abnormal_y, thresholds)


# def convert_to_unix_timestamp(timestamp):
#     try:
#         # 尝试解析为标准时间格式
#         dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
#         return int(dt.timestamp())
#     except ValueError:
#         try:
#             # 尝试解析为Unix时间戳
#             return int(timestamp)
#         except ValueError:
#             raise ValueError(f"Unknown timestamp format: {timestamp}")

def convert_to_unix_timestamp(timestamp):
    # 调试信息，输出每个 timestamp 的类型和内容
    # print(f"Processing timestamp: {timestamp}, type: {type(timestamp)}")

    try:
        # 1. 处理字符串形式的 time.struct_time
        if isinstance(timestamp, str) and timestamp.startswith("time.struct_time"):
            print("Detected time.struct_time in string format. Attempting to parse it...")
            # 使用正则表达式提取 struct_time 字段
            struct_time_pattern = r"time\.struct_time\(tm_year=(\d+), tm_mon=(\d+), tm_mday=(\d+), tm_hour=(\d+), tm_min=(\d+), tm_sec=(\d+), tm_wday=(\d+), tm_yday=(\d+), tm_isdst=(-?\d+)\)"
            match = re.match(struct_time_pattern, timestamp)

            if match:
                # 将提取的字段转换为整数并构造 time.struct_time 对象
                fields = list(map(int, match.groups()))
                timestamp = time.struct_time(fields)
            else:
                raise ValueError(f"Cannot parse timestamp from string: {timestamp}")

        # 2. 处理 time.struct_time 类型
        if isinstance(timestamp, time.struct_time):
            print("Converting time.struct_time to Unix timestamp...")
            # 如果 tm_isdst 是 -1，手动将其设置为 0
            timestamp = time.struct_time(timestamp[:8] + (0,))
            # 转换为 Unix 时间戳
            return int(time.mktime(timestamp))

        # 3. 处理 datetime 类型
        if isinstance(timestamp, datetime):
            print("Converting datetime to Unix timestamp...")
            return int(timestamp.timestamp())

        # 4. 处理字符串类型的时间戳
        if isinstance(timestamp, str):
            # 按顺序尝试不同的格式
            time_formats = [
                "%Y-%m-%d %H:%M:%S",  # 标准日期时间格式
                "%Y-%m-%dT%H:%M:%S",  # ISO 8601 格式
                "%Y/%m/%d %H:%M:%S"   # 另一种可能的日期时间格式
            ]

            # 处理标准时间格式: 年-月-日 小时:分钟:秒
            for time_format in time_formats:
                try:
                    dt = datetime.strptime(timestamp, time_format)
                    # print(f"Converting string '{time_format}' to Unix timestamp...")
                    return int(dt.timestamp())
                except ValueError:
                    continue

            # 处理 Unix 时间戳（整型或浮点型字符串）
            try:
                print("Converting numeric string to Unix timestamp...")
                return int(float(timestamp))
            except ValueError:
                pass

        # 5. 打印异常的时间戳信息
        print(f"Error: Unable to convert timestamp: {timestamp}")

    except Exception as e:
        print(f"Exception occurred: {e}")

    # 6. 抛出异常，无法处理的时间戳
    raise ValueError(f"Unknown timestamp format: {timestamp}")


# 示例调用
# timestamp = time.struct_time((2024, 10, 7, 0, 0, 0, 0, 281, -1))  # 原始 tm_isdst = -1
# print(convert_to_unix_timestamp(timestamp))
        
# def convert_to_unix_timestamp(timestamp):
#     # 打印调试信息，确认类型
#     print(f"Processing timestamp: {timestamp}, type: {type(timestamp)}")
    
#     # 如果时间戳是 time.struct_time 类型，使用 time.mktime 进行转换
#     if isinstance(timestamp, time.struct_time):
#         print(f"Converting time.struct_time: {timestamp}")
#         return int(time.mktime(timestamp))
    
#     try:
#         # 尝试解析为标准时间格式
#         dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
#         return int(dt.timestamp())
#     except (ValueError, TypeError) as e:
#         print(f"Error parsing string timestamp: {e}")
#         try:
#             # 尝试解析为 Unix 时间戳（字符串或整数形式）
#             return int(timestamp)
#         except (ValueError, TypeError):
#             print(f"Unknown timestamp format: {timestamp}")
#             raise ValueError(f"Unknown timestamp format: {timestamp}")

def get_csv_plot_data(data_file, origin_data_file):
    timestamps = []
    values = []
    abnormal_x = []
    abnormal_y = []
    change_start_ts = ''

    line_cnt = 0

    with open(origin_data_file) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if line_cnt == 0:
                line_cnt += 1
                continue
            timestamps.append(row[0])
            values.append(row[1])
            line_cnt += 1

    timestamps = timestamps[-HUITU:]
    values = values[-HUITU:]

    abnormal_scores = []
    scores = ['0'] * HUITU
    ths = ['0'] * HUITU
    line_cnt = 0
    with open(data_file, mode='r') as csvfile:
        spamreader = csv.reader(csvfile)

        for row in spamreader:
            if line_cnt == 0:
                line_cnt += 1
                continue
            #timestamps.append(str(datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")+timedelta(minutes=2)))
            converted_timestamps = [convert_to_unix_timestamp(ts) for ts in timestamps]
            # # 增加2分钟并转换为字符串格式
            # # new_timestamp = (converted_timestamps[-1] + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
            # timestamps.append(str(datetime.strptime(converted_timestamps[-1], "%Y-%m-%d %H:%M:%S")+timedelta(minutes=2)))

            timestamp_value = converted_timestamps[-1]
            # print(timestamp_value)
            # 如果是时间戳（假设为整数或字符串表示的整数）
            if isinstance(timestamp_value, (int, str)) and str(timestamp_value).isdigit():
                timestamp_value = int(timestamp_value)
                date_string = datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_string = timestamp_value

            # 将日期字符串转换为 datetime 对象并加上时间增量
            new_timestamp = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=2)

            # 将结果添加到 timestamps 列表
            timestamps.append(str(new_timestamp))

            scores.append(row[2])
            ths.append(row[3])
            if line_cnt == 1:
                change_start_ts = timestamps[-1]
                OVERALLTIMESTAMP = timestamps[-1]
            #print(OVERALLTIMESTAMP)
            values.append(row[1])
            if row[4] == '1':
                abnormal_x.append( timestamps[-1]  )
                abnormal_y.append(row[1])
                abnormal_scores.append(row[2])
            line_cnt += 1
        #print(timestamps)
        # print(values)
        #print(abnormal_x)
        # print(abnormal_y)
    #print(timestamps)
    return (timestamps, values, abnormal_x, abnormal_y, change_start_ts, abnormal_scores,scores, ths), OVERALLTIMESTAMP


def generate_plot_image(output_file, data, score_data, metrics_id, promql):
    promql = promql[1:]
    length = len(data[0])
    fig, fig_a = plt.subplots(3, 1)
    plt.sca(fig_a[0])
    plt.xticks(fontsize=5, rotation=70)
    plt.yticks(fontsize=5)

    fig_a[0].set_title(promql, size=8)
    fig_a[0].set_xlabel('t', size=8)
    
    # 变更时间点分割线
    change_ts = datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S')
    fig_a[0].axvline(x=change_ts, color='green', label='axvline - full height')

    # 处理 tss 列表，确保所有元素为字符串格式，并将 time.struct_time 转换为字符串
    tss = []
    struct_time_pattern = re.compile(
        r"time\.struct_time\(tm_year=(\d+), tm_mon=(\d+), tm_mday=(\d+), tm_hour=(\d+), tm_min=(\d+), tm_sec=(\d+), .*?\)"
    )

    for d in data[0]:
        if isinstance(d, time.struct_time):
            d_str = time.strftime('%Y-%m-%d %H:%M:%S', d)  # 转换 struct_time 为字符串
            tss.append(d_str)
        elif isinstance(d, str):
            if d.startswith('time.struct_time'):
                # 使用正则表达式解析 time.struct_time 字符串
                match = struct_time_pattern.match(d)
                if match:
                    # 提取匹配的值，并转换为整数
                    tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec = map(int, match.groups())
                    # 使用 datetime 构造符合格式的字符串
                    d_str = datetime(tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec).strftime('%Y-%m-%d %H:%M:%S')
                    tss.append(d_str)
                else:
                    raise ValueError(f"Could not parse struct_time string: {d}")
            else:
                tss.append(d)  # 字符串直接保留
        elif isinstance(d, (int, float)):
            try:
                # 处理数字时间戳，注意要确保是有效的 UNIX 时间戳
                d_str = datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S')
                tss.append(d_str)
            except (ValueError, OSError) as e:
                print(f"Error converting timestamp {d}: {e}")
        else:
            raise ValueError(f"Unsupported type for timestamp conversion: {type(d)}, value: {d}")

    # 调试输出转换后的 tss
    # print("Processed tss:", tss)

    # 将 tss 列表中的日期字符串解析为 datetime 对象
    try:
        tss = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in tss]
    except ValueError as e:
        print(f"Error parsing date in tss: {e}")
        raise

    fig_a[0].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    fig_a[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=240))

    fig_a[0].set_ylabel('value', size=8)
    fig_a[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig_a[0].yaxis.set_major_locator(mticker.AutoLocator())

    ydata0 = [float(s) for s in data[1] if s.replace('.', '', 1).isdigit()]
    min_length = min(len(tss[0:SUBHUITU]), len(ydata0[0:SUBHUITU]))
    fig_a[0].plot(tss[0:min_length], ydata0[0:min_length])
    ydata3_0 = [float(s) for s in data[3]]
    fig_a[0].scatter(data[2], ydata3_0, color='red')

    # 单指标异常分数
    plt.sca(fig_a[1])
    plt.xticks(fontsize=5, rotation=70)
    plt.yticks(fontsize=5)

    fig_a[1].set_title('anomaly score  ' + promql, size=8)
    fig_a[1].set_xlabel('t', size=8)

    fig_a[1].axvline(x=change_ts, color='green', label='axvline - full height')

    fig_a[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    fig_a[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=240))

    fig_a[1].set_ylabel('AnomalyScore', size=8)

    fig_a[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig_a[1].yaxis.set_major_locator(mticker.AutoLocator())

    ydata_scores = [float(s) for s in data[-2]]
    min_length = min(len(tss[0:SUBHUITU]), len(ydata_scores[0:SUBHUITU]))
    fig_a[1].plot(tss[0:min_length], ydata_scores[0:min_length])
    ydata_ths = [float(s) for s in data[-1]]
    min_length = min(len(tss[0:SUBHUITU]), len(ydata_ths[0:SUBHUITU]))
    fig_a[1].plot(tss[0:min_length], ydata_ths[0:min_length], color='yellow')
    y_anomaly_scores = [float(s) for s in data[-3]]
    fig_a[1].scatter(data[2], y_anomaly_scores, color='red')

    # 整体异常分数
    plt.sca(fig_a[2])
    plt.xticks(fontsize=5, rotation=70)
    plt.yticks(fontsize=5)
    fig_a[2].set_title('summary anomaly score', size=8)
    fig_a[2].set_xlabel('t', size=8)

    # 处理 tss1 列表，确保所有元素为字符串格式
    tss1 = []
    for d in data[0]:
        if isinstance(d, time.struct_time):
            d_str = time.strftime('%Y-%m-%d %H:%M:%S', d)
            tss1.append(d_str)
        elif isinstance(d, str):
            if d.startswith('time.struct_time'):
                match = struct_time_pattern.match(d)
                if match:
                    tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec = map(int, match.groups())
                    d_str = datetime(tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec).strftime('%Y-%m-%d %H:%M:%S')
                    tss1.append(d_str)
                else:
                    raise ValueError(f"Could not parse struct_time string: {d}")
            else:
                tss1.append(d)
        elif isinstance(d, (int, float)):
            try:
                d_str = datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S')
                tss1.append(d_str)
            except (ValueError, OSError) as e:
                print(f"Error converting timestamp {d}: {e}")
        else:
            raise ValueError(f"Unsupported type for timestamp conversion: {type(d)}, value: {d}")

    try:
        tss1 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in tss1]
    except ValueError as e:
        print(f"Error parsing date in tss1: {e}")
        raise

    fig_a[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    fig_a[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=240))

    fig_a[2].set_ylabel('value2', size=8)
    fig_a[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig_a[2].yaxis.set_major_locator(mticker.AutoLocator())

    ydata1 = [float(s) for s in score_data[1]]
    min_length = min(len(tss1[0:SUBHUITU]), len(ydata1[0:SUBHUITU]))
    fig_a[2].plot(tss1[0:min_length], ydata1[0:min_length])
    ydata3_1 = [float(s) for s in score_data[3]]
    fig_a[2].scatter(score_data[2], ydata3_1, color='red')

    # 画阈值
    ydata_ts = [float(s) for s in score_data[4]]
    min_length_ts = min(len(tss1[0:SUBHUITU]), len(ydata_ts[0:SUBHUITU]))
    fig_a[2].plot(tss1[0:min_length_ts], ydata_ts[0:min_length_ts], color='yellow')

    # 保存
    fig.tight_layout()
    fig.savefig(output_file)



# def run_all(service, sc_id):
#     BASE_DIR = f'result_byte_dance/result_json_and_csv_{sc_id}'
#     output_base_dir = BASE_DIR + '/output/'+service+'/'
#     if os.path.exists(output_base_dir) is False:
#         os.makedirs(output_base_dir, exist_ok=True)
#     run_one(
#         dir=BASE_DIR+'/result_csv/'+sc_id,
#         meta_file=BASE_DIR+'/result_json/' + 'result_' + sc_id + '.json',
#         output_dir=BASE_DIR + '/output/' + service + '/' + sc_id,
#     )

# def run_all(service, sc_id):
#     BASE_DIR = f'result_byte_dance/{service}'  # 修改为 service 作为顶级目录
#     sc_dir = f'{BASE_DIR}/result_json_and_csv_{sc_id}'
    
#     if not os.path.exists(sc_dir):
#         os.makedirs(sc_dir, exist_ok=True)

#     output_dir = f'{sc_dir}/output'
#     os.makedirs(output_dir, exist_ok=True)

#     run_one(
#         dir=f'{sc_dir}/result_csv',
#         meta_file=f'{sc_dir}/result_json/result_{sc_id}.json',
#         output_dir=output_dir,
#     )

# 修改位置2：修改了保存结果的路径
def run_all(service, sc_id):
    if not service:
        raise ValueError("Error: service is empty!")  # 防止 service 为空

    BASE_DIR = f'result_byte_dance/{service}'
    os.makedirs(BASE_DIR, exist_ok=True)  # 确保 service 目录存在

    sc_dir = f'{BASE_DIR}/result_json_and_csv_{sc_id}'
    os.makedirs(sc_dir, exist_ok=True)  # 确保 sc_id 目录存在

    output_dir = f'{sc_dir}/output/{service}/{sc_id}'
    os.makedirs(output_dir, exist_ok=True)

    # print(f"BASE_DIR: {BASE_DIR}")  # 调试信息
    # print(f"sc_dir: {sc_dir}")      # 调试信息

    run_one(
        dir=f'{sc_dir}/result_csv/{sc_id}',
        meta_file=f'{sc_dir}/result_json/result_{sc_id}.json',
        output_dir=output_dir,
    )




def result_plot(service, test_length, sc_id):
    global SUBHUITU
    SUBHUITU = HUITU + test_length
    run_all(service, sc_id)


if __name__ == '__main__':
    result_plot()
# run_one(
#     dir=BASE_DIR+'/result_csv/'+ORDER_ID,
#     meta_file=BASE_DIR+'/result_json/'+ 'result_' + ORDER_ID + '.json',
#     output_dir=BASE_DIR + '/output/' + ORDER_ID,
# )

