import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
import json
import csv
import os
import time

HUITU = 2160
SUBHUITU = 0

def run_one(dir, meta_file, output_dir):
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    f = open(meta_file)
    jdata = json.load(f)
    metadata = jdata['metadata']
    id = metadata['id']
    dim_infos = metadata['dim_info']

    metrics_cnt = len(dim_infos)
    # fig, fig_a = plt.subplots(metrics_cnt, 1)

    i = 0
    for info in dim_infos:
        key = ''
        promql = ''
        for k, v in info.items():
            key = k
            promql = v
            break
        if(promql == 'timestamp'):
            continue
        # print('key: ' + key + ', promql: ' + promql)

        data_file = dir + '/' + key + '.csv'
        origin_data_file = dir + '/' + key + '_train_origin.csv'
        # 读取训练集和测试集的原始值文件路径
        data, OVERALLTIMESTAMP  = get_csv_plot_data(data_file, origin_data_file)
        output_file = output_dir + '/' + str(int(time.time()))  + key + '.png'
        score_data = get_overall_csv_plot_data(dir + '/overall.csv', HUITU, OVERALLTIMESTAMP)
        # print(f"promql:{promql}")
        # print(f"data:{data[1]}")
        generate_plot_image(output_file, data, score_data, key, promql)
        # plot_data(fig_a, i, data, key, promql)
        i += 1


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


def convert_to_unix_timestamp(timestamp):
    try:
        # 尝试解析为标准时间格式
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except ValueError:
        try:
            # 尝试解析为Unix时间戳
            return int(timestamp)
        except ValueError:
            raise ValueError(f"Unknown timestamp format: {timestamp}")

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
    fig_a[0].axvline(x=change_ts, color='green',
                     label='axvline - full height')

    # fig_a.set_xticks([data[0][0],data[0][length-1]])
    # https://blog.csdn.net/shener_m/article/details/81047862
    # tss = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in data[0]]
    tss = [datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S') if str(d).isdigit() else d for d in data[0]]

    # 将日期字符串解析为 datetime 对象
    tss = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in tss]
    #print(tss)
    # fig_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
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
    
    fig_a[1].axvline(x=change_ts, color='green',
                     label='axvline - full height')
    
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
    
    # ~~~~~~~~~~~~~~~~~~~~    

    # 整体异常分数
    plt.sca(fig_a[2])
    plt.xticks(fontsize=5, rotation=70)
    plt.yticks(fontsize=5)
    fig_a[2].set_title('summary anomaly score', size=8)
    fig_a[2].set_xlabel('t', size=8)

    # tss1 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in data[0]]
    tss1 = [datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S') if str(d).isdigit() else d for d in data[0]]
    tss1 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in tss1]
    #print(tss1)
    # fig_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig_a[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    fig_a[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=240))

    fig_a[2].set_ylabel('value2', size=8)
    fig_a[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig_a[2].yaxis.set_major_locator(mticker.AutoLocator())

    ydata1 = [float(s) for s in score_data[1]]
    min_length = min(len(tss1[0:SUBHUITU]), len(ydata1[0:SUBHUITU]))
    fig_a[2].plot(tss1[0:min_length], ydata1[0:min_length])
    ydata3_1 = [float(s) for s in score_data[3]]
    # print(score_data[2])
    fig_a[2].scatter(score_data[2], ydata3_1, color='red')

    #画阈值
    ydata_ts = [float(s) for s in score_data[4]]
    min_length_ts = min(len(tss1[0:SUBHUITU]), len(ydata_ts[0:SUBHUITU]))
    fig_a[2].plot(tss1[0:min_length_ts], ydata_ts[0:min_length_ts], color='yellow')
    # 保存
    fig.tight_layout()
    fig.savefig(output_file)






def run_all(service, sc_id):
    BASE_DIR = f'result_json_and_csv_{sc_id}'
    output_base_dir = BASE_DIR + '/output/'+service+'/'
    if os.path.exists(output_base_dir) is False:
        os.makedirs(output_base_dir, exist_ok=True)
    run_one(
        dir=BASE_DIR+'/result_csv/'+sc_id,
        meta_file=BASE_DIR+'/result_json/' + 'result_' + sc_id + '.json',
        output_dir=BASE_DIR + '/output/' + service + '/' + sc_id,
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

