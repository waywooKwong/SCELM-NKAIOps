import click


@click.command()
@click.option('--prometheus-address', 'prometheus_addr', help='prometheus address', default='127.0.0.1:9091')
@click.option('--metrics', 'metrics', multiple=True, help='prometheus metrics to query', default=['hello', 'world'])
@click.option('--data-dir', 'data_dir', help='data directory', default='./data_output')
@click.option('--train-start-time', 'train_start_time', help='train start time', default='2023-03-10T00:00:00')
@click.option('--train-end-time', 'train_end_time', help='train end time', default='2023-03-10T00:10:00')
@click.option('--predict-start-time', 'predict_start_time', help='predict start time', default='2023-03-10T00:00:00')
@click.option('--predict-end-time', 'predict_end_time', help='predict end time', default='2023-03-10T00:10:00')
@click.option('--predict-interval', 'predict_interval', help='predict interval', default='30s')
def run(
        prometheus_addr,
        metrics,
        data_dir,
        train_start_time,
        train_end_time,
        predict_start_time,
        predict_end_time,
        predict_interval,
):

    # 连接 prometheus
    click.echo(f"prometheus address: {prometheus_addr}")
    click.echo(f"metrics: {metrics}")
    tst = click.DateTime(train_start_time)
    click.echo(f"train start time: {tst}")

    # 模型训练阶段-开始
    # 查询各个指标项数据
    # 拼接成向量数据

    # data_dir 中需要存储的数据包括：
    # - 训练数据
    # - 模型文件
    # - 预测数据
    # - 报告文件

    # 调用 pytorch 开始训练得到模型
    # TODO: 步骤待补充

    # 模型文件存储
    # 先存储到本地文件即可

    # 模型训练阶段-结束

    # 预测阶段-开始
    # 加载模型

    # 批量
    # 判断当前时间与 predict_start_time 的关系
    # 对早于当前时间的时间段
    # 以批量的方式, 直接查询 prometheus 获取指标项数据
    # 拼接成向量数据
    # 调用模型进行预测

    # 增量
    # 每间隔 predict_interval 查询一次 prometheus 指标项数据
    # 调用模型进行预测
    # 直到 current_time > predict_end_time 时结束

    # 预测结果写入本地文件 (json格式)


if __name__ == '__main__':
    run()
