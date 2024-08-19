import torch
from model import Model
import matplotlib.pyplot as plt
import pandas as pd

configs = {
    'description': 'Sample dataset',
    'pred_len': 10,
    'seq_len': 8,
    'top_k': 3,
    'enc_in': 4
}
model = Model(configs)

df = pd.read_csv('/home/sunyongqian/liuheng/aiops-scwarn/data/sc/yid/29836/train_kpi.csv')
input_string = df.columns
for col_index in range(1, len(input_string)):

    input_data = df.iloc[:,col_index]#按需求读取数据
    input_data = torch.tensor(input_data.values).view(1, len(input_data), 1) #转换成默认的形状

    prompt = model.forecast(input_data)
    with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/output.txt', 'a') as file:
        file.write(str(col_index) + ': ')
        for p in prompt:
            file.write(str(p))
        file.write('指标名称：'+input_string[col_index] + '\n') # name
print('write finish in to output.txt')
# print("prompt:")
# for p in prompt:
#     print(p)

# input_data = torch.randn(2, 5, 3) 这是默认的输入形状
# input_data = torch.randn(1, 20, 1)
# torch.randn 生成一个张量，其中的元素是从标准正态分布（均值为0，方差为1）中随机采样的。
# 参数 (2, 5, 3) 指定了生成张量的形状，具体含义如下：
    # 第一个维度大小为 2，表示批处理中有两个样本。
    # 第二个维度大小为 5，表示每个样本有 5 个时间步的数据。
    # 第三个维度大小为 3，表示每个时间步有 3 个特征。
    # 因此，生成的 input_data 张量表示了一个批处理中包含两个样本，每个样本有 5 个时间步的数据，每个时间步有 3 个特征。

# 绘图
# input_data_flat = input_data.view(-1)
# plt.plot(input_data_flat.numpy())
# plt.title('Flattened Input Data')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.savefig('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/input_data_flat_plot.png')
# plt.show()
