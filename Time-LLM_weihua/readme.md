# 1 文件说明
model.py 是TimeLLM.py 删除掉有关LLM的部分
forcast.py 是单独领出来forcast以及依赖的函数
demo文件夹是把forcast拆分成单文件，运行入口是main.py

# 2 接入自己的时序数据
要接入自己的时序数据，你需要按照模型的输入要求准备数据，并在 main.py 中使用你的数据替换随机生成的示例数据。
在模型中，forecast 方法的输入参数 x_enc 表示时序数据，其形状应为 (batch_size, seq_len, input_dim)，其中：

batch_size 是批处理大小，表示同时处理的样本数量。
seq_len 是输入序列的长度，表示模型用多少个时间步的数据进行预测。
input_dim 是输入数据的特征维度，表示每个时间步的特征数量。
你需要根据自己的数据格式和特征选择合适的 batch_size、seq_len 和 input_dim。然后，将你的数据传递给模型的 forecast 方法进行预测。

下面是一个示例，假设你有一个名为 your_data 的时序数据，你可以像这样接入：
```
import torch
from model import Model
#模型配置
configs = {
    'description': 'Your dataset description',
    'pred_len': 10,  # 预测的时间跨度长度
    'seq_len': 8,    # 输入序列的长度
    'top_k': 3,      # 在 calcute_lags 方法中计算滞后值时要考虑的前 k 个滞后值的数量
    'enc_in': 4      # 输入数据的输入维度
}
#实例化模型
model = Model(configs)

#准备你的时序数据，替换示例数据
your_data = torch.randn(2, 8, 4)  # 假设你的数据形状为 (batch_size=2, seq_len=8, input_dim=4)
#调用 forecast 方法生成提示信息
prompt = model.forecast(your_data)

#打印提示信息
for p in prompt:
    print(p)
```
