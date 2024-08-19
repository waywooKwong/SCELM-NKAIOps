import torch
import torch.nn as nn
# from torch.nn import Normalize

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.description = configs['description']
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.top_k = configs['top_k']
        #normalize_layers 是 Class Normalize 中的一个函数，但是torch.nn中也有一个Normalize库
        self.normalize_layers = Normalize(configs['enc_in'], affine=False)

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def forecast(self, x_enc):

        # 这里需要自行实现 normalize_layers 方法
        x_enc = self.normalize_layers(x_enc, 'norm')

        # 对输入数据进行形状变换，将其形状调整为 (B * N, T, 1)
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 计算输入数据的最小值、最大值和中位数
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values

        # 调用 calcute_lags 方法计算输入数据的滞后值
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # 构建提示信息
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"Dataset description: {self.description} "  
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        # 将输入数据形状恢复为原始形状 (B, T, N)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        return prompt
'''
这是演示Prompt是啥样：

提示信息：
Dataset description: Sample dataset
Task description: Forecast the next 10 steps given the previous 5 steps information
Input statistics: min value 0.1, max value 0.9, median value 0.5, the trend of input is upward, top 5 lags are : [0.1, 0.2, 0.3, 0.4, 0.5]

Prompt：
prompt = [
    "Dataset description: Sample dataset Task description: Forecast the next 10 steps given the previous 5 steps information Input statistics: min value 0.1, max value 0.9, median value 0.5, the trend of input is upward, top 5 lags are : [0.1, 0.2, 0.3, 0.4, 0.5]",
    "Dataset description: Another dataset Task description: Forecast the next 8 steps given the previous 4 steps information Input statistics: min value 10, max value 100, median value 50, the trend of input is downward, top 5 lags are : [10, 20, 30, 40, 50]"
]
'''

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
'''
这段代码定义了一个自定义的归一化层 Normalize，它可以在神经网络中用来执行特征归一化（Normalization）和反归一化（Denormalization）操作。让我解释一下主要部分：

__init__ 方法：这个方法用于初始化归一化层的参数。参数包括 num_features，表示特征或通道的数量；eps，用于数值稳定性的小值；affine，一个布尔值，表示是否使用可学习的仿射参数；subtract_last，一个布尔值，表示是否减去最后一个值；non_norm，一个布尔值，表示是否对输入进行规范化。
forward 方法：这个方法定义了归一化层的前向计算逻辑。它接受输入 x 和模式 mode，根据模式调用 _normalize 或 _denormalize 方法。
_init_params 方法：如果 affine 参数为 True，这个方法会初始化仿射参数 affine_weight 和 affine_bias。
_get_statistics 方法：这个方法用于计算输入张量 x 的统计信息，包括均值和标准差。
_normalize 方法：这个方法执行归一化操作，减去均值、除以标准差，并根据 affine 参数进行仿射变换。
_denormalize 方法：这个方法执行反归一化操作，根据 affine 参数逆变换，并乘以标准差，最后加回均值。
这个归一化层提供了灵活性，可以根据需求选择是否使用仿射参数，以及在训练时是否应用归一化操作。
'''

configs = {
    'description': 'Sample dataset',#数据集的描述或名称
    'pred_len': 10,#预测的时间跨度长度
    'seq_len': 8,#输入序列的长度
    'top_k': 3,#在 calcute_lags 方法中计算滞后值时要考虑的前 k 个滞后值的数量。它与 torch.topk 函数一起使用，根据一定的标准选择前 k 个滞后值。
    'enc_in': 4 #输入数据的输入维度
}
# 实例化模型
model = Model(configs)


# 构造输入数据，假设输入数据的形状为 (batch_size, seq_len, input_dim)
# 这里为了演示，假设 batch_size=2, seq_len=5, input_dim=3
input_data = torch.randn(2, 5, 3)  # 生成随机输入数据

# 调用 forecast 方法生成提示信息
prompt = model.forecast(input_data)

# 打印提示信息
for p in prompt:
    print(p)