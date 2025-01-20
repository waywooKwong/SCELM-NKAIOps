import torch
import torch.nn as nn
from normalize import Normalize

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.description = configs['description']
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.top_k = configs['top_k']
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
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values

        # 保留四位小数
        min_values = torch.round(min_values * 10000) / 10000
        max_values = torch.round(max_values * 10000) / 10000
        medians = torch.round(medians * 10000) / 10000

        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        prompt = []
        num_list = []
        for b in range(x_enc.shape[0]):
            min_value = float(min_values[b].item())
            max_value = float(max_values[b].item())
            median = float(medians[b].item())

            num_list.append([min_value, max_value, median])
            # print('num_list[0]',num_list[0])
            # print('num_list',num_list[0][0],' ',num_list[0][1],num_list[0][1])

            prompt_ = (
                f"最小值: {min_value}; "
                f"最大值: {max_value}; "
                f"均值: {median}; "
                f"整体的趋势: {'上升' if trends[b] > 0 else '下降'}; "
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        return prompt, num_list

