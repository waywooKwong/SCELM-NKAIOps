import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
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
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        
    # Long类型数据不支持torch.mean()和torch.var()方法
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            x=x.float()
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    
    # def _get_statistics(self, x):
    #     dim2reduce = tuple(range(1, x.ndim - 1))
    #     if self.subtract_last:
    #         self.last = x[:, -1, :].unsqueeze(1)
    #     else:
    #         if x.dtype == torch.float32:  # Check if the input is float
    #             self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
    #             self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    #         elif x.dtype == torch.int64:  # Check if the input is long
    #             self.mean = torch.mean(x.float(), dim=dim2reduce, keepdim=True).detach()
    #             self.stdev = torch.sqrt(torch.var(x.float(), dim=dim2reduce, keepdim=True) + self.eps).detach()
    #         else:
    #             raise ValueError("Unsupported data type. Only float32 and int64 are supported.")


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
