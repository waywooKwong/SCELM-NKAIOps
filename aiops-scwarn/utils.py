# -*-coding:utf-8-*-
import numpy as np
import math
'''
around(arr,decimals=?)？表示保留多少位小数
'''

mul_kpi = [[1,1,1,1,1,0.99,1,1,1,1,1,1,1,1, 3], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.8]]

class NormMethod:
    @classmethod
    def Min_MaxNorm(cls, arr, x_max, x_min, x_mean, x_std):
        arr = np.around(((arr - x_min) / (x_max - x_min)), decimals=8)
        # print("min_max 标准化:{}".format(arr))
        return arr

    @classmethod
    def Z_ScoreNorm(cls, arr, x_max, x_min, x_mean, x_std):
        arr = np.around((arr - x_mean) / x_std, decimals=8)
        return arr
        # print("Z_Score标准化:{}".format(arr))

    @classmethod
    def Decimal_ScalingNorm(cls, arr, x_max, x_min, x_mean, x_std):
        power = 1
        maxValue = x_max
        while maxValue / 10 >= 1.0:
            power += 1
            maxValue /= 10
        arr = np.around((arr / pow(10, power)), decimals=8)
        return arr
        # print("小数定标标准化:{}".format(arr))

    @classmethod
    def MeanNorm(cls, arr, x_max, x_min, x_mean, x_std):
        first_arr = np.around((arr-x_mean) / (x_max - x_min), decimals=8)
        second_arr = np.around((arr - x_mean)/x_max, decimals=8)
        return first_arr, second_arr
        # print("均值归一法：\n公式一:{}\n公式二:{}".format(first_arr, second_arr))

    @classmethod
    def Vector(cls, arr):
        arr = np.around((arr/arr.sum()), decimals=8)
        return arr
        # print("向量归一法:{}".format(arr))
    
    @classmethod
    def exponeential(cls, arr, x_max, x_min, x_mean, x_std):
        first_arr = np.around(np.log10(arr) / np.log10(x_max), decimals=8)
        second_arr = np.around(np.exp(arr)/sum(np.exp(arr)), decimals=8)
        three_arr = np.around(1/(1+np.exp(-1*arr)), decimals=8)
        return first_arr, second_arr, three_arr
        # print("lg函数:{}\nSoftmax函数:{}\nSigmoid函数:{}\n".format(first_arr,second_arr,three_arr))

    
def select_norm_method(method="minmax"):
    run = None
    if method == "minmax":
        run = NormMethod.Min_MaxNorm
    elif method == "z_norm":
        return
    elif method == "z_norm":
        return
    elif method == "decimal_scaling_norm":
        return
    elif method == "mean_norm":
        return
    elif method == "vector_norm":
        return
    elif method == "exponeential_norm":
        return
    return run

def select_kpi(kpi)->bool:
    if kpi.count(1)/len(kpi) > 0.8:
        return True
    return False
    
def run_norm(kpi_names, mul_kpi , method = "minmax"):
    kpi_index = 0
    np_mul_kpi = []
    run = select_norm_method(method)
    for kpi in mul_kpi:
        np_kpi = np.array(kpi)
        x_max = np_kpi.max() #数组元素中的最大值
        x_min = np_kpi.min() #数组元素中的最小值
        x_mean = np_kpi.mean() # 数组元素中平均值
        x_std = np_kpi.std() #数组元素中的标准差
        if method == "minmax" and select_kpi(kpi):
            x_min = 0.0
        if 'node_network_transmit_bytes_total' in kpi_names[kpi_index]:
            x_max = 100000000  
        #x_min = 0.0
        if kpi.count(1) == len(kpi) or kpi.count(0) == len(kpi):
            arr = np_kpi
        elif x_max == x_min:
            arr = np.array([1]*len(kpi))
        elif kpi.count(100) == len(kpi):
            arr = np.array([1]*len(kpi))
        elif kpi.count(5) == len(kpi):
            arr = np.array([1]*len(kpi))
        else:
            arr = run(np_kpi, x_max, x_min, x_mean, x_std)
        np_mul_kpi.append(arr)
        kpi_index += 1
    
    train_data = np.vstack(np_mul_kpi).transpose()
    return train_data

if __name__ == "__main__":
    # print(run_norm(mul_kpi))
    run_norm(mul_kpi)
