import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

# 从CSV文件中加载时序数据
data = pd.read_csv("/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/res.csv")

# 假设你的时序数据在CSV文件中的一列名为"value"，将其转换为NumPy数组
time_series = data["value"].values

# 创建ruptures模型，选择算法
model = "l2"
algo = rpt.Dynp(model=model, min_size=3, jump=5)

# 拟合模型并检测突变点
algo.fit(time_series)
result = algo.predict(n_bkps=10)  # 假设我们要检测4个突变点
# 打印突变点
print("Detected changepoints:", result)

# 可视化结果
rpt.display(time_series, result)
plt.savefig("result_plot.png")  # 将结果保存为图片文件
plt.show()
