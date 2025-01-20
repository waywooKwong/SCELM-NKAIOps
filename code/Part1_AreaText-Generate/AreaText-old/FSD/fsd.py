# import matplotlib.pyplot as plt
# import ruptures as rpt

# # generate signal
# n_samples, dim, sigma = 1000, 1, 4
# n_bkps = 4  # number of breakpoints
# signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

# # detection
# algo = rpt.Pelt(model="rbf").fit(signal)
# result = algo.predict(pen=10)

# # display
# fig, axarr = rpt.display(signal, bkps, result)
# fig.suptitle('Change Point Detection using PELT Algorithm')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.savefig("/home/sunyongqian/liuheng/Time-LLM/weihua/FSD/change_point_detection.png")
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt

# 读取 CSV 文件
df = pd.read_csv("/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/res.csv")

# 假设您的 CSV 文件中有一个名为 'signal' 的列，包含信号数据
signal = df['value'].values

# 运行变化点检测
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10)

# 显示结果
fig, axarr = rpt.display(signal, [], result)
fig.suptitle('Change Point Detection using PELT Algorithm')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')

# 保存图像到指定路径
plt.savefig("/home/sunyongqian/liuheng/Time-LLM/weihua/FSD/change_point_detection.png")

# 显示图像
plt.show()
