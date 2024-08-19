import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("/home/sunyongqian/liuheng/Time-LLM/weihua/LSTM_try2/data.csv")
print(df.head())  # 打印前几行数据以确保成功读取文件

# 绘制图像
# plt.plot(df['timestamp'], df['value'])
plt.plot(df['timestamp'][:1000], df['value'][:1000])
plt.xlabel('Timestamp')
plt.ylabel('MLSTM')
plt.title('MLSTM over time')
plt.xticks(rotation=45)  # 旋转x轴标签以避免重叠
plt.tight_layout()  # 调整布局以确保标签不重叠
plt.show()
plt.savefig("/home/sunyongqian/liuheng/Time-LLM/weihua/output.png")