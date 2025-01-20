import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取CSV文件
df = pd.read_csv('/home/sunyongqian/liuheng/Time-LLM/weihua/AreaText/data/k1_csv_anomaly/csv_anomaly_10015.csv')

# 第一步：提取“指标名称”的所有取值作为一个“metrics”列表
metrics = df['指标名称'].unique().tolist()

# 第二步：对“metric”列表中涉及的指标，根据其物理意义进行聚类
# 假设我们使用KMeans进行文本聚类，可以使用TfidfVectorizer提取文本特征
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(metrics)

# 假设我们要分成3类，可以调整
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 获取聚类结果
clusters = kmeans.labels_
cluster_dict = {i: [] for i in range(3)}

for metric, cluster in zip(metrics, clusters):
    cluster_dict[cluster].append(metric)

# 将聚类结果保存为 cluster 列表
cluster = [cluster_dict[i] for i in range(len(cluster_dict))]

# 第三步：提取“异常情况”为“出现异常”的指标
anomaly_metrics = df[df['异常情况'] == '出现异常']['指标名称'].unique().tolist()

# 根据聚类结果对异常指标进行划分
cluster_result = {i: [] for i in range(len(cluster_dict))}

for anomaly_metric in anomaly_metrics:
    for cluster_id, metrics_list in cluster_dict.items():
        if anomaly_metric in metrics_list:
            cluster_result[cluster_id].append(anomaly_metric)

# 去除重复
cluster_result = {k: list(set(v)) for k, v in cluster_result.items()}

# 输出结果
print("Metrics List:", metrics)
print("Clusters:", cluster)
print("Anomaly Metrics:", anomaly_metrics)
print("Cluster Result:", cluster_result)
