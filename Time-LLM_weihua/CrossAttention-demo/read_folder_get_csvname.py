import os

# 指定文件夹路径
folder_path = '/home/sunyongqian/liuheng/aiops-scwarn/result_json_and_csv——29811/result_csv/29811'  # 替换成你的文件夹路径

# 存储前缀的集合
prefix_set = set()

# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    # 确保文件是以.csv结尾的
    if file_name.endswith('.csv'):
        # 分割文件名，获取前缀部分
        prefix = file_name.split('_')[0]
        if prefix.endswith('.csv'):
            prefix_set.add(prefix[:-4])
        else:
            # 否则，将前缀添加到集合中
            prefix_set.add(prefix)
# 将集合转换为列表
prefix_list = list(prefix_set)

if "overall" in prefix_list:
    # 如果存在，就删除它
    prefix_list.remove("overall")

# 打印删除后的列表
print(len(prefix_list))




