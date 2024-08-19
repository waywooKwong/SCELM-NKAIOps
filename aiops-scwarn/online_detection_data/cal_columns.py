import csv

# 打开CSV文件
with open('25969/test_kpi.csv', 'r') as file:
    # 使用逗号作为分隔符创建CSV阅读器
    reader = csv.reader(file, delimiter=',')
    
    # 读取文件的第一行，通常是列标题行
    first_row = next(reader)
    
    # 计算列数
    num_columns = len(first_row)
    
    print(f'CSV文件中有 {num_columns} 列')

