# 执行实验
python3 -m run --publish_date '20240620' --prometheus_address '172.16.17.114:19192' --service 'geass' --sc_id '32207' --train_date '2024-06-20 18:29:13' --task_count 5 --step 120 --timeout 10000 --train_duration 432000 --detection_duration 86400 --predict_interval '30'
# 实验结果打包
tar -zcvf result_json_and_csv.tar.gz result_json_and_csv

# 移动到文件夹
mv result_json_and_csv.tar.gz /tmp/

# 删除机器中的实验记录
rm -rf result_json_and_csv/


