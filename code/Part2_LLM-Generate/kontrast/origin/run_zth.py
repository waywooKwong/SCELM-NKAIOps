# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import langchain.chains.history_aware_retriever as test_retriever

# from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from module import *

"""
1. 新建cmd, 启动 ollama: ollama serve
2. 新建cmd, 启动 redis: 
    先查看是否已经启动 redis: redis-cli 
    如果没有启动: redis-server --protected-mode no
3. 运行python函数

"""

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
is_ablation = False # 1. 是否消融
model_name = "qwen2.5" # 2. 切换模型 qwen2.5/llama3.1/gemma2
add_to_redis = False  # 3. 是否添加到数据库 "True" or "False"
base_folder_path = "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/areaText/yunzhanghu"

real_status = "failure"  # "normal" or "failure"
chat_model = ChatOllama(model=model_name)

"""
r1 是历史经验加载的数据库
r2 是插入历史经验的数据库

db = 0 云账户
db = 2 kontrast-gemma2
db = 3 yunzhanghu-llama3.1
db = 4 yunzhanghu-qwen2
db = 5 yuzhanghu-qwen2.5
"""

import redis

r1 = redis.Redis(host="localhost", port=6379, db=0)
r2 = redis.Redis(host="localhost", port=6379, db=5)


ids = []
for filename in os.listdir(base_folder_path):
    match = re.match(r"areaText_(\d+)_v4_English\.txt", filename)
    if match:
        ids.append(match.group(1))
print("提取出的ID列表: ", ids)


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "/home/zhengtinghua/kuangweihua/ollama/models/all-MiniLM-L6-v2"
)



### 向量数据库 ###
embeddings = HuggingFaceEmbeddings(
    model_name="kuangweihua/ollama/models/m3e-base", model_kwargs={"device": "cpu"}
)

print("Embedding from huggingface:", embeddings.model_name)

# 定义的分类标准
combine_classification_path = "/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/combine_classification_v2.json"
with open(combine_classification_path, "r", encoding="utf-8") as file:
    combine_classification = file.read()

for id in ids:

    if str(id) > "20000":
        real_status = "normal"
    else:
        real_status = "failure"

    file_name = f"areaText_{id}_v4_English.txt"
    file_path = os.path.join(base_folder_path, file_name)
    print("file name:", file_name)

    docs = []
    ### 加载异常指标名称
    cluster_json_path = (
        "/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/cluster-gemma2.json"
    )
    anomaly_csv_path =  f"/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/k1_csv_anomaly/csv_anomaly_{id}.csv"
        
    anomaly_cluster, anomaly_metrics = cluster_check_list_with_metrics(
        cluster_json_path=cluster_json_path, anomaly_csv_path=anomaly_csv_path
    )
    anomaly_cluster_str = str(anomaly_cluster)

    # 得到了当前文本对应的指标聚类 anomaly_cluster 作为 key
    # 在数据库中匹配，出现对应的内容，加载到LLM中；没有严格匹配到的，加载相似度最相近的两个key
    from document import Document

    target_key = str("cluster:" + anomaly_cluster_str)
    value_set = r1.smembers(target_key)
    doc_from_redis = []

    # 如果匹配到
    if value_set:
        print("matched value_set")
        # first_value = next(iter(value_set), None)  # 使用 next() 获取第一个元素
        # if first_value:
        #     first_value = first_value.decode("utf-8")
        #     doc_from_redis.append(first_value)
        for value in value_set:
            value = value.decode("utf-8")
            doc_from_redis.append(value)

    # 如果没有匹配到
    else:
        # !! 这里可以有预警提示，交给运维人员处理 - 240909
        print("none value_set")
        # 获取所有键，然后提取最相近的两个key值
        keys = r1.keys("cluster:*")

        similarity_scores = []

        # 对每个 key 进行相似度计算
        embeddings2 = model.encode(target_key)
        for key in keys:
            key = key.decode("utf-8")
            print("key:", key)

            embeddings1 = model.encode(key)
            # 计算相似度分数
            similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()
            print("solution similarity_score:", similarity_score)

            # 存储相似度和对应的 key
            similarity_scores.append((key, similarity_score))

        # 根据相似度分数进行排序，提取前两个最高的 key
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # 提取相似度最高的两个 key
        similarity_key = [item[0] for item in similarity_scores[:2]]
        print("Top 2 most similar keys:", similarity_key)

        for key in similarity_key:
            values = r1.smembers(key)
            # first_value = next(iter(value_set), None)  # 使用 next() 获取第一个元素
            # if first_value:
            #     first_value = first_value.decode("utf-8")
            #     doc_from_redis.append(first_value)
            for value in values:
                value = value.decode("utf-8")
                doc_from_redis.append(value[0])
                print("value:", value)

    root_cause_from_redis = r1.smembers("root:" + anomaly_cluster_str)
    # 解码所有成员
    root_cause_from_redis = [item.decode("utf-8") for item in root_cause_from_redis]

    if root_cause_from_redis:  # 检查是否不为空
        print("root_cause_from_redis[0]:", root_cause_from_redis[0])
        root_cause_from_redis = root_cause_from_redis[0]  # 取第一个元素


    if real_status == "normal":  # "normal" or "failure"
        classification_response = "Expected Software Changes"
        
    

    elif real_status == "failure":
        from document import Document

        classification_prompt = """
        现在我对出现的异常指标聚类, 得到聚类结果是: {anomaly_cluster_str},
        请你根据指标对应的实际物理含义, 对应在下述变更类型分类中的symptoms, 给这一组聚类结果划分最契合的一条变更分类,
        历史经验得出的分类结果是：{root_cause_from_redis},
        变更分类列表是:{combine_classification},
        只给出最终的格式结果, 不需要任何多余的解释
        {format_instruction}
        """

        response_schema = [
            ResponseSchema(
                name="item", description="one item in List, according to symptoms"
            ),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas=response_schema
        )
        format_instruction = output_parser.get_format_instructions()
        prompt_template = PromptTemplate.from_template(
            template=classification_prompt,
            partial_variables={"format_instruction": format_instruction},
        )

        prompt_str_input = prompt_template.format(
            anomaly_cluster_str=anomaly_cluster_str,
            combine_classification=combine_classification,
            root_cause_from_redis = root_cause_from_redis,
        )

        classification_response: AIMessage = chat_model.invoke(
            input=prompt_str_input
        )
        classification_response = str(classification_response.content)

        ft = extract_between_quotes(
            classification_response, start_str='"item": "', end_str='"'
        )
        ft_output_file = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/yunzhanghu/{timestamp}/{id}/ft.csv"
        # 检查文件是否存在
        file_exists = os.path.exists(ft_output_file)
        os.makedirs(os.path.dirname(ft_output_file), exist_ok=True)
        with open(ft_output_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["id", "ft"])
            writer.writerow([id, ft])

        # print("classification_response:", classification_response)
    # root_cause_from_redis = r1.smembers("root:" + anomaly_cluster_str)
    # # 解码所有成员
    # root_cause_from_redis = [item.decode("utf-8") for item in root_cause_from_redis]

    # if root_cause_from_redis:  # 检查是否不为空
    #     print("root_cause_from_redis[0]:", root_cause_from_redis[0])
    #     root_cause_from_redis = root_cause_from_redis[0]  # 取第一个元素

    
    embeddings1 = model.encode(str(classification_response))
    embeddings2 = model.encode(str(root_cause_from_redis))

    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    print("root reason1:(llm judge classification)", str(classification_response))
    print(
        "root reason2:(get from redis according cluster)",
        str(root_cause_from_redis),
    )

    if add_to_redis == True:
        print("====")
        r2.sadd(str("root:"+ anomaly_cluster_str), classification_response)
        print("root:", anomaly_cluster_str, "\n", classification_response)
        print("====")

    docs = []  # 预置到向量库中的文本
    for snippet in doc_from_redis:
        # snippet_str = snippet.decode("utf-8")  # 解码字节数据为字符串
        snippet_str = snippet
        docs.append(
            Document(
                page_content=snippet_str, metadata={"cluster": anomaly_cluster_str}
            )
        )  # 创建 Document 对象并添加到 docs 列表

    # 20240825 test
    docs.append(
        Document(
            page_content=classification_response,
            metadata={"cluster": anomaly_cluster_str},
        )
    )
    
    if is_ablation == True:
        docs = []
        docs.append(
        Document(
            page_content="none",
            metadata={"cluster": "none"},
        ))

    from langchain_community.vectorstores import Qdrant

    vector = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="reason from cluster",
    )

    vector_retriever = vector.as_retriever()

    history_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", """需求的描述是{input}"""),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.",
            ),
        ]
    )
    # history_chain = create_history_aware_retriever(
    #     llm=chat_model, prompt=history_prompt, retriever=vector_retriever
    # )
    history_chain = test_retriever.create_history_aware_retriever(
        llm=chat_model,
        retriever=vector_retriever,
        prompt=history_prompt,
    )

    doc_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            """
            您现在是一个企业日常进行变更维护的工程师，\n
            你擅长通过读取记录变更服务相关指标判断变更是否导致发生异常并提供可能的修复措施。\n
            我现在有一些记录变更服务指标数据的领域文本文件信息，请判断这次变更是否符合预期。\n
            {context}
            """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    documents_chain = create_stuff_documents_chain(chat_model, doc_prompt)

    retriever_chain = create_retrieval_chain(history_chain, documents_chain)
    chat_history = []
    ################################################ 功能实现区域 ####################################################
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as file:
        target_doc = file.read()

    fact_extraction_str_01 = """
    You are an expert in software change analysis, specifically tasked with identifying and classifying changes as either normal or failure. 
    Below is a cluster of metrics identified from a recent change. 
    Your job is to classify this change based on the symptoms and provide a clear conclusion.
    The current cluster of anomaly metrics is: {combine_classification}.
    
    Following is your analysis content,
    1. Attention: Considering this is injected data from virtual platform, 
    "Types of single kpi anomalies related to overall anomalies" are strongly connected with "failure" Change Type.
    
    2. Attention: part "Types of single kpi anomalies related to overall anomalies" 
    in content are considered indicator data which exceed the threshold, which is considered abnormal
    also recorded new log structure pattern, referred to as the "newly generated log pattern." 
    
    Use these criteria to make an accurate classification :
        1. **Pre-Approved Process Adherence:** Determine if the change followed a pre-approved and documented change management process.
        2. **Post-Change Metrics Comparison:** Compare the key performance indicators (KPIs) from pre-change and post-change. The KPIs include system latency, error rates, uptime, and throughput.
        3. **Impact Assessment:**
        - **Normal Change:** If the KPIs remain within historical norms and there is no significant degradation in system performance.
        - **Failure Change:** If KPIs show significant deviations indicating disruptions, increased error rates, or decreased system availability.
        4. **Anomaly Detection:** Identify any anomalies flagged by the monitoring system which might suggest the change deviated from expected behavior patterns.

    Your analysis should follow this format:
    Attention: not including KPI named with 'build' in "Top 5 abnormal kpi".
    - **Change Type**: Specify 'normal' or 'failure'.
    - **Top 5 abnormal kpi**: If faliure, give the top 5 kpi that are primarily responsible. 
        not including KPI named with 'build' in list
    - **Reason**: Provide a reason for your classification, based on the metrics and symptoms.
    - **Solution**: If classified as a failure, suggest a solution to address the issues.
    
    
    {text_input}
    please strictly answer in format: {format_instruction}
    """

    if real_status == "failure":
        change_type_discription = f"""
        Specify 'normal' or 'failure',
        after data inspection,
        the indicator data exceeds the threshold or has anomalies,
        """ 
    #   check the anomalies really reveal 'failure' or is 'normal',
    #   having anomalies not necessarily related to 'failure',
    #   if anomalies not obvious, judge as 'normal'
    elif real_status == "normal":
        change_type_discription = """
        Specify 'normal' or 'failure',
        This set of data has been tested, 
        and all indicator data does not exceed the threshold, which is considered normal
        """

    if is_ablation == True:
        change_type_discription = """
        Specify 'normal' or 'failure'
        """
    # change_type_discription="'failure' most associated with 'SCWARN algorithm identifies anomalies at the following timestamps' and 'single kpi anomalies related to overall anomalies'"

    response_schema_01 = [
        ResponseSchema(name="change_type", description=change_type_discription),
        ResponseSchema(
            name="Top 5 abnormal kpi",
            description="""If faliure, give the top 5 kpi that are primarily responsible.
            getting kpis from retriever docs.Attention: not including KPI named with 'build' in "Top 5 abnormal kpi". """,
        ),
        ResponseSchema(
            name="reason",
            description="Provide a reason for your classification, based on the metrics and symptoms.",
        ),
        ResponseSchema(
            name="solution",
            description="If classified as a failure, suggest a solution to address the issues.",
        ),
    ]

    output_parser_01 = StructuredOutputParser.from_response_schemas(
        response_schemas=response_schema_01
    )

    format_instruction_01 = output_parser_01.get_format_instructions()
    prompt_template_01 = PromptTemplate.from_template(
        template=fact_extraction_str_01,
        partial_variables={"format_instruction": format_instruction_01},
    )

    prompt_str_input_01 = prompt_template_01.format(
        text_input=target_doc, combine_classification=classification_response
    )
    output_completion_01: AIMessage = retriever_chain.invoke(
        {"input": prompt_str_input_01, "chat_history": chat_history}
    )
    content = output_completion_01["answer"]
    print("output_completion_01.content:", content)
    os.makedirs(
        f"kuangweihua/ollama/mydocuments/cluster/{model_name}/yunzhanghu/{timestamp}/{id}",
        exist_ok=True,
    )
    output_completion_file_path = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/yunzhanghu/{timestamp}/{id}/output_completion.txt"
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"\n{file_name}\n")
        file.write(f"root_cause_from_redis: {str(root_cause_from_redis)}\n")
        file.write(f"{classification_response}\n")
        file.write(f"{content}\n")
    print(f"{file_name}已经写入文本")
    
    change_type = extract_between_quotes(
        content, start_str='"change_type": "', end_str='"'
    )
    print("change_type:", change_type)

    # 2024/11/24 尽量将 top5_kpi 写入一个csv中
    top5_kpi = extract_between_quotes(
        content, start_str='"Top 5 abnormal kpi": "', end_str='",'
    )
    print("Top 5 abnormal kpi:", change_type)
    top5_output_file = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/yunzhanghu/{timestamp}/{id}/top5_kpi.csv"
    # 打开文件，准备写入
    file_exists = os.path.exists(top5_output_file)

    os.makedirs(os.path.dirname(top5_output_file), exist_ok=True)
    top5_data = (id, top5_kpi)
    with open(top5_output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            # 写入表头
            writer.writerow(["id", "kpi1", "kpi2", "kpi3", "kpi4", "kpi5"])

        # 写入数据
        id_value, str_value = top5_data
        # 将字符串按','拆分，并补充空白字段
        kpis = str_value.split(",")
        kpis += [""] * (5 - len(kpis))  # 如果不够5个kpi，补充空字符串
        writer.writerow([id_value] + kpis)

    ## 定义内容和聚类标签 并插入 redis
    content = f"""{chat_model}\n""" + content

    """
    提取 reason 进行相似度匹配
    """
    
    print("doc_from_redis[0]:",doc_from_redis[0])
    test_doc = str(doc_from_redis[0])
    
    # reason1 = (
    #     "this is root reason:"
    #     + str(classification_response)
    #     + "this is detailed reason:"
    #     + extract_between_quotes(str(test_doc))
    # )
    print("====")
    reason1 = (
        extract_between_quotes(str(test_doc))
    )
    print("====")
    # print("reason1:", reason1)
    # reason2 = (
    #     "this is root reason:"
    #     + str(root_cause_from_redis)
    #     + "this is detailed reason:"
    #     + extract_between_quotes(content)
    # )
    reason2 = (
        extract_between_quotes(content)
    )
    print("reason2:", reason2)

    embeddings1 = model.encode(reason1)
    embeddings2 = model.encode(reason2)

    reason_similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    print("reason similarity_score:", reason_similarity_score)
    print("====")
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"reason similarity score:{reason_similarity_score}\n")

    """
    提取 solution 进行相似度匹配
    """
    solution1 = (
        "this is root reason:"
        + str(classification_response)
        + "this is detailed solution:"
        + extract_between_quotes(
            str(doc_from_redis), start_str='"solution"', end_str="}"
        )
    )
    print("====")
    print("solution1:", reason1)
    solution2 = (
        "this is root reason:"
        + str(root_cause_from_redis)
        + "this is detailed solution:"
        + extract_between_quotes(content, start_str='"solution"', end_str="}")
    )
    print("solution2:", reason2)
    embeddings1 = model.encode(solution1)
    embeddings2 = model.encode(solution2)

    solution_similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    print("solution similarity_score:", solution_similarity_score)
    print("====")
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"solution similarity score:{solution_similarity_score}\n")
    # reason_similarity_score = 0
    # solution_similarity_score = 0

    # real_status = "0"

    change_result_csv_file = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/yunzhanghu/{timestamp}/{id}/change_result.csv"
    write_to_csv(
        id,
        change_type,
        real_status,
        reason_similarity_score,
        solution_similarity_score,
        csv_file=change_result_csv_file,
    )

    cluster_label = anomaly_cluster
    if add_to_redis == True:
        r2.sadd(str("cluster:"+ str(cluster_label)), content)

        print("====")
        print(f"redis insert-\nkey:{cluster_label},\nvalue:{content}")
        print("====")

# 计算 F1 分数，并追加到原文本中
# change_result_csv_file = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/kontrast/{timestamp}/{kpi}/change_result.csv"
# output_completion_file_path = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/kontrast/{timestamp}/{kpi}/output_completion.txt"
calculate_f1_and_average_scores(
    csv_file=change_result_csv_file, txt_file=output_completion_file_path
)
