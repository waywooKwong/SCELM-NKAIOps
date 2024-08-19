# -*- coding: utf-8 -*-
import os
import re
import json
import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate

model_name = "qwen2"
chat_model = ChatOllama(model=model_name)
# 读取 output 的路径，英文模型 english ，中文模型 chinese]
folder_path = "/home/zhengtinghua/kuangweihua/ollama/areaText/areaText_v5/normal_chinese"
ids = []
for filename in os.listdir(folder_path):
    match = re.match(r'areaText_(\d+)_v4\.txt', filename)
    if match:
        ids.append(match.group(1))
print("提取出的ID列表: ", ids)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = f"scwarn-{model_name}"  # 项目名称

def cluster_check_list(cluster_json_path,anomaly_csv_path):
    ### 1. 加载聚类的列表
    with open(cluster_json_path, 'r') as file: # 从JSON文件中加载数据
        data = json.load(file)
    clusters = {cluster['name']: cluster['metrics'] for cluster in data['clusters']}

    ### 2. 加载异常指标名称
    df = pd.read_csv(anomaly_csv_path)
    anomaly_metrics = df[df['异常情况'] == '出现异常']['指标名称'].tolist()
    # print("anomaly metrics:",anomaly_metrics)
    best_cluster = []

    ### 3. 遍历 anomaly_metrics 列表对每个 metric 进行聚类划分
    for new_metric in anomaly_metrics:
        all_metrics = [metric for metrics in clusters.values() for metric in metrics] + [new_metric]     # 将所有metrics转换为向量
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_metrics)

        new_metric_vector = tfidf_matrix[-1]     # 计算新metric与每个cluster的相似性
        similarities = {}
        start_idx = 0
        for cluster_name, metrics in clusters.items():
            cluster_vectors = tfidf_matrix[start_idx:start_idx + len(metrics)]
            similarity = cosine_similarity(new_metric_vector, cluster_vectors).mean()
            similarities[cluster_name] = similarity
            start_idx += len(metrics)

        best_match_cluster = max(similarities, key=similarities.get)     # 找出相似度最高的类
        if best_match_cluster not in best_cluster:     # 如果新metric还没有添加到best_cluster中，则添加
            best_cluster.append(best_match_cluster)

    # 输出best_cluster列表
    print(f"anomaly metric 划分出的聚类列表: {best_cluster}")
    return best_cluster

import redis
r = redis.Redis(host='localhost', port=6379, db=0)

### 向量数据库 ###
embeddings = HuggingFaceEmbeddings(model_name="kuangweihua/ollama/models/m3e-base", model_kwargs={"device": "cuda"})
print("Embedding from huggingface:", embeddings.model_name)


for id in ids:
    file_name = f"areaText_{id}_v4.txt"
    file_path = os.path.join(folder_path, file_name)
    
    docs = []
    ### 加载异常指标名称
    ### redis
    cluster_json_path = '/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/cluster-gemma2.json'
    anomaly_csv_path = os.path.join("/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/k1_csv_anomaly", f"csv_anomaly_{id}.csv")
    anomaly_cluster = cluster_check_list(cluster_json_path=cluster_json_path, anomaly_csv_path=anomaly_csv_path)
    anomaly_cluster_str = str(anomaly_cluster)
    
    from document import Document
    doc_from_redis = r.smembers(anomaly_cluster_str)
    # print("====")
    # print("doc_from_redis:",doc_from_redis)
    # print("====")
    
    for snippet in doc_from_redis:
        snippet_str = snippet.decode('utf-8')  # 解码字节数据为字符串
        docs.append(Document(page_content=snippet_str, metadata={"cluster": anomaly_cluster_str}))  # 创建 Document 对象并添加到 docs 列表
    
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
            ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.",
            ),
        ]
    )
    history_chain = create_history_aware_retriever(
        llm=chat_model, prompt=history_prompt, retriever=vector_retriever
    )

    doc_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","""
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
    with open(file_path, 'r', encoding='utf-8') as file:
        target_doc = file.read()

    fact_extraction_str_01 = """
        您现在是一个企业日常进行变更维护的工程师，
        你擅长通过读取记录变更服务相关指标判断变更是否导致发生异常并提供可能的修复措施。
        我现在有一些记录变更服务指标数据的领域文本文件信息，请判断这次变更是否符合预期。
        
        分析文本时的重点:
        如果 kpi 出现了 still in abnormal 的异常, 结合指标对应的数值整体变化趋势描述,
        如果 kpi 没出现异常, 趋势变化不足以作为异常变化的判断依据
        1. 分析异常时重点关注下述部分涉及到的 kpi 是Still in abnormal state 还是 Recover to normal state
        Types of single kpi anomalies related to overall anomalies (single kpi anomalies not related to overall anomalies are not output):
        2. 不要把数值的整体趋势变化作为重要参考依据, 只作为参考
        
        下面是需要你分析的文本内容: {text_input}
        
        please strictly answer in format: {format_instruction}
        对format的补充:
        1. "change_type": 直接给出 true/failure 的判断
        2. "reason": 结合数据进行分析,如果得到 failure, 判断原因,
        3. "solution": 如果得到 failure, 给出对应的解决方案,
        
        Let's work this out in a step by step way to be sure we have the right answer.
    """

    response_schema_01 = [
        ResponseSchema(name="change_type", description="normal/failure"),
        ResponseSchema(name="reason", description="Reason for the conclusion according to analysis"),
        ResponseSchema(name="solution", description="Provide targeted solution"),
    ]

    output_parser_01 = StructuredOutputParser.from_response_schemas(
        response_schemas=response_schema_01
    )

    format_instruction_01 = output_parser_01.get_format_instructions()
    prompt_template_01 = PromptTemplate.from_template(
        template=fact_extraction_str_01,
        partial_variables={"format_instruction": format_instruction_01},
    )

    prompt_str_input_01 = prompt_template_01.format(text_input=target_doc)
    output_completion_01: AIMessage = chat_model.invoke(input=prompt_str_input_01)
    print("output_completion_01.content:",output_completion_01.content)
    os.makedirs(f"kuangweihua/ollama/mydocuments/cluster/{model_name}/{timestamp}",exist_ok=True)
    output_completion_file_path = f"kuangweihua/ollama/mydocuments/cluster/{model_name}/{timestamp}/output_completion.txt"
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"\n{file_name}\n")
        file.write(f"{output_completion_01.content}\n")
    print(f"{file_name}已经写入文本")

    ## 定义内容和聚类标签 并插入 redis
    content = f"""{chat_model}\n"""+output_completion_01.content
    cluster_label = anomaly_cluster
    r.sadd(str(cluster_label), content)
    print("====")
    print(f"redis insert-\nkey:{cluster_label},\nvalue:{content}")
    print("====")