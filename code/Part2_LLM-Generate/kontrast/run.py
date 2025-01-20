#!/usr/bin/env python3
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
import redis
from sentence_transformers import SentenceTransformer, util
from document import Document
from langchain_community.vectorstores import Qdrant

# Constants and Configuration
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
IS_ABLATION = False  # 1. Whether to perform ablation study
MODEL_NAME = "qwen2.5"  # 2. Switch model qwen2.5/llama3.1/gemma2
ADD_TO_REDIS = False  # 3. Whether to add to database "True" or "False"
BASE_FOLDER_PATH = "/home/zhengtinghua/kuangweihua/SCWarn_kontrast/areaText/yunzhanghu"

# Redis Configuration
r1 = redis.Redis(host="localhost", port=6379, db=0)  # Historical experience database
r2 = redis.Redis(host="localhost", port=6379, db=5)  # Database for inserting historical experience

# Model Initialization
chat_model = ChatOllama(model=MODEL_NAME)
model = SentenceTransformer("/home/zhengtinghua/kuangweihua/ollama/models/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="kuangweihua/ollama/models/m3e-base",
    model_kwargs={"device": "cpu"}
)

# Load classification data
combine_classification_path = "/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/combine_classification_v2.json"
with open(combine_classification_path, "r", encoding="utf-8") as file:
    combine_classification = file.read()

# Extract IDs from filenames
ids = []
for filename in os.listdir(BASE_FOLDER_PATH):
    match = re.match(r"areaText_(\d+)_v4_English\.txt", filename)
    if match:
        ids.append(match.group(1))
print("Extracted ID list: ", ids)

# Main processing loop
for id in ids:
    # Determine status based on ID
    real_status = "normal" if str(id) > "20000" else "failure"
    file_name = f"areaText_{id}_v4_English.txt"
    file_path = os.path.join(BASE_FOLDER_PATH, file_name)
    print("file name:", file_name)

    # Load anomaly data
    cluster_json_path = "/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/cluster-gemma2.json"
    anomaly_csv_path = f"/home/zhengtinghua/kuangweihua/ollama/areaText/cluster/k1_csv_anomaly/csv_anomaly_{id}.csv"
    
    anomaly_cluster, anomaly_metrics = cluster_check_list_with_metrics(
        cluster_json_path=cluster_json_path,
        anomaly_csv_path=anomaly_csv_path
    )
    anomaly_cluster_str = str(anomaly_cluster)

    # Redis data retrieval
    target_key = str("cluster:" + anomaly_cluster_str)
    value_set = r1.smembers(target_key)
    doc_from_redis = []

    if value_set:
        print("matched value_set")
        for value in value_set:
            value = value.decode("utf-8")
            doc_from_redis.append(value)
    else:
        print("none value_set")
        keys = r1.keys("cluster:*")
        similarity_scores = []

        embeddings2 = model.encode(target_key)
        for key in keys:
            key = key.decode("utf-8")
            print("key:", key)

            embeddings1 = model.encode(key)
            similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()
            print("solution similarity_score:", similarity_score)
            similarity_scores.append((key, similarity_score))

        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        similarity_key = [item[0] for item in similarity_scores[:2]]
        print("Top 2 most similar keys:", similarity_key)

        for key in similarity_key:
            values = r1.smembers(key)
            for value in values:
                value = value.decode("utf-8")
                doc_from_redis.append(value[0])
                print("value:", value)

    root_cause_from_redis = r1.smembers("root:" + anomaly_cluster_str)
    root_cause_from_redis = [item.decode("utf-8") for item in root_cause_from_redis]

    if root_cause_from_redis:
        print("root_cause_from_redis[0]:", root_cause_from_redis[0])
        root_cause_from_redis = root_cause_from_redis[0]

    if real_status == "normal":
        classification_response = "Expected Software Changes"
    elif real_status == "failure":
        classification_prompt = """
        Based on the anomaly metrics clustering results: {anomaly_cluster_str},
        Please analyze the actual physical meaning of the metrics and match them with the symptoms in the change type classification below.
        Assign the most fitting change classification for this group of clustering results.
        Historical experience classification result is: {root_cause_from_redis},
        Change classification list is: {combine_classification},
        Only provide the final formatted result, no additional explanation needed
        {format_instruction}
        """

        response_schema = [
            ResponseSchema(
                name="item",
                description="one item in List, according to symptoms"
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
            root_cause_from_redis=root_cause_from_redis,
        )

        classification_response: AIMessage = chat_model.invoke(
            input=prompt_str_input
        )
        classification_response = str(classification_response.content)

        ft = extract_between_quotes(
            classification_response,
            start_str='"item": "',
            end_str='"'
        )
        ft_output_file = f"kuangweihua/ollama/mydocuments/cluster/{MODEL_NAME}/yunzhanghu/{TIMESTAMP}/{id}/ft.csv"
        os.makedirs(os.path.dirname(ft_output_file), exist_ok=True)
        with open(ft_output_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not os.path.exists(ft_output_file):
                writer.writerow(["id", "ft"])
            writer.writerow([id, ft])

    # Vector store setup
    docs = []
    for snippet in doc_from_redis:
        docs.append(
            Document(
                page_content=snippet,
                metadata={"cluster": anomaly_cluster_str}
            )
        )

    docs.append(
        Document(
            page_content=classification_response,
            metadata={"cluster": anomaly_cluster_str},
        )
    )
    
    if IS_ABLATION:
        docs = []
        docs.append(
            Document(
                page_content="none",
                metadata={"cluster": "none"},
            )
        )

    vector = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="reason from cluster",
    )

    vector_retriever = vector.as_retriever()

    # Create retrieval chains
    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", """需求的描述是{input}"""),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.",
        ),
    ])

    history_chain = test_retriever.create_history_aware_retriever(
        llm=chat_model,
        retriever=vector_retriever,
        prompt=history_prompt,
    )

    doc_prompt = ChatPromptTemplate.from_messages([
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
    ])

    documents_chain = create_stuff_documents_chain(chat_model, doc_prompt)
    retriever_chain = create_retrieval_chain(history_chain, documents_chain)
    chat_history = []

    # Process file content
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
    
    Use these criteria to make an accurate classification:
        1. **Pre-Approved Process Adherence:** Determine if the change followed a pre-approved and documented change management process.
        2. **Post-Change Metrics Comparison:** Compare the key performance indicators (KPIs) from pre-change and post-change. The KPIs include system latency, error rates, uptime, and throughput.
        3. **Impact Assessment:**
        - **Normal Change:** If the KPIs remain within historical norms and there is no significant degradation in system performance.
        - **Failure Change:** If KPIs show significant deviations indicating disruptions, increased error rates, or decreased system availability.
        4. **Anomaly Detection:** Identify any anomalies flagged by the monitoring system which might suggest the change deviated from expected behavior patterns.

    Your analysis should follow this format:
    Attention: not including KPI named with 'build' in "Top 5 abnormal kpi".
    - **Change Type**: Specify 'normal' or 'failure'.
    - **Top 5 abnormal kpi**: If failure, give the top 5 kpi that are primarily responsible. 
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
    elif real_status == "normal":
        change_type_discription = """
        Specify 'normal' or 'failure',
        This set of data has been tested, 
        and all indicator data does not exceed the threshold, which is considered normal
        """

    if IS_ABLATION:
        change_type_discription = """
        Specify 'normal' or 'failure'
        """

    response_schema_01 = [
        ResponseSchema(
            name="change_type",
            description=change_type_discription
        ),
        ResponseSchema(
            name="Top 5 abnormal kpi",
            description="""If failure, give the top 5 kpi that are primarily responsible.
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
        text_input=target_doc,
        combine_classification=classification_response
    )
    output_completion_01 = retriever_chain.invoke(
        {"input": prompt_str_input_01, "chat_history": chat_history}
    )
    content = output_completion_01["answer"]
    print("output_completion_01.content:", content)

    # Save results
    output_dir = f"kuangweihua/ollama/mydocuments/cluster/{MODEL_NAME}/yunzhanghu/{TIMESTAMP}/{id}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_completion_file_path = f"{output_dir}/output_completion.txt"
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"\n{file_name}\n")
        file.write(f"root_cause_from_redis: {str(root_cause_from_redis)}\n")
        file.write(f"{classification_response}\n")
        file.write(f"{content}\n")
    print(f"{file_name} has been written to text")

    # Extract and save change type
    change_type = extract_between_quotes(
        content,
        start_str='"change_type": "',
        end_str='"'
    )
    print("change_type:", change_type)

    # Extract and save top 5 KPIs
    top5_kpi = extract_between_quotes(
        content,
        start_str='"Top 5 abnormal kpi": "',
        end_str='",'
    )
    print("Top 5 abnormal kpi:", change_type)
    
    top5_output_file = f"{output_dir}/top5_kpi.csv"
    file_exists = os.path.exists(top5_output_file)
    os.makedirs(os.path.dirname(top5_output_file), exist_ok=True)
    
    with open(top5_output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["id", "kpi1", "kpi2", "kpi3", "kpi4", "kpi5"])
        
        kpis = top5_kpi.split(",")
        kpis += [""] * (5 - len(kpis))
        writer.writerow([id] + kpis)

    # Calculate similarity scores
    test_doc = str(doc_from_redis[0])
    reason1 = extract_between_quotes(str(test_doc))
    reason2 = extract_between_quotes(content)
    
    embeddings1 = model.encode(reason1)
    embeddings2 = model.encode(reason2)
    reason_similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"reason similarity score:{reason_similarity_score}\n")

    solution1 = (
        "this is root reason:"
        + str(classification_response)
        + "this is detailed solution:"
        + extract_between_quotes(str(doc_from_redis), start_str='"solution"', end_str="}")
    )
    solution2 = (
        "this is root reason:"
        + str(root_cause_from_redis)
        + "this is detailed solution:"
        + extract_between_quotes(content, start_str='"solution"', end_str="}")
    )
    
    embeddings1 = model.encode(solution1)
    embeddings2 = model.encode(solution2)
    solution_similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    
    with open(output_completion_file_path, "a", encoding="utf-8") as file:
        file.write(f"solution similarity score:{solution_similarity_score}\n")

    # Save change results
    change_result_csv_file = f"{output_dir}/change_result.csv"
    write_to_csv(
        id,
        change_type,
        real_status,
        reason_similarity_score,
        solution_similarity_score,
        csv_file=change_result_csv_file,
    )
    
    # Add to Redis if configured
    if ADD_TO_REDIS:
        r2.sadd(str("cluster:" + str(anomaly_cluster)), content)
        r2.sadd(str("root:" + anomaly_cluster_str), content)

# Calculate final F1 scores
calculate_f1_and_average_scores(
    csv_file=change_result_csv_file,
    txt_file=output_completion_file_path
)

if __name__ == "__main__":
    main()
