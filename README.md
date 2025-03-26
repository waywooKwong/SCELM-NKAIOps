# NKAIOps-SoftwareChange

Repository of SCELM, SoftwareChange Project from AIOps Lab, Nankai University.

Maintained by [Weihua Kuang](https://github.com/waywooKwong), Guided by [Yongqian Sun](https://nkcs.iops.ai/yongqiansun/) and [Shenglin Zhang](https://nkcs.iops.ai/zhangshenglin/).

feel free to email `weihua.kwong@mail.nankai.edu.cn` if you have any function problems.

## News

⭐️ 2025/03/25: Our Paper ***A Multimodal Intelligent Change Assessment Framework for Microservice
Systems Based on Large Language Models*** (SCELM) accepted by FSE 2025 Industry (45/165)!

2025/03/10: Test & Deploy SCELM on ByteDance enterprise data.

2025/01/23: Paper submission to FSE 2025.

2025/01/10: Refactor code.

... : more detailed depolyment tutorials comming soon!

## About

Using Large Language Model to improve efficiency of software change maintainance.

Our method achieved SOTA performance on the business scenario:

- *Yun Zhanghu Technology*
- preliminary tests on *ByteDance*

<img src="pic/image.png" width="800" alt="示意图">

## Environment

### Tools

- Ollama, deploy models on local server
  - version 0.5.1
  - recommended model: gemma2-9b, qwen2.5-7b,llama3.1-7b
- Redis, vector database
  - version 7.2.5

### Models

download the model files from the web and place them under the `/model` folder

- [m3e-base](https://huggingface.co/moka-ai/m3e-base), vector embeeding model.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), semantic similarity matching model.

## How to run

### 1. Run Ollama server

```
ollama serve
```

(optional) Test ollama model chat

models supported according to:  [Ollama Models](https://ollama.com/search)

```
ollama pull <model_name>
<input chat content>
```

### 2. Run Redis server

```
redis-server
```

more detailed command according to [Redis tutorials](https://redis.io/docs/latest/),

(recommended) Use [Redis Insight](https://redis.io/insight/) to visually manage data, mind your connection port.

### 3. Run SCELM function code

1. Part1_AreaText-Generate

   Generate areatext according to change information.
2. Part2_LLM-Generate

   Change analysis by LLM using Retriever-Augmented Generation.

### 4. Evaluate results

according to codes in `/utils`

## Contributions

Any function problems connect [Weihua Kuang](https://github.com/waywooKwong) by `weihua.kwong@mail.nankai.edu.cn`

Any baseline experiments connect [Chao Shen](https://github.com/sc2213404) by `2213404@mail.nankai.edu.cn`

- RAG functional code, experiment design & Paper script by Weihua Kuang,
- Baseline experiment by Chao Shen,
- Co-guided & Mainly paper writing by other writers.

## Thanks

- Tinghua Zheng, his intern experience in *Yun Zhanghu Technology* leads to this project.
- [Guangba Yu](https://yuxiaoba.github.io/) helps for one baseline experiment about his work [ChangeRCA](https://github.com/IntelligentDDS/ChangeRCA).
- [Hailin Zhang](https://github.com/HeilynZhang) helps run our SCELM on preliminary data of *Bytedance*.
