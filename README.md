# NKAIOps-SoftwareChange
Repository of Paper: SCELM, NKAIOps, SoftwareChange Project. Maintained by [Weihua Kuang](https://github.com/waywooKwong).

Using Large Language Model to improve efficiency of software change maintainance.

<img src="pic/image.png" width="500" alt="示意图">

## Environment

### Tools

- Ollama, deploy models on local server

    recommended gemma2-9b, qwen2.5-7b,llama3.1-7b
- Redis

### Models

- m3e-base, vector embeeding model.
- all-MiniLM-L6-v2, semantic similarity matching model.

## How to run

### 1. Run Ollama server

### 2. Run Redis server

### 3. Run code

1. Part1_AreaText-Generate

    Generate areatext according to change information.

2. Part2_LLM-Generate

    Change analysis by LLM using Retriever-Augmented Generation.

## Work

LLM-RAG part code & experiment design by Weihua Kuang,

Baseline experiment by ShenChao.

