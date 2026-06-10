# PIKE-RAG: Hierarchical RAG Retrieval System

面向技术手册/层级文档的分层检索增强生成 (RAG) 框架，专注于通过**层级感知**（Hierarchy-Aware）检索策略提升问答准确性。

## Overview

传统的 RAG 检索器在向量搜索时，仅依赖 `chunk` 的语义相似度，忽略了文档的层级结构信息。在电力设备说明书、技术文档等结构化的文档中，同一章节下的多个 chunks 往往包含了问题所需的完整上下文，但传统检索可能只返回其中部分 chunks，丢失关键信息。

HIER（Hierarchical Retrieval）通过三层核心策略解决这个问题：

1. **Multi-Aspect Indexing（富文本索引）**: 将层级路径和内容拼接为 `[层级] {title_path} [正文] {content}` 格式进行嵌入，让标题关键词获得更高的向量权重
2. **Hierarchical Expansion（层级扩展）**: 当 top-K 结果中超过 60% 的 chunks 属于同一父章节时，自动展开该章节下所有 chunks，保证上下文完整性
3. **Entity-Based Reranking（实体重排序）**: 提取 Query 和文档中的实体，按实体匹配数 + 向量相似度重新排序

实验表明，在 DPS-500G-PPR 极保护系统技术说明书测试集上，HIER 将准确率从基线 RAG 的 ~70% 提升至 ~80%。

## Architecture

```
                              ┌──────────────────────────┐
                              │     YAML Configuration    │
                              └────────────┬─────────────┘
                                           │
                                   QaWorkflow / Service
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
              QA Protocol           Retriever (HIER)          LLM Client
         (Prompt Templates)              │              (OpenAI / Qwen API)
                    │           ┌────────┼────────┐           │
                    │           │        │        │           │
                    ▼           ▼        ▼        ▼           ▼
              Answer Gen    Hierarchy  Entity   Iterative   Caching
                           Expansion  Ranking   Retry       (PickleDB)
                                         │
                                         ▼
                                    Evaluator
                          (ExactMatch / F1 / LLM-Judge)
```

### Retrieval Pipeline

```
Query ──► Vector Search (top-K×2) ──► Check Parent Title Consensus (>60%)
                                              │
                           ┌──────────────────┴──────────────────┐
                           ▼ YES                                  ▼ NO
                    Expand to ALL chunks                    Return top-K
                    in that section                         by similarity
                           │
                           ▼
                    Re-score against query
                    (cosine similarity)
                           │
                           ▼
                    Entity-based Reranking
                    score = entity_weight × matched_count
                          + similarity_weight × sim_score
                           │
                           ▼
                    Return top-K content strings
```

## Retrieval Strategies

| 检索器 | 文件 | 策略描述 |
|--------|------|----------|
| `QaChunkRetriever` | `chroma_qa_retriever.py` | **基线**: 单步向量检索 |
| `HierarchicalChunkRetriever` | `hierarchical_retriever.py` | **核心**: 层级感知检索，富文本嵌入 + 章节展开 |
| `HierarchicalChunkRetrieverWithMeta` | `hierarchical_retriever.py` | 层级检索 + 元数据过滤 |
| `EntityHierarchicalRetriever` | `entity_hierarchical_retriever.py` | 层级检索 + 实体匹配重排序 |
| `EntityMultiQueryHierarchicalRetriever` | `entity_hierarchical_retriever.py` | 多查询分解 + 层级 + 实体 + RRF 融合 |
| `SelfReflectiveHierarchicalRetriever` | `self_reflective_retriever.py` | Self-RAG 风格，检索后由 LLM 判断是否可答，不可则重试 |
| `AnswerFirstReflectiveRetriever` | `answer_first_retriever.py` | 先生成答案再验证，含"无法确定"则重试 |
| `IterHierarchicalEntityRetriever` | `iter_hierarchical_entity_retriever.py` | **旗舰**: 层级 + 实体 + 迭代重试 (k=10→15→20→25) |
| `RagFusionRetriever` | `rag_fusion_retriever.py` | 多查询变体 + RRF 融合 |
| `HydeQaChunkRetriever` | `hyde_retriever.py` | 假设文档嵌入 (HyDE) |
| `ChunkAtomRetriever` | `chunk_atom_retriever.py` | 双向量库: chunks + atomic questions |
| `BM25QaChunkRetriever` | `bm25_retriever.py` | BM25 稀疏检索 |

## Multi-Aspect Indexing

每个文档 chunk 以富文本格式嵌入，将层级信息注入向量空间：

```
[层级] DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点 > 1.2.1 高性能的硬件架构
[正文] 基于2.5GH总线频率的超高速串行背板通信总线...
```

层级元数据：
| 字段 | 示例 |
|------|------|
| `root` | DPS-500G-PPR极保护系统技术说明书 |
| `section_1` | 1. 概述 |
| `section_2` | 1.2 性能特点 |
| `section_3` | 1.2.1 高性能的硬件架构 |
| `parent_title` | DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点 |

## Hierarchical Expansion 示例

查询: "极保护装置的硬件架构有什么特点"

1. 初始向量检索 16 个 chunks
2. 发现 10/16 (62.5%) 属于 `> 1. 概述 > 1.2 性能特点` 章节
3. 62.5% > 60% 阈值，触发展开
4. 将该章节下所有 chunks 召回
5. 按与 query 的相似度重新评分，返回 top-8

## Project Structure

```
.
├── pikerag/                              # 核心库
│   ├── knowledge_retrievers/             # 检索器（12种策略）
│   │   ├── base_qa_retriever.py          # 抽象基类
│   │   ├── chroma_qa_retriever.py        # 基线检索器
│   │   ├── hierarchical_retriever.py     # ★ 层级检索器
│   │   ├── entity_hierarchical_retriever.py  # ★ 实体+层级检索器
│   │   ├── iter_hierarchical_entity_retriever.py  # ★ 迭代+层级+实体
│   │   ├── self_reflective_retriever.py  # Self-RAG 检索器
│   │   ├── answer_first_retriever.py     # 先答后验检索器
│   │   ├── rag_fusion_retriever.py       # RAG Fusion 检索器
│   │   ├── hyde_retriever.py             # HyDE 检索器
│   │   ├── chunk_atom_retriever.py       # 原子问题检索器
│   │   ├── bm25_retriever.py             # BM25 检索器
│   │   ├── mixins/                       # ChromaDB 操作 mixin
│   │   └── query_parsers/                # 查询解析器
│   ├── llm_client/                       # LLM 客户端（OpenAI/Azure/HF）
│   ├── prompts/                          # Prompt 模板
│   │   ├── qa/                           # 问答协议
│   │   ├── chunking/                     # 递归分割（中/英）
│   │   ├── decomposition/                # 问题分解
│   │   ├── ircot/                        # IRCoT 推理
│   │   └── tagging/                      # 语义标注
│   ├── workflows/                        # 实验流程
│   │   ├── qa.py                         # 主 QA 流程
│   │   ├── qa_decompose.py               # 原子分解流程
│   │   ├── qa_ircot.py                   # IRCoT 流程
│   │   ├── evaluate.py                   # 评估流程
│   │   └── evaluation/                   # 评估指标
│   └── utils/                            # 工具函数
├── examples/electricity/                 # 电力领域示例
│   ├── configs/                          # 10 种实验配置
│   │   ├── qa.yml                        # 基线 RAG
│   │   ├── hierarchical_rag.yml          # 层级 RAG
│   │   ├── hierarchical_rag_entity.yml   # 层级+实体 RAG
│   │   ├── hierarchical_rag_iter_hier_entity.yml  # 迭代+层级+实体
│   │   ├── rag_fusion.yml                # RAG Fusion
│   │   ├── hyde.yml                      # HyDE
│   │   └── ...
│   ├── service.py                        # Flask REST API 服务
│   └── extract_entities.py               # 实体提取脚本
├── data_process/                         # 数据预处理脚本
│   ├── pdf2md.py                         # PDF → Markdown
│   ├── chunk_by_sentence.py              # 句子级分割
│   ├── generate_qa.py                    # QA 对生成
│   ├── augment_answers.py                # 答案增强
│   └── tag.py                            # 实体标注
├── data/electricity/                     # 电力领域数据
├── env_configs/                          # 环境变量（API keys）
└── examples/
    ├── qa.py                             # ★ 主入口：运行 QA 实验
    ├── evaluate.py                       # 评估入口
    ├── chunking.py                       # 分割入口
    └── tagging.py                        # 标注入口
```

## Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r examples/requirements.txt

# 3. Set environment variables
# 编辑 env_configs/.env，填入 API key
# LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_API_KEY=your-api-key
```

## Quick Start

### 1. 运行 QA 实验

```bash
# Windows PowerShell
$Env:PYTHONPATH=$PWD
python examples/qa.py examples/electricity/configs/hierarchical_rag.yml

# Linux / macOS
export PYTHONPATH=$PWD
python examples/qa.py examples/electricity/configs/hierarchical_rag.yml
```

### 2. 启动检索服务（Flask API）

```bash
python examples/electricity/service.py \
    --config examples/electricity/configs/hierarchical_rag_iter_hier_entity.yml \
    --port 5000
```

API 端点:
- `POST /retrieve` — 检索接口
- `GET /health` — 健康检查

### 3. 运行评估

```bash
python examples/evaluate.py examples/evaluate.yml
```

## Configuration

所有实验通过 YAML 配置文件驱动，支持动态类加载。示例配置 `hierarchical_rag.yml`：

```yaml
experiment_name: hierarchical_rag

workflow:                           # 流程类
  module_path: pikerag.workflows.qa
  class_name: QaWorkflow

qa_protocol:                        # Prompt 模板
  module_path: pikerag.prompts.qa
  attr_name: generation_qa_with_reference_protocol

llm_client:                         # LLM 配置
  module_path: pikerag.llm_client
  class_name: StandardOpenAIClient
  llm_config:
    model: qwen-plus-latest
    temperature: 0

retriever:                          # ★ 检索器配置
  module_path: pikerag.knowledge_retrievers
  class_name: HierarchicalChunkRetriever
  args:
    retrieve_k: 8                          # 最终返回 chunk 数
    hierarchy_threshold: 0.6               # 展开触发阈值 (60%)
    hierarchy_meta_name: "parent_title"    # 章节分组字段
    expand_to_section: true                # 是否启用章节展开

    vector_store:
      collection_name: electricity_hierarchical
      persist_directory: data/vector_stores/electricity_hierarchical
      id_document_loading:
        module_path: pikerag.utils.data_protocol_utils
        func_name: load_ids_and_chunks_with_hierarchy
        args:
          filepath: data/electricity/chunks_output.jsonl
          title_prefix: "[层级]"
          content_prefix: "[正文]"
      embedding_setting:
        module_path: pikerag.llm_client.standard_openai_api
        class_name: StandardOpenAIEmbedding
        args:
          model: text-embedding-v3

evaluator:                          # 评估指标
  metrics:
    - ExactMatch
    - F1
    - Precision
    - Recall
    - LLM
```

### 关键参数调优

| 参数 | 作用 | 建议 |
|------|------|------|
| `hierarchy_threshold` | 越大越保守（展开越少） | 0.5 ~ 0.7 |
| `retrieve_k` | 最终返回 chunk 数 | 8 ~ 20 |
| `hierarchy_meta_name` | 切换分组级别 | `parent_title` / `section_1` / `section_2` |
| `entity_weight` | 实体匹配权重（仅 entity 类） | 0.3 ~ 0.7 |
| `max_retries` | 迭代重试次数（仅 iter 类） | 2 ~ 4 |

## Data Preprocessing Pipeline

```
PDF ──► pdf2md.py ──► Markdown
                         │
                    chunk_by_sentence.py ──► chunks_output.jsonl
                         │
                    extract_entities.py ──► chunks_with_entities.jsonl
                         │
                    tag.py ──► 注释标注
                         │
                    generate_qa.py ──► qa_output.jsonl
                         │
                    augment_answers.py ──► qa_output_augmented.jsonl
                         │
                    load_ids_and_chunks_with_hierarchy()
                         │
                    ChromaDB Vector Store
```

## Evaluation

### 评估指标

| 指标 | 描述 |
|------|------|
| **ExactMatch** | 完全匹配（字符串完全一致） |
| **F1** | F1 分数（词级别） |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **LLM-Judge** | LLM 判定准确率（最灵活） |

### 评估流程

1. 运行一次 QA 实验（生成 `logs/{experiment_name}/{experiment_name}.jsonl`）
2. 使用 `examples/evaluate.py` 对结果计算各项指标
3. 对比不同检索策略的效果

## License

MIT License — Copyright (c) Microsoft Corporation.

## Acknowledgements

本项目使用以下技术栈：
- **ChromaDB** — 轻量级向量数据库（metadata filtering 支持）
- **LangChain** — ChromaDB wrapper、文档抽象
- **OpenAI API / DashScope** — LLM 推理与 Embedding（兼容 OpenAI 协议）
- **Qwen** — 主要使用的 LLM 模型（通义千问）
- **RRF (Reciprocal Rank Fusion)** — 多查询结果融合
