# Custom Hierarchical RAG for Electricity Domain

This document describes the custom RAG workflow built for the electricity domain (DPS-500G-PPR极保护系统).

## Architecture Overview

### 1. Multi-Aspect Indexing (索引阶段)

Instead of embedding raw `content`, we create a `rich_text` field:

```
[层级] {title} [正文] {content}
```

**Example:**
```
[层级] DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点 > 1.2.1 高性能的硬件架构 [正文] 基于2.5GH总线频率的超高速串行背板通信总线...
```

**Benefits:**
- Title keywords (如"极保护", "自诊断") get higher weight in similarity
- Structural information preserved in embedding space
- Original `content` stored separately for retrieval output

### 2. Hierarchical Metadata

Title paths are parsed into hierarchical components:

| Field | Example |
|-------|---------|
| `root` | DPS-500G-PPR极保护系统技术说明书 |
| `section_1` | 1. 概述 |
| `section_2` | 1.2 性能特点 |
| `section_3` | 1.2.1 高性能的硬件架构 |
| `parent_title` | DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点 |

### 3. Hierarchical Retrieval (检索阶段)

```
Initial Query → Vector Search (top 16) → Check if majority (>60%) share same parent
                                                    ↓
                              YES                      NO
                              ↓                        ↓
                    Expand to all chunks         Return top 8
                    under that parent            by similarity
```

**Example Scenario:**
- Query: "极保护装置的硬件架构有什么特点"
- Initial retrieval finds 10/16 results under "DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点"
- Since 62.5% > 60% threshold, expand to ALL chunks in that section
- Re-score and return top 8 by similarity

## Files Created

| File | Description |
|------|-------------|
| `pikerag/knowledge_retrievers/hierarchical_retriever.py` | Hierarchical retrieval implementation |
| `pikerag/utils/data_protocol_utils.py` | Added `load_ids_and_chunks_with_hierarchy()` |
| `examples/electricity/configs/hierarchical_rag.yml` | Main configuration file |
| `examples/electricity/test_hierarchical_data.py` | Data loading test script |

## Usage

### Run QA

```powershell
$Env:PYTHONPATH=$PWD
python examples/qa.py examples/electricity/configs/hierarchical_rag.yml
```

### 2. Configuration Options

Key parameters in `hierarchical_rag.yml`:

```yaml
retriever:
  class_name: HierarchicalChunkRetriever
  args:
    retrieve_k: 8                    # Final number of chunks returned
    hierarchy_threshold: 0.6         # Expansion trigger ratio (60%)
    hierarchy_meta_name: "parent_title"  # Field for grouping
    expand_to_section: true          # Enable/disable expansion
```

### 3. Tuning Parameters

| Parameter | Effect |
|-----------|--------|
| `hierarchy_threshold` | Higher = less expansion (more selective) |
| `retrieve_k` | More = better recall, less = faster |
| `hierarchy_meta_name` | Change grouping level (`section_1`, `section_2`, etc.) |

## Comparison with Baseline

| Metric | Baseline (QaChunkRetriever) | Hierarchical |
|--------|------------------------------|--------------|
| Retrieval | Direct similarity | Section expansion |
| Section coverage | May miss related chunks | Guaranteed section coverage |
| Use case | General QA | Hierarchical docs |

## Data Flow

```
chunks_output.jsonl
       ↓
load_ids_and_chunks_with_hierarchy()
       ↓
Document(rich_text, metadata={title, parent_title, ...})
       ↓
ChromaDB Vector Store
       ↓
HierarchicalChunkRetriever
       ↓
Output: content strings for LLM
```
