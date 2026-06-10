# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Self-Reflective Hierarchical Retriever

Strategy:
1. Initial retrieval with hierarchical strategy
2. Ask LLM: "Can you answer the question with these retrieved documents?"
3. If no, retry with expanded parameters (larger retrieve_k, lower threshold)
4. Repeat until success or max retries

This implements Self-RAG / FLARE-like self-correction mechanism.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from langchain_core.documents import Document

from pikerag.knowledge_retrievers.hierarchical_retriever import HierarchicalChunkRetriever
from pikerag.utils.logger import Logger


class SelfReflectiveHierarchicalRetriever(HierarchicalChunkRetriever):
    """
    Hierarchical Retriever with Self-Reflection for retrieval improvement.

    Pipeline:
    1. Initial retrieval (hierarchical)
    2. Ask LLM: Can you answer with these docs?
    3. If no, retry with expanded parameters
    4. Return final documents
    """

    name: str = "SelfReflectiveHierarchicalRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Self-reflection LLM
        self.reflection_llm_config: dict = retriever_config.get("reflection_llm", {})
        self.max_retries: int = retriever_config.get("max_retries", 3)
        self.initial_retrieve_k: int = retriever_config.get("initial_retrieve_k", self.retrieve_k)
        self.retry_increment: int = retriever_config.get("retry_increment", 4)
        self.initial_threshold: float = self.retrieve_score_threshold

        # Entity-based ranking (optional)
        self.use_entity_ranking: bool = retriever_config.get("use_entity_ranking", False)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_weight: float = retriever_config.get("entity_weight", 0.6)
        self.similarity_weight: float = retriever_config.get("similarity_weight", 0.4)

        self._init_components()

    def _init_components(self) -> None:
        """Initialize reflection LLM."""
        self.reflection_llm = None
        if self.reflection_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.reflection_llm_config.get("module_path", "pikerag.llm_client"),
                self.reflection_llm_config.get("class_name", "BaseLLMClient")
            )
            self.reflection_llm = llm_cls(self.reflection_llm_config.get("args", {}))
            self.logger.info("Reflection LLM initialized")

        # Entity extraction LLM
        self.entity_llm = None
        if self.use_entity_ranking and self.entity_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.entity_llm_config.get("module_path", "pikerag.llm_client"),
                self.entity_llm_config.get("class_name", "BaseLLMClient")
            )
            self.entity_llm = llm_cls(self.entity_llm_config.get("args", {}))

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _extract_query_entities(self, query: str) -> set:
        """Extract entities from query using LLM."""
        if self.entity_llm is None:
            return set()

        prompt = f"""从以下问题中提取所有实体（设备名、技术术语、标准规范等），只返回逗号分隔的列表。

问题: {query}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            entities = [e.strip().lower() for e in content.split(',') if e.strip()]
            return set(entities)
        except:
            return set()

    def _get_doc_entities(self, doc: Document) -> set:
        """Get entities from document metadata."""
        entities_str = doc.metadata.get("entities", "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def _check_can_answer(self, query: str, contexts: List[str]) -> Tuple[bool, str]:
        """Ask LLM if it can answer the question with given contexts."""
        if self.reflection_llm is None:
            return True, ""

        context_text = "\n\n".join([f"[文档{i+1}]: {c[:500]}" for i, c in enumerate(contexts)])

        prompt = f"""你是一个信息检索评估系统。给定一个问题和相关文档，判断是否能够根据这些文档回答问题。

问题: {query}

文档内容:
{context_text}

请用以下格式回答：
CAN_ANSWER: YES 或 NO
MISSING_INFO: 详细说明缺少什么信息（如果不能回答）

只返回这两行，不要其他内容。"""

        try:
            response = self.reflection_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)

            can_answer = "CAN_ANSWER: YES" in content.upper()
            missing_info = ""

            # Extract missing info
            for line in content.split('\n'):
                if "MISSING_INFO" in line or "缺少" in line or "missing" in line.lower():
                    missing_info = line.split(':', 1)[-1].strip()
                    break

            return can_answer, missing_info
        except Exception as e:
            self.logger.warning(f"Reflection check failed: {e}")
            return True, ""  # Assume can answer if check fails

    def _generate_improved_query(self, query: str, missing_info: str) -> str:
        """Generate improved query based on missing information."""
        if self.reflection_llm is None:
            return query

        prompt = f"""根据缺少的信息，重新生成优化后的检索查询，使其更容易找到相关文档。

原始问题: {query}
缺少的信息: {missing_info}

请生成一个或多个优化的查询词（用逗号分隔），这些查询词应该更容易在文档中找到答案。

优化查询:"""

        try:
            response = self.reflection_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            # Return original query + improved parts
            return f"{query} {content}"
        except:
            return query

    def _entity_based_ranking(
        self,
        query: str,
        query_entities: set,
        docs: List[Tuple[Document, float]],
        retrieve_k: int
    ) -> List[str]:
        """Rank documents by entity matching."""
        if not query_entities:
            return [doc.metadata.get("content", doc.page_content) for doc, _ in docs[:retrieve_k]]

        scored = []
        for doc, sim_score in docs:
            doc_entities = self._get_doc_entities(doc)
            matched = len(query_entities.intersection(doc_entities))
            combined = matched * self.entity_weight + sim_score * self.similarity_weight
            content = doc.metadata.get("content", doc.page_content)
            scored.append((content, matched, combined))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [c for c, m, s in scored[:retrieve_k]]

    def _hierarchical_retrieve(
        self,
        query: str,
        retrieve_k: int,
        score_threshold: float
    ) -> List[str]:
        """Perform hierarchical retrieval."""
        # Initial retrieval
        initial_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, score_threshold)

        if len(initial_docs) == 0:
            return []

        # Check expansion
        parent_titles = [doc.metadata.get(self.hierarchy_meta_name, "") for doc, _ in initial_docs]
        title_counts = Counter(parent_titles)
        most_common, count = title_counts.most_common(1)[0]
        ratio = count / len(initial_docs)

        if ratio >= self.hierarchy_threshold:
            expanded_docs = self._expand_to_section(most_common)
            if len(expanded_docs) > 0:
                # Re-score
                from pikerag.utils.config_loader import load_embedding_func
                embedding_config = self._retriever_config.get("vector_store", {}).get("embedding_setting", {})
                embedding = load_embedding_func(
                    module_path=embedding_config.get("module_path", None),
                    class_name=embedding_config.get("class_name", None),
                    **embedding_config.get("args", {}),
                )
                query_emb = embedding.embed_query(query)
                scored = []
                for doc in expanded_docs:
                    doc_emb = embedding.embed_query(doc.page_content)
                    sim = float(np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
                    ))
                    scored.append((doc, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [doc.metadata.get("content", doc.page_content) for doc, _ in scored[:retrieve_k]]

        # Return top results
        return [doc.metadata.get("content", doc.page_content) for doc, _ in initial_docs[:retrieve_k]]

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Self-reflective retrieval with retry mechanism.

        Returns:
            List of retrieved content strings
        """
        retrieve_k = kwargs.get("retrieve_k", self.initial_retrieve_k)
        current_threshold = kwargs.get("score_threshold", self.initial_threshold)

        # Extract query entities for ranking
        query_entities = set()
        if self.use_entity_ranking:
            query_entities = self._extract_query_entities(query)
            self.logger.debug(f"Query entities: {query_entities}")

        # Self-reflection loop
        for retry in range(self.max_retries + 1):
            # Retrieval
            retrieved_docs = self._hierarchical_retrieve(query, retrieve_k, current_threshold)

            if len(retrieved_docs) == 0:
                # No docs retrieved, expand and retry
                if retry < self.max_retries:
                    self.logger.info(f"Retry {retry + 1}: No documents retrieved, expanding search...")
                    retrieve_k += self.retry_increment
                    current_threshold = max(0.0, current_threshold - 0.1)
                    continue
                else:
                    return []

            # Self-reflection check
            can_answer, missing_info = self._check_can_answer(query, retrieved_docs)

            if can_answer:
                self.logger.info(f"Success on attempt {retry + 1}")
                break

            if retry >= self.max_retries:
                self.logger.info(f"Max retries reached ({self.max_retries}), returning best effort")
                break

            # Generate improved query
            improved_query = self._generate_improved_query(query, missing_info)
            if improved_query != query:
                self.logger.info(f"Retry {retry + 1}: Using improved query")
                query = improved_query

            # Expand parameters for next attempt
            retrieve_k += self.retry_increment
            current_threshold = max(0.0, current_threshold - 0.1)
            self.logger.info(f"Retry {retry + 1}: Expanding to k={retrieve_k}, threshold={current_threshold:.2f}")

        # Final entity-based ranking if enabled
        if self.use_entity_ranking and query_entities:
            # Re-fetch docs for entity ranking
            all_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, 0.0)
            return self._entity_based_ranking(query, query_entities, all_docs, kwargs.get("retrieve_k", self.retrieve_k))

        return retrieved_docs[:kwargs.get("retrieve_k", self.retrieve_k)]


class SelfReflectiveMultiQueryRetriever(HierarchicalChunkRetriever):
    """
    Combines:
    1. Query Decomposition (Multi-Query)
    2. Hierarchical Retrieval
    3. Self-Reflection for retry
    4. Entity-based Ranking
    """

    name: str = "SelfReflectiveMultiQueryRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Multi-query config
        self.num_sub_queries: int = retriever_config.get("num_sub_queries", 3)
        self.decomposition_llm_config: dict = retriever_config.get("decomposition_llm", {})

        # Self-reflection config
        self.reflection_llm_config: dict = retriever_config.get("reflection_llm", {})
        self.max_retries: int = retriever_config.get("max_retries", 2)

        # Entity ranking
        self.use_entity_ranking: bool = retriever_config.get("use_entity_ranking", True)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_weight: float = retriever_config.get("entity_weight", 0.6)
        self.similarity_weight: float = retriever_config.get("similarity_weight", 0.4)

        # RRF
        self.rrf_k: int = retriever_config.get("rrf_k", 60)

        self._init_components()

    def _init_components(self) -> None:
        """Initialize LLMs."""
        # Decomposition LLM
        self.decomp_llm = None
        if self.decomposition_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.decomposition_llm_config.get("module_path", "pikerag.llm_client"),
                self.decomposition_llm_config.get("class_name", "BaseLLMClient")
            )
            self.decomp_llm = llm_cls(self.decomposition_llm_config.get("args", {}))

        # Reflection LLM
        self.reflection_llm = None
        if self.reflection_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.reflection_llm_config.get("module_path", "pikerag.llm_client"),
                self.reflection_llm_config.get("class_name", "BaseLLMClient")
            )
            self.reflection_llm = llm_cls(self.reflection_llm_config.get("args", {}))

        # Entity LLM
        self.entity_llm = None
        if self.use_entity_ranking and self.entity_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.entity_llm_config.get("module_path", "pikerag.llm_client"),
                self.entity_llm_config.get("class_name", "BaseLLMClient")
            )
            self.entity_llm = llm_cls(self.entity_llm_config.get("args", {}))

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose query into sub-queries."""
        if self.decomp_llm is None:
            return [query]

        prompt = f"""将以下问题分解为{self.num_sub_queries}个独立子问题。

问题: {query}

子问题:
1."""

        try:
            response = self.decomp_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)

            sub_queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    sq = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('-').strip()
                    if sq:
                        sub_queries.append(sq)

            return [query] + sub_queries[:self.num_sub_queries - 1]
        except:
            return [query]

    def _extract_query_entities(self, query: str) -> set:
        """Extract entities from query."""
        if self.entity_llm is None:
            return set()

        prompt = f"""从问题中提取实体，只返回逗号分隔列表。

问题: {query}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            entities = [e.strip().lower() for e in content.split(',') if e.strip()]
            return set(entities)
        except:
            return set()

    def _check_can_answer(self, query: str, contexts: List[str]) -> Tuple[bool, str]:
        """Check if LLM can answer with given contexts."""
        if self.reflection_llm is None:
            return True, ""

        context_text = "\n\n".join([f"[文档{i+1}]: {c[:400]}" for i, c in enumerate(contexts)])

        prompt = f"""能否根据以下文档回答问题？

问题: {query}

文档:
{context_text}

回复格式：
CAN_ANSWER: YES/NO
MISSING: 缺少什么信息

只返回两行。"""

        try:
            response = self.reflection_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip()
            can_answer = "CAN_ANSWER: YES" in content.upper()
            missing = ""
            for line in content.split('\n'):
                if "MISSING" in line.upper():
                    missing = line.split(':', 1)[-1].strip()
            return can_answer, missing
        except:
            return True, ""

    def _hierarchical_retrieve(self, query: str, retrieve_k: int) -> List[str]:
        """Hierarchical retrieval for a single query."""
        initial_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, None)

        if len(initial_docs) == 0:
            return []

        # Check hierarchy
        parent_titles = [doc.metadata.get(self.hierarchy_meta_name, "") for doc, _ in initial_docs]
        title_counts = Counter(parent_titles)
        most_common, count = title_counts.most_common(1)[0]
        ratio = count / len(initial_docs)

        if ratio >= self.hierarchy_threshold:
            expanded_docs = self._expand_to_section(most_common)
            if len(expanded_docs) > 0:
                from pikerag.utils.config_loader import load_embedding_func
                embedding_config = self._retriever_config.get("vector_store", {}).get("embedding_setting", {})
                embedding = load_embedding_func(
                    module_path=embedding_config.get("module_path", None),
                    class_name=embedding_config.get("class_name", None),
                    **embedding_config.get("args", {}),
                )
                query_emb = embedding.embed_query(query)
                scored = []
                for doc in expanded_docs:
                    doc_emb = embedding.embed_query(doc.page_content)
                    sim = float(np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
                    ))
                    content = doc.metadata.get("content", doc.page_content)
                    scored.append((content, sim, doc))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [c for c, s, d in scored[:retrieve_k]]

        return [doc.metadata.get("content", doc.page_content) for doc, _ in initial_docs[:retrieve_k]]

    def _rrf_fuse(self, rankings: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """RRF fusion."""
        combined: Dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc, _) in enumerate(ranking):
                rrf = 1.0 / (self.rrf_k + rank + 1)
                combined[doc] = combined.get(doc, 0) + rrf
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def _get_doc_entities(self, doc: Document) -> set:
        """Get entities from document."""
        entities_str = doc.metadata.get("entities", "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Self-reflective multi-query retrieval.
        """
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        target_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Extract query entities
        query_entities = set()
        if self.use_entity_ranking:
            query_entities = self._extract_query_entities(query)

        # Decompose query
        sub_queries = self._decompose_query(query)

        # Multi-query retrieval with self-reflection
        all_rankings: List[List[Tuple[str, float]]] = []

        for retry in range(self.max_retries + 1):
            current_k = retrieve_k + retry * self.retry_increment

            for sq in sub_queries:
                docs = self._hierarchical_retrieve(sq, current_k)
                all_rankings.append([(d, 0.0) for d in docs])

            # Check if we can answer
            if self.reflection_llm and all_rankings:
                # Get fused docs for checking
                fused = self._rrf_fuse(all_rankings)
                context_docs = [d for d, _ in fused[:retrieve_k]]
                can_answer, missing = self._check_can_answer(query, context_docs)

                if can_answer:
                    self.logger.info(f"Success on retry {retry}")
                    break

                if retry >= self.max_retries:
                    self.logger.info("Max retries reached")
                    break

                self.logger.info(f"Retry {retry}: Expanding search (missing: {missing[:50]}...)")
            else:
                break

        if not all_rankings:
            return []

        # RRF fusion
        fused = self._rrf_fuse(all_rankings)
        candidates = [d for d, _ in fused[:retrieve_k * 2]]

        # Entity-based final ranking
        if self.use_entity_ranking and query_entities:
            # Get docs with metadata
            docs_with_meta = []
            for content in candidates:
                result = self.vector_store.get(where={"content": content})
                if result["documents"]:
                    doc = Document(page_content=content, metadata=result["metadatas"][0])
                    docs_with_meta.append((doc, 0.0))

            scored = []
            for doc, _ in docs_with_meta:
                doc_entities = self._get_doc_entities(doc)
                matched = len(query_entities.intersection(doc_entities))
                content = doc.metadata.get("content", doc.page_content)
                scored.append((content, matched))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, m in scored[:target_k]]

        return candidates[:target_k]
