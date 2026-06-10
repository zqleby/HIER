# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Iterative Retrieval + Hierarchical Retrieval + Entity Ranking

Combines:
1. Hierarchical retrieval (section expansion)
2. Entity-based ranking
3. Iterative retry: if answer is "cannot determine", expand search and retry

This is iter_retgen approach adapted for hierarchical retrieval.
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import re
import numpy as np

from langchain_core.documents import Document

from pikerag.knowledge_retrievers.hierarchical_retriever import HierarchicalChunkRetriever
from pikerag.utils.logger import Logger


class IterHierarchicalEntityRetriever(HierarchicalChunkRetriever):
    """
    Iterative Hierarchical Retrieval with Entity Ranking.

    Pipeline:
    1. Initial hierarchical retrieval with entity ranking
    2. Generate answer using LLM
    3. Check if answer indicates "cannot determine"
    4. If cannot, expand parameters and retry
    5. Return final documents
    """

    name: str = "IterHierarchicalEntityRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Retrieval parameters
        self.retrieve_k = retriever_config.get("retrieve_k", 10)
        self.retrieve_score_threshold = 0.0  # Lower threshold

        # Iterative configuration
        self.max_retries: int = retriever_config.get("max_retries", 3)
        self.retry_increment: int = retriever_config.get("retry_increment", 5)

        # LLM for answer generation
        self.answer_llm_config: dict = retriever_config.get("answer_llm", {})

        # Entity configuration
        self.use_entity_ranking: bool = retriever_config.get("use_entity_ranking", True)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_weight: float = retriever_config.get("entity_weight", 0.5)
        self.similarity_weight: float = retriever_config.get("similarity_weight", 0.5)

        self._init_components()

    def _init_components(self) -> None:
        """Initialize LLMs."""
        # Answer generation LLM
        self.answer_llm = None
        if self.answer_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.answer_llm_config.get("module_path", "pikerag.llm_client"),
                self.answer_llm_config.get("class_name", "BaseLLMClient")
            )
            self.answer_llm = llm_cls(self.answer_llm_config.get("args", {}))
            self.logger.info("Answer generation LLM initialized")

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

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer from contexts."""
        if self.answer_llm is None:
            return "[LLM not configured]"

        context_text = "\n\n".join([f"[文档{i+1}]: {c[:800]}" for i, c in enumerate(contexts)])

        prompt = f"""基于以下文档内容，回答问题。如果文档中包含答案，请给出具体答案；如果无法从文档中找到答案，请明确说明"无法确定"或"未提及"。

问题: {question}

文档内容:
{context_text}

请基于以上文档内容直接回答问题："""

        try:
            response = self.answer_llm.generate([{"role": "user", "content": prompt}])
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.warning(f"Answer generation failed: {e}")
            return "[生成答案失败]"

    def _check_cannot_answer(self, answer: str) -> Tuple[bool, str]:
        """
        Check if answer indicates inability to answer.

        Returns: (is_unable, reason)
        """
        cannot_answer_phrases = [
            "无法确定", "无法回答", "不确定", "不知道", "未提及", "未提供",
            "cannot determine", "cannot answer", "not enough information",
            "insufficient information", "无法从文档", "没有提到",
            "无法找到", "找不到", "不存在", "not found"
        ]

        answer_lower = answer.lower()
        for phrase in cannot_answer_phrases:
            if phrase in answer or phrase in answer_lower:
                return True, f"Found: '{phrase}'"

        return False, ""

    def _extract_query_entities(self, question: str) -> set:
        """Extract entities from question."""
        if self.entity_llm is None:
            words = re.findall(r'[\w]+', question)
            stopwords = {"的", "是", "在", "了", "和", "与", "或", "及", "哪些", "什么", "多少", "是否", "能不能", "怎么", "如何"}
            keywords = [w for w in words if w not in stopwords and len(w) > 1]
            return set([k.lower() for k in keywords[:10]])

        prompt = f"""从问题中提取关键实体（设备名、技术术语、数值等），只返回逗号分隔列表。

问题: {question}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            entities = [e.strip().lower() for e in content.split(',') if e.strip()]
            return set(entities)
        except:
            return set()

    def _get_doc_entities(self, doc: Document) -> set:
        """Get entities from document."""
        entities_str = doc.metadata.get("entities", "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def _hierarchical_retrieve(self, query: str, retrieve_k: int) -> List[Tuple[Document, float]]:
        """Hierarchical retrieval returning docs with scores."""
        initial_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, 0.0)

        if len(initial_docs) == 0:
            return []

        # Check hierarchy for expansion
        parent_titles = [doc.metadata.get(self.hierarchy_meta_name, "") for doc, _ in initial_docs]
        title_counts = Counter(parent_titles)
        most_common, count = title_counts.most_common(1)[0]
        ratio = count / len(initial_docs)

        if ratio >= self.hierarchy_threshold:
            expanded_docs = self._expand_to_section(most_common)
            if len(expanded_docs) > 0:
                # Re-score expanded docs
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
                return scored[:retrieve_k]

        return initial_docs[:retrieve_k]

    def _entity_ranking(
        self,
        query_entities: set,
        docs: List[Tuple[Document, float]],
        retrieve_k: int
    ) -> List[str]:
        """Rank by entity matching + similarity."""
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

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Iterative hierarchical retrieval with entity ranking.

        Pipeline:
        1. Hierarchical retrieval + entity ranking
        2. Generate answer
        3. Check if cannot answer
        4. Retry with expanded parameters
        """
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        target_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Extract query entities
        query_entities = set()
        if self.use_entity_ranking:
            query_entities = self._extract_query_entities(query)
            self.logger.debug(f"Query entities: {query_entities}")

        # Iterative retrieval
        for retry in range(self.max_retries + 1):
            current_k = retrieve_k + retry * self.retry_increment

            # Step 1: Hierarchical retrieval
            retrieved_docs = self._hierarchical_retrieve(query, current_k)

            if len(retrieved_docs) == 0:
                if retry < self.max_retries:
                    continue
                return []

            # Step 2: Entity-based ranking
            ranked_contents = self._entity_ranking(query_entities, retrieved_docs, target_k)

            # Step 3: Generate answer
            generated_answer = self._generate_answer(query, ranked_contents)
            self.logger.debug(f"Iter {retry + 1} answer: {generated_answer[:80]}...")

            # Step 4: Check if cannot answer
            is_unable, reason = self._check_cannot_answer(generated_answer)

            if not is_unable:
                self.logger.info(f"Success on iteration {retry + 1}")
                return ranked_contents[:target_k]

            if retry >= self.max_retries:
                self.logger.info(f"Max retries ({self.max_retries}), returning best effort")
                return ranked_contents[:target_k]

            self.logger.info(f"Iteration {retry + 1}: {reason}, expanding search...")

        return ranked_contents[:target_k]


class IterMultiQueryHierarchicalEntityRetriever(HierarchicalChunkRetriever):
    """
    Iterative + Multi-Query + Hierarchical + Entity Ranking

    Pipeline:
    1. Query decomposition (generate sub-queries)
    2. Hierarchical retrieval for each sub-query
    3. RRF fusion
    4. Entity ranking
    5. Generate answer
    6. Check if cannot answer -> retry
    """

    name: str = "IterMultiQueryHierarchicalEntityRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Multi-query config
        self.num_sub_queries: int = retriever_config.get("num_sub_queries", 3)
        self.decomposition_llm_config: dict = retriever_config.get("decomposition_llm", {})

        # Iterative config
        self.max_retries: int = retriever_config.get("max_retries", 2)
        self.retry_increment: int = retriever_config.get("retry_increment", 3)

        # LLM for answer
        self.answer_llm_config: dict = retriever_config.get("answer_llm", {})

        # Entity config
        self.use_entity_ranking: bool = retriever_config.get("use_entity_ranking", True)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_weight: float = retriever_config.get("entity_weight", 0.5)
        self.similarity_weight: float = retriever_config.get("similarity_weight", 0.5)

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

        # Answer LLM
        self.answer_llm = None
        if self.answer_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.answer_llm_config.get("module_path", "pikerag.llm_client"),
                self.answer_llm_config.get("class_name", "BaseLLMClient")
            )
            self.answer_llm = llm_cls(self.answer_llm_config.get("args", {}))

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

    def _extract_query_entities(self, question: str) -> set:
        """Extract entities from question."""
        if self.entity_llm is None:
            words = re.findall(r'[\w]+', question)
            stopwords = {"的", "是", "在", "了", "和", "与", "或", "及", "哪些", "什么", "多少"}
            keywords = [w for w in words if w not in stopwords and len(w) > 1]
            return set([k.lower() for k in keywords[:10]])

        prompt = f"""从问题中提取关键实体，只返回逗号分隔列表。

问题: {question}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip()
            entities = [e.strip().lower() for e in content.split(',') if e.strip()]
            return set(entities)
        except:
            return set()

    def _get_doc_entities(self, doc: Document) -> set:
        """Get entities from document."""
        entities_str = doc.metadata.get("entities", "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def _hierarchical_retrieve(self, query: str, retrieve_k: int) -> List[Tuple[Document, float]]:
        """Hierarchical retrieval."""
        initial_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, 0.0)

        if len(initial_docs) == 0:
            return []

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
                    scored.append((doc, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:retrieve_k]

        return initial_docs[:retrieve_k]

    def _rrf_fuse(self, rankings: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """RRF fusion."""
        combined: Dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc, _) in enumerate(ranking):
                rrf = 1.0 / (self.rrf_k + rank + 1)
                combined[doc] = combined.get(doc, 0) + rrf
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer from contexts."""
        if self.answer_llm is None:
            return "[LLM not configured]"

        context_text = "\n\n".join([f"[文档{i+1}]: {c[:600]}" for i, c in enumerate(contexts)])

        prompt = f"""基于文档回答问题。如果无法确定，请明确说"无法确定"。

问题: {question}

文档:
{context_text}

回答:"""

        try:
            response = self.answer_llm.generate([{"role": "user", "content": prompt}])
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except:
            return "[生成失败]"

    def _check_cannot_answer(self, answer: str) -> bool:
        """Check if answer indicates inability."""
        cannot_phrases = ["无法确定", "无法回答", "不确定", "不知道", "未提及", "cannot", "not found"]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in cannot_phrases)

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """Iterative multi-query hierarchical retrieval."""
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        target_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Extract query entities
        query_entities = set()
        if self.use_entity_ranking:
            query_entities = self._extract_query_entities(query)

        # Decompose query
        sub_queries = self._decompose_query(query)

        # Iterative loop
        for retry in range(self.max_retries + 1):
            current_k = retrieve_k + retry * self.retry_increment

            # Multi-query hierarchical retrieval
            all_rankings: List[List[Tuple[str, float]]] = []

            for sq in sub_queries:
                docs = self._hierarchical_retrieve(sq, current_k)
                ranked = [(doc.metadata.get("content", doc.page_content), sim) for doc, sim in docs]
                all_rankings.append(ranked)

            if not all_rankings:
                if retry < self.max_retries:
                    continue
                return []

            # RRF fusion
            fused = self._rrf_fuse(all_rankings)
            candidates = [doc for doc, _ in fused[:target_k * 2]]

            # Entity ranking
            if self.use_entity_ranking and query_entities:
                docs_with_meta = []
                for content in candidates:
                    result = self.vector_store.get(where={"content": content})
                    if result["documents"]:
                        doc = Document(page_content=content, metadata=result["metadatas"][0])
                        docs_with_meta.append(doc)

                scored = []
                for doc in docs_with_meta:
                    doc_entities = self._get_doc_entities(doc)
                    matched = len(query_entities.intersection(doc_entities))
                    content = doc.metadata.get("content", doc.page_content)
                    scored.append((content, matched))

                scored.sort(key=lambda x: x[1], reverse=True)
                final_docs = [c for c, m in scored[:target_k]]
            else:
                final_docs = candidates[:target_k]

            # Generate answer
            answer = self._generate_answer(query, final_docs)
            self.logger.debug(f"Iter {retry + 1}: {answer[:60]}...")

            # Check
            if not self._check_cannot_answer(answer):
                self.logger.info(f"Success on iter {retry + 1}")
                return final_docs

            if retry >= self.max_retries:
                self.logger.info(f"Max retries, returning best")
                return final_docs

            self.logger.info(f"Iter {retry + 1}: Cannot answer, expanding...")

        return final_docs[:target_k]
