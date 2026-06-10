# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Self-Reflective Retrieval - Answer First, Then Verify

Pipeline:
1. Initial retrieval with hierarchical strategy
2. Ask LLM to generate answer based on retrieved docs
3. Check if answer is "unable to determine" or similar
4. If cannot answer, retry with expanded parameters
5. Return final documents

This implements Self-RAG style verification after generation.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple
import re
import numpy as np

from langchain_core.documents import Document

from pikerag.knowledge_retrievers.hierarchical_retriever import HierarchicalChunkRetriever
from pikerag.utils.logger import Logger


class AnswerFirstReflectiveRetriever(HierarchicalChunkRetriever):
    """
    Answer-First Self-Reflective Hierarchical Retriever.

    Key difference from traditional approach:
    1. Generate answer first, then verify
    2. Check if answer indicates inability to answer
    3. Retry if needed
    """

    name: str = "AnswerFirstReflectiveRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Retrieval parameters
        self.retrieve_k = retriever_config.get("retrieve_k", 10)
        self.retrieve_score_threshold = 0.0  # Lower threshold for more candidates

        # LLM for answer generation and verification
        self.answer_llm_config: dict = retriever_config.get("answer_llm", retriever_config.get("reflection_llm", {}))
        self.verify_llm_config: dict = retriever_config.get("verify_llm", retriever_config.get("reflection_llm", {}))

        # Retry configuration
        self.max_retries: int = retriever_config.get("max_retries", 3)
        self.retry_increment: int = retriever_config.get("retry_increment", 5)

        # Entity ranking
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

        # Verification LLM
        self.verify_llm = None
        if self.verify_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.verify_llm_config.get("module_path", "pikerag.llm_client"),
                self.verify_llm_config.get("class_name", "BaseLLMClient")
            )
            self.verify_llm = llm_cls(self.verify_llm_config.get("args", {}))

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
        """Generate answer based on retrieved contexts."""
        if self.answer_llm is None:
            return "[LLM not configured]"

        context_text = "\n\n".join([f"[文档{i+1}]: {c[:800]}" for i, c in enumerate(contexts)])

        prompt = f"""基于以下文档内容，回答问题。如果文档中包含答案，请给出具体答案；如果无法从文档中找到答案，请明确说明"无法确定"或"文档中未提及"。

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

    def _check_answer_validity(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Check if answer indicates inability to answer.

        Returns:
            Tuple of (can_answer, reason)
            - can_answer: True if answer is valid, False if indicates inability
            - reason: Explanation for the decision
        """
        if self.verify_llm is None:
            # Simple keyword-based check
            cannot_answer_phrases = [
                "无法确定", "无法回答", "不确定", "不知道", "未提及", "未提供",
                "cannot determine", "cannot answer", "not enough information",
                "insufficient information", "无法从文档", "没有提到"
            ]
            answer_lower = answer.lower()
            for phrase in cannot_answer_phrases:
                if phrase in answer or phrase in answer_lower:
                    return False, f"Answer contains: '{phrase}'"
            return True, "No inability indicator found"

        # LLM-based verification
        prompt = f"""判断以下回答是否能够回答问题。

问题: {question}

回答: {answer}

请判断：
1. 回答是否给出了具体、明确的答案？
2. 回答是否表示"无法确定"、"不知道"、"未提及"等无法回答的表述？

回复格式：
CAN_ANSWER: YES 或 NO
REASON: 判断理由

只返回这两行。"""

        try:
            response = self.verify_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip()

            can_answer = "CAN_ANSWER: YES" in content.upper()

            # Extract reason
            reason = ""
            for line in content.split('\n'):
                if line.startswith("REASON:"):
                    reason = line[7:].strip()

            return can_answer, reason
        except Exception as e:
            self.logger.warning(f"Verification failed: {e}")
            return True, "Verification failed, assuming valid"

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract key search terms from question."""
        words = re.findall(r'[\w]+', question)
        stopwords = {"的", "是", "在", "了", "和", "与", "或", "及", "以及", "哪些", "什么", "多少", "是否", "能不能", "怎么", "如何", "一个", "这个", "那个"}
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords[:10]

    def _extract_query_entities(self, question: str) -> set:
        """Extract entities from question."""
        if self.entity_llm is None:
            keywords = self._extract_keywords(question)
            return set([k.lower() for k in keywords])

        prompt = f"""从问题中提取关键实体（设备名、技术术语、数值等），只返回逗号分隔列表。

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
        """Get entities from document metadata."""
        entities_str = doc.metadata.get("entities", "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def _hierarchical_retrieve(self, query: str, retrieve_k: int) -> List[str]:
        """Hierarchical retrieval."""
        initial_docs = self._get_doc_with_query(query, self.vector_store, retrieve_k * 2, 0.0)

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
                    scored.append((doc, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [doc.metadata.get("content", doc.page_content) for doc, _ in scored[:retrieve_k]]

        return [doc.metadata.get("content", doc.page_content) for doc, _ in initial_docs[:retrieve_k]]

    def _entity_ranking(self, query: str, query_entities: set, docs: List[str], retrieve_k: int) -> List[str]:
        """Rank by entity matching."""
        if not query_entities:
            return docs[:retrieve_k]

        docs_with_meta = []
        for content in docs:
            result = self.vector_store.get(where={"content": content})
            if result["documents"]:
                doc = Document(page_content=content, metadata=result["metadatas"][0])
                docs_with_meta.append((doc, content))

        scored = []
        for doc, content in docs_with_meta:
            doc_entities = self._get_doc_entities(doc)
            matched = len(query_entities.intersection(doc_entities))
            scored.append((content, matched))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, m in scored[:retrieve_k]]

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Answer-first self-reflective retrieval.

        Pipeline:
        1. Retrieve documents
        2. Generate answer from documents
        3. Check if answer indicates inability
        4. Retry if cannot answer
        5. Return documents (not the answer)
        """
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        target_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Extract query entities
        query_entities = set()
        if self.use_entity_ranking:
            query_entities = self._extract_query_entities(query)

        # For tracking best result
        best_answer = None
        best_answer_valid = False

        for retry in range(self.max_retries + 1):
            current_k = retrieve_k + retry * self.retry_increment

            # Step 1: Retrieval
            retrieved_docs = self._hierarchical_retrieve(query, current_k)

            if len(retrieved_docs) == 0:
                if retry < self.max_retries:
                    continue
                return []

            # Step 2: Generate answer
            generated_answer = self._generate_answer(query, retrieved_docs)
            self.logger.debug(f"Attempt {retry + 1} answer: {generated_answer[:100]}...")

            # Step 3: Check if answer is valid
            can_answer, reason = self._check_answer_validity(query, generated_answer)

            if can_answer:
                self.logger.info(f"Success on attempt {retry + 1}")
                best_answer = generated_answer
                best_answer_valid = True
                break

            # Cannot answer - need to retry
            if retry >= self.max_retries:
                self.logger.info(f"Max retries ({self.max_retries}), last answer: {generated_answer[:100]}...")
                best_answer = generated_answer
                best_answer_valid = False
                break

            self.logger.info(f"Attempt {retry + 1}: {reason}, expanding search...")

        # Step 5: Entity-based final ranking
        if self.use_entity_ranking and query_entities:
            # Get fresh retrieval for entity ranking
            final_docs = self._hierarchical_retrieve(query, retrieve_k * 2)
            return self._entity_ranking(query, query_entities, final_docs, target_k)

        return retrieved_docs[:target_k]
