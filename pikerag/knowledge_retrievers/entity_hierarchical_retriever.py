# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entity-Enhanced Hierarchical Retriever

Strategy:
1. Indexing: Use LLM to extract entities from each document, store in metadata
2. Retrieval: Extract entities from query, use hierarchical retrieval
3. Ranking: Sort by number of matched entities (not similarity score)

This improves recall for entity-centric queries.
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import re

from langchain_core.documents import Document

from pikerag.knowledge_retrievers.hierarchical_retriever import HierarchicalChunkRetriever
from pikerag.utils.logger import Logger


class EntityHierarchicalRetriever(HierarchicalChunkRetriever):
    """
    Hierarchical Retriever with Entity-based Ranking.

    Pipeline:
    1. Extract entities from document during indexing (done externally)
    2. Extract entities from query
    3. Hierarchical retrieval (unchanged)
    4. Rank by matched entity count (instead of similarity)
    """

    name: str = "EntityHierarchicalRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Entity extraction config
        self.entity_extraction_enabled: bool = retriever_config.get("entity_extraction_enabled", True)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_field_name: str = retriever_config.get("entity_field_name", "entities")

        # Ranking weights
        self.entity_weight: float = retriever_config.get("entity_weight", 0.6)
        self.similarity_weight: float = retriever_config.get("similarity_weight", 0.4)

        # Entity matching
        self.normalize_entities: bool = retriever_config.get("normalize_entities", True)

        self._init_components()

    def _init_components(self) -> None:
        """Initialize entity extraction LLM."""
        self.entity_llm = None
        if self.entity_extraction_enabled and self.entity_llm_config:
            from pikerag.utils.config_loader import load_class
            llm_cls = load_class(
                self.entity_llm_config.get("module_path", "pikerag.llm_client"),
                self.entity_llm_config.get("class_name", "BaseLLMClient")
            )
            self.entity_llm = llm_cls(self.entity_llm_config.get("args", {}))
            self.logger.info("Entity extraction LLM initialized")

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using LLM."""
        if self.entity_llm is None:
            # Fallback: simple keyword extraction
            return self._simple_entity_extraction(text)

        prompt = f"""从以下文本中提取所有实体（如人名、地名、机构名、技术术语、专有名词等）。
只返回实体列表，用逗号分隔，不要其他解释。

文本: {text}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)

            # Parse entities
            entities = [e.strip() for e in content.split(',') if e.strip()]
            return entities
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            return self._simple_entity_extraction(text)

    def _simple_entity_extraction(self, text: str) -> List[str]:
        """Simple entity extraction using regex (fallback)."""
        # Extract capitalized words and technical terms
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
            r'\b\w+(?:技术|系统|方法|协议|标准|规范|理论)\b',  # Chinese technical terms
            r'\b[A-Z]{2,}(?:\s+[A-Z]{{2,}})*\b',  # Acronyms
        ]

        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)

        # Normalize
        return [self._normalize_entity(e) for e in entities]

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity for matching."""
        if not self.normalize_entities:
            return entity
        return entity.lower().strip()

    def _extract_query_entities(self, query: str) -> Set[str]:
        """Extract entities from query."""
        entities = self._extract_entities(query)
        return set([self._normalize_entity(e) for e in entities])

    def _get_doc_entities(self, doc: Document) -> Set[str]:
        """Get entities from document metadata."""
        entities_str = doc.metadata.get(self.entity_field_name, "")
        if isinstance(entities_str, list):
            entity_list = entities_str
        elif isinstance(entities_str, str) and entities_str:
            entity_list = [e.strip() for e in entities_str.split(',')]
        else:
            # Extract from content on-the-fly
            entity_list = self._extract_entities(doc.page_content)
        return set([self._normalize_entity(e) for e in entity_list])

    def _count_matched_entities(self, query_entities: Set[str], doc_entities: Set[str]) -> int:
        """Count number of matched entities."""
        if not query_entities:
            return 0
        matched = query_entities.intersection(doc_entities)
        return len(matched)

    def _entity_based_ranking(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
        retrieve_k: int
    ) -> List[str]:
        """Rank documents by matched entity count."""
        # Extract query entities
        query_entities = self._extract_query_entities(query)

        self.logger.debug(f"Query entities: {query_entities}")

        if not query_entities:
            # Fallback to similarity ranking
            return [doc.metadata.get("content", doc.page_content) for doc, _ in docs[:retrieve_k]]

        # Score each document
        scored = []
        for doc, sim_score in docs:
            doc_entities = self._get_doc_entities(doc)
            matched_count = self._count_matched_entities(query_entities, doc_entities)

            # Combined score: entity matches + similarity
            combined_score = matched_count * self.entity_weight + sim_score * self.similarity_weight

            content = doc.metadata.get("content", doc.page_content)
            scored.append((content, matched_count, combined_score, doc))

        # Sort by entity matches first, then combined score
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

        self.logger.debug(f"Top doc matched entities: {scored[0][1] if scored else 0}")

        return [item[0] for item in scored[:retrieve_k]]

    def _check_hierarchy_and_rank(
        self,
        query: str,
        initial_docs: List[Tuple[Document, float]],
        retrieve_k: int
    ) -> List[str]:
        """Hierarchical expansion + entity ranking."""
        # Check expansion
        parent_to_expand = self._check_expansion_criteria(initial_docs)

        if parent_to_expand:
            expanded_docs = self._expand_to_section(parent_to_expand)

            if len(expanded_docs) > 0:
                # Re-score expanded docs
                from pikerag.utils.config_loader import load_embedding_func
                embedding_config = self._retriever_config.get("vector_store", {}).get("embedding_setting", {})
                embedding = load_embedding_func(
                    module_path=embedding_config.get("module_path", None),
                    class_name=embedding_config.get("class_name", None),
                    **embedding_config.get("args", {}),
                )

                query_embedding = embedding.embed_query(query)
                scored = []

                for doc in expanded_docs:
                    doc_embedding = embedding.embed_query(doc.page_content)
                    sim_score = float(np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
                    ))
                    scored.append((doc, sim_score))

                scored.sort(key=lambda x: x[1], reverse=True)
                docs_to_rank = scored[:retrieve_k * 2]
            else:
                docs_to_rank = initial_docs
        else:
            docs_to_rank = initial_docs[:retrieve_k * 2]

        # Entity-based ranking
        return self._entity_based_ranking(query, docs_to_rank, retrieve_k)

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Retrieve using hierarchical + entity-based ranking.
        """
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k * 2)

        # Initial retrieval
        initial_docs = self._get_doc_and_score_with_query(query, retrieve_k=retrieve_k)

        if len(initial_docs) == 0:
            return []

        # Hierarchical expansion + Entity ranking
        return self._check_hierarchy_and_rank(query, initial_docs, retrieve_k)


class EntityMultiQueryHierarchicalRetriever(HierarchicalChunkRetriever):
    """
    Combines:
    1. Query Decomposition (Multi-Query)
    2. Entity-based Ranking
    3. Hierarchical Retrieval
    """

    name: str = "EntityMultiQueryHierarchicalRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # Multi-query config
        self.num_sub_queries: int = retriever_config.get("num_sub_queries", 3)
        self.decomposition_llm_config: dict = retriever_config.get("decomposition_llm", {})

        # Entity config
        self.entity_extraction_enabled: bool = retriever_config.get("entity_extraction_enabled", True)
        self.entity_llm_config: dict = retriever_config.get("entity_llm", {})
        self.entity_field_name: str = retriever_config.get("entity_field_name", "entities")
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

        # Entity extraction LLM
        self.entity_llm = None
        if self.entity_extraction_enabled and self.entity_llm_config:
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

    def _extract_query_entities(self, query: str) -> Set[str]:
        """Extract entities from query."""
        if self.entity_llm is None:
            # Simple extraction
            import re
            patterns = [r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', r'\b\w+(?:技术|系统|方法)\b']
            entities = set()
            for pattern in patterns:
                matches = re.findall(pattern, query)
                entities.update([e.lower() for e in matches])
            return entities

        prompt = f"""从以下问题中提取所有实体，只返回逗号分隔的列表。

问题: {query}

实体:"""

        try:
            response = self.entity_llm.generate([{"role": "user", "content": prompt}])
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            entities = [e.strip().lower() for e in content.split(',') if e.strip()]
            return set(entities)
        except:
            return set()

    def _get_doc_entities(self, doc: Document) -> Set[str]:
        """Get entities from document."""
        entities_str = doc.metadata.get(self.entity_field_name, "")
        if isinstance(entities_str, list):
            return set([e.lower() for e in entities_str])
        elif isinstance(entities_str, str) and entities_str:
            return set([e.strip().lower() for e in entities_str.split(',')])
        return set()

    def _retrieve_and_rank(
        self,
        query: str,
        query_entities: Set[str],
        retrieve_k: int
    ) -> List[Tuple[str, float]]:
        """Single query retrieval + entity ranking."""
        # Hierarchical retrieval
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
                    sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8)
                    content = doc.metadata.get("content", doc.page_content)
                    scored.append((content, sim, doc))
                scored.sort(key=lambda x: x[1], reverse=True)
                docs_to_rank = scored[:retrieve_k]
            else:
                docs_to_rank = [(doc, sim) for doc, sim in initial_docs[:retrieve_k]]
        else:
            docs_to_rank = [(doc.metadata.get("content", doc.page_content), sim) for doc, sim in initial_docs[:retrieve_k]]

        # Entity ranking
        final_ranking = []
        for content, sim_score, doc in docs_to_rank:
            doc_entities = self._get_doc_entities(doc)
            matched = len(query_entities.intersection(doc_entities))
            combined = matched * self.entity_weight + sim_score * self.similarity_weight
            final_ranking.append((content, matched, combined))

        final_ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(c, 0) for c, m, s in final_ranking]

    def _rrf_fuse(self, rankings: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """RRF fusion."""
        combined: Dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc, _) in enumerate(ranking):
                rrf = 1.0 / (self.rrf_k + rank + 1)
                combined[doc] = combined.get(doc, 0) + rrf
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """Multi-query + Entity-based ranking.

        Pipeline:
        1. Decompose query into sub-queries
        2. Retrieve for each sub-query (hierarchical)
        3. RRF fusion
        4. Entity-based final ranking
        """
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)

        # Extract entities from original query
        query_entities = self._extract_query_entities(query)
        self.logger.debug(f"Query entities: {query_entities}")

        # Decompose query
        sub_queries = self._decompose_query(query)

        # Retrieve for each sub-query (keep doc objects for entity extraction)
        all_docs: List[Tuple[Document, float, str]] = []  # (doc, score, sub_query)
        for sq in sub_queries:
            # Hierarchical retrieval
            initial_docs = self._get_doc_with_query(sq, self.vector_store, retrieve_k * 2, None)
            if len(initial_docs) == 0:
                continue

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
                    for doc in expanded_docs:
                        doc_emb = embedding.embed_query(doc.page_content)
                        sim = float(np.dot(query_emb, doc_emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
                        ))
                        all_docs.append((doc, sim, sq))
                else:
                    for doc, sim in initial_docs:
                        all_docs.append((doc, sim, sq))
            else:
                for doc, sim in initial_docs:
                    all_docs.append((doc, sim, sq))

        if not all_docs:
            return []

        # Deduplicate by content
        seen = set()
        unique_docs = []
        for doc, sim, sq in all_docs:
            content = doc.metadata.get("content", doc.page_content)
            if content not in seen:
                seen.add(content)
                unique_docs.append((doc, sim))

        # RRF fusion to get candidate documents
        rrf_scores: Dict[str, float] = {}
        for doc, sim in unique_docs:
            content = doc.metadata.get("content", doc.page_content)
            # Simple similarity-based initial score
            rrf_scores[content] = sim

        # Sort by RRF score first to get candidates
        candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_contents = [doc for doc, _ in candidates[:retrieve_k * 2]]

        # Get document objects for entity ranking
        docs_for_entity_ranking = []
        for doc, sim in unique_docs:
            content = doc.metadata.get("content", doc.page_content)
            if content in candidate_contents:
                docs_for_entity_ranking.append((doc, sim))

        # Entity-based final ranking
        final_ranked = []
        for doc, sim_score in docs_for_entity_ranking:
            doc_entities = self._get_doc_entities(doc)
            matched = len(query_entities.intersection(doc_entities))
            combined = matched * self.entity_weight + sim_score * self.similarity_weight
            content = doc.metadata.get("content", doc.page_content)
            final_ranked.append((content, matched, combined, doc))

        # Sort by matched entities first, then combined score
        final_ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)

        self.logger.debug(f"Top doc matched entities: {final_ranked[0][1] if final_ranked else 0}")

        return [item[0] for item in final_ranked[:retrieve_k]]
