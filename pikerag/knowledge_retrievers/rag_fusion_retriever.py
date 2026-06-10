# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
RAG-Fusion Retriever Implementation.

RAG-Fusion combines multi-query generation with RRF (Reciprocal Rank Fusion) fusion.

Strategy:
1. Generate multiple query variations using LLM
2. Perform vector search for each query variation
3. Use RRF to merge and rerank results from all queries
"""

from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chroma_qa_retriever import QaChunkRetriever
from pikerag.utils.logger import Logger


class RagFusionRetriever(QaChunkRetriever):
    """
    RAG-Fusion Retriever with Multi-Query Generation and RRF Fusion.

    Based on the RAG-Fusion paper:
    - Generate N query variations using LLM
    - Retrieve documents for each query
    - Fuse results using Reciprocal Rank Fusion
    """

    name: str = "RagFusionRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # RAG-Fusion configuration
        self.num_queries: int = retriever_config.get("num_queries", 4)
        self.query_generation_llm_config: dict = retriever_config.get("query_generation_llm", {})
        self.rrf_k: int = retriever_config.get("rrf_k", 60)
        self.per_query_k: int = retriever_config.get("per_query_k", 8)

        self._init_components()

    def _init_components(self) -> None:
        """Initialize query generation LLM."""
        from pikerag.utils.config_loader import load_class
        from pikerag.llm_client.base import BaseLLMClient

        self.query_llm_client = None
        self.query_llm_config = {}

        if self.query_generation_llm_config:
            llm_cls = load_class(
                self.query_generation_llm_config.get("module_path", "pikerag.llm_client"),
                self.query_generation_llm_config.get("class_name", "BaseLLMClient")
            )

            # Get llm_config if present
            self.query_llm_config = self.query_generation_llm_config.get("llm_config", {})

            # Get cache config
            cache_config = self.query_generation_llm_config.get("cache_config", {})
            auto_dump = cache_config.get("auto_dump", True)

            # Create client logger
            client_logger = Logger(name="rag_fusion_client", dump_mode="a", dump_folder=self._log_dir)

            self.query_llm_client = llm_cls(
                location=None,
                auto_dump=auto_dump,
                logger=client_logger,
                llm_config=self.query_llm_config,
                **self.query_generation_llm_config.get("args", {}),
            )
            self.logger.info("Query generation LLM initialized for RAG-Fusion")

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _generate_query_variations(self, original_query: str) -> List[str]:
        """Generate multiple query variations using LLM."""
        if self.query_llm_client is None:
            return [original_query]

        prompt = f"""你是一个查询改写专家。请为以下问题生成 {self.num_queries - 1} 个不同的改写版本。

原始问题: {original_query}

请生成 {self.num_queries - 1} 个改写问题（用编号分隔）：
1. """

        try:
            response = self.query_llm_client.generate_content_with_messages(
                [{"role": "user", "content": prompt}],
                **self.query_llm_config
            )
            content = response.strip()

            variations = [original_query]
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    sub_query = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('-').strip()
                    if sub_query and sub_query != original_query:
                        variations.append(sub_query)

            if len(variations) == 0:
                variations = [original_query]

            return variations[:self.num_queries]

        except Exception as e:
            self.logger.warning(f"Query generation failed: {e}")
            return [original_query]

    def _rrf_fuse(self, ranking_lists: List[List[Tuple[str, float]]], k: int = None) -> List[Tuple[str, float]]:
        """Fuse multiple ranking lists using RRF (Reciprocal Rank Fusion)."""
        if k is None:
            k = self.rrf_k

        combined_scores: Dict[str, float] = {}

        for ranking_list in ranking_lists:
            for rank, (doc_content, _) in enumerate(ranking_list):
                rrf_score = 1.0 / (k + rank + 1)
                combined_scores[doc_content] = combined_scores.get(doc_content, 0) + rrf_score

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in sorted_docs]

    def retrieve_contents_by_query(self, query: str, retrieve_id: str = "", **kwargs) -> List[str]:
        """Retrieve contents using RAG-Fusion strategy."""
        query_variations = self._generate_query_variations(query)
        self.logger.debug(f"RAG-Fusion: {len(query_variations)} queries generated", tag=self.name)

        if len(query_variations) == 1:
            return super().retrieve_contents_by_query(query, retrieve_id, **kwargs)

        all_rankings: List[List[Tuple[str, float]]] = []
        for i, q in enumerate(query_variations):
            self.logger.debug(f"  Query {i+1}/{len(query_variations)}: {q[:50]}...", tag=self.name)
            docs_with_scores = self._get_doc_and_score_with_query(q, retrieve_id, retrieve_k=self.per_query_k)
            rankings = [(doc.page_content, score) for doc, score in docs_with_scores]
            all_rankings.append(rankings)

        fused_results = self._rrf_fuse(all_rankings)
        final_contents = [doc for doc, _ in fused_results[:self.retrieve_k]]

        self.logger.debug(f"{retrieve_id}: RAG-Fusion returned {len(final_contents)} chunks", tag=self.name)

        return final_contents
