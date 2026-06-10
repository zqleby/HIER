# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
HyDE (Hypothetical Document Embeddings) Retriever Implementation.

HyDE improves retrieval by:
1. Generating a hypothetical answer document using LLM
2. Embedding the hypothetical document (not the query)
3. Using the hypothetical document embedding for similarity search

Based on the HyDE paper: "Precise Zero-shot Dense Retrieval without Relevance Labels"
"""

from typing import List

from pikerag.knowledge_retrievers.chroma_qa_retriever import QaChunkRetriever
from pikerag.utils.logger import Logger


class HydeQaChunkRetriever(QaChunkRetriever):
    """
    HyDE (Hypothetical Document Embeddings) Retriever.

    Strategy:
    1. Generate a hypothetical document that answers the query
    2. Embed the hypothetical document for retrieval
    3. Retrieve documents similar to the hypothetical document

    This bridges the semantic gap between queries and documents.
    """

    name: str = "HydeQaChunkRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        # HyDE configuration
        self.hyde_llm_config: dict = retriever_config.get("hyde_llm", {})
        self.hypothetical_max_tokens: int = retriever_config.get("hypothetical_max_tokens", 500)

        self._init_hyde_components()

    def _init_hyde_components(self) -> None:
        """Initialize HyDE LLM client."""
        from pikerag.utils.config_loader import load_class
        from pikerag.llm_client.base import BaseLLMClient

        self.hyde_llm_client = None
        self.hyde_gen_config = {}

        if self.hyde_llm_config:
            llm_cls = load_class(
                self.hyde_llm_config.get("module_path", "pikerag.llm_client"),
                self.hyde_llm_config.get("class_name", "BaseLLMClient")
            )

            # Get llm_config if present
            self.hyde_gen_config = self.hyde_llm_config.get("llm_config", {})

            # Get cache config
            cache_config = self.hyde_llm_config.get("cache_config", {})
            auto_dump = cache_config.get("auto_dump", True)

            # Create client logger
            client_logger = Logger(name="hyde_client", dump_mode="a", dump_folder=self._log_dir)

            self.hyde_llm_client = llm_cls(
                location=None,
                auto_dump=auto_dump,
                logger=client_logger,
                llm_config=self.hyde_gen_config,
                **self.hyde_llm_config.get("args", {}),
            )
            self.logger.info("HyDE LLM client initialized")

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        if self.hyde_llm_client is None:
            return query

        prompt = f"""请生成一个详细、准确的答案来回答以下问题。
这个答案将用于信息检索，请确保它包含足够的细节和正确的术语。

问题: {query}

假设的答案文档:"""

        try:
            response = self.hyde_llm_client.generate_content_with_messages(
                [{"role": "user", "content": prompt}],
                **self.hyde_gen_config
            )
            content = response.strip()
            return content
        except Exception as e:
            self.logger.warning(f"HyDE document generation failed: {e}")
            return query

    def _get_embedding_func(self):
        """Get the embedding function from config."""
        from pikerag.utils.config_loader import load_embedding_func

        embedding_config = self._retriever_config.get("vector_store", {}).get("embedding_setting", {})
        embedding = load_embedding_func(
            module_path=embedding_config.get("module_path", None),
            class_name=embedding_config.get("class_name", None),
            **embedding_config.get("args", {}),
        )
        return embedding

    def retrieve_contents_by_query(self, query: str, retrieve_id: str = "", **kwargs) -> List[str]:
        """Retrieve contents using HyDE strategy."""
        # Step 1: Generate hypothetical document
        hyde_doc = self._generate_hypothetical_document(query)
        self.logger.debug(f"HyDE generated {len(hyde_doc)} chars", tag=self.name)

        # Step 2: Embed the hypothetical document
        try:
            embedding_func = self._get_embedding_func()
            hyde_embedding = embedding_func.embed_query(hyde_doc)
        except Exception as e:
            self.logger.warning(f"HyDE embedding failed: {e}, falling back to original query")
            return super().retrieve_contents_by_query(query, retrieve_id, **kwargs)

        # Step 3: Retrieve using the hypothetical document embedding
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k)
        retrieve_score_threshold = kwargs.get("retrieve_score_threshold", self.retrieve_score_threshold)

        # Use the ChromaDB's internal _collection for embedding-based search
        infos = self.vector_store._collection.query(
            query_embeddings=hyde_embedding,
            n_results=retrieve_k,
        )

        # Parse results
        contents = []
        if infos.get("documents") and len(infos["documents"]) > 0:
            for i, doc in enumerate(infos["documents"][0]):
                contents.append(doc)

        self.logger.debug(f"{retrieve_id}: HyDE retrieved {len(contents)} chunks", tag=self.name)

        return contents
