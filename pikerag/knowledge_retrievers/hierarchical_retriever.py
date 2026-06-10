# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Hierarchical Retrieval Retriever for Electricity RAG.

This retriever implements:
1. Multi-Aspect Indexing: Rich text with title + content for embedding
2. Hierarchical Retrieval: Section-based expansion when majority of results share same parent

Key Features:
- Initial retrieval uses rich_text (title + content) for similarity
- If majority of top-K results share same parent title, expand to all sibling chunks
- Supports both direct retrieval and hierarchical expansion
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document

from pikerag.knowledge_retrievers.base_qa_retriever import BaseQaRetriever
from pikerag.knowledge_retrievers.mixins.chroma_mixin import ChromaMetaType, ChromaMixin
from pikerag.utils.logger import Logger


class HierarchicalChunkRetriever(BaseQaRetriever, ChromaMixin):
    """
    A retriever that supports hierarchical retrieval strategy.

    Retrieval Strategy:
    1. Initial retrieval using rich_text (title + content) embedding
    2. Check if majority of top-K results share the same parent title
    3. If yes, expand retrieval to all chunks under that parent section

    Configuration:
        - retrieve_k: Number of chunks to retrieve initially (default: 8)
        - hierarchy_threshold: Ratio threshold for expansion (default: 0.6 = 60%)
        - hierarchy_meta_name: Metadata field for parent title grouping (default: "parent_title")
    """

    name: str = "HierarchicalChunkRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        self._init_hierarchy_config()
        self._load_vector_store()
        self._init_chroma_mixin()

        self.logger = Logger(name=self.name, dump_mode="w", dump_folder=self._log_dir)

    def _init_hierarchy_config(self) -> None:
        """Initialize hierarchical retrieval configuration."""
        self.hierarchy_threshold: float = self._retriever_config.get("hierarchy_threshold", 0.6)
        self.hierarchy_meta_name: str = self._retriever_config.get("hierarchy_meta_name", "parent_title")
        self.expand_to_section: bool = self._retriever_config.get("expand_to_section", True)

    def _load_vector_store(self) -> None:
        """Load the Chroma vector store."""
        assert "vector_store" in self._retriever_config, "vector_store must be defined in retriever part!"
        vector_store_config = self._retriever_config["vector_store"]

        from pikerag.knowledge_retrievers.chroma_qa_retriever import load_vector_store_from_configs

        self.vector_store: Chroma = load_vector_store_from_configs(
            vector_store_config=vector_store_config,
            embedding_config=vector_store_config.get("embedding_setting", {}),
            collection_name=vector_store_config.get("collection_name", self.name),
            persist_directory=vector_store_config.get("persist_directory", self._log_dir),
        )

    def _get_doc_and_score_with_query(
        self, query: str, retrieve_k: Optional[int] = None, score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Get relevant documents with scores for a query."""
        if retrieve_k is None:
            retrieve_k = self.retrieve_k
        if score_threshold is None:
            score_threshold = self.retrieve_score_threshold

        return self._get_doc_with_query(query, self.vector_store, retrieve_k, score_threshold)

    def _get_parent_title(self, doc: Document) -> str:
        """Extract parent title from document metadata for grouping."""
        return doc.metadata.get(self.hierarchy_meta_name, doc.metadata.get("title", ""))

    def _check_expansion_criteria(self, doc_infos: List[Tuple[Document, float]]) -> Optional[str]:
        """
        Check if hierarchical expansion should be triggered.

        Args:
            doc_infos: List of (document, score) tuples from initial retrieval

        Returns:
            Parent title to expand if criteria met, None otherwise
        """
        if not self.expand_to_section or len(doc_infos) == 0:
            return None

        # Extract parent titles from top results
        parent_titles = [self._get_parent_title(doc) for doc, _ in doc_infos]

        # Count occurrences
        title_counts = Counter(parent_titles)
        most_common_title, most_common_count = title_counts.most_common(1)[0]

        # Calculate ratio
        ratio = most_common_count / len(doc_infos)

        self.logger.debug(
            msg=f"Hierarchy analysis: {most_common_count}/{len(doc_infos)} ({ratio:.2%}) "
                f"share parent '{most_common_title}'",
            tag=self.name,
        )

        # Trigger expansion if majority threshold met
        if ratio >= self.hierarchy_threshold:
            return most_common_title

        return None

    def _expand_to_section(self, parent_title: str) -> List[Document]:
        """
        Retrieve all chunks under a given parent title section.

        Args:
            parent_title: The parent title to expand to

        Returns:
            List of all documents in the section
        """
        _, chunks, _ = self._get_infos_with_given_meta(
            store=self.vector_store,
            meta_name=self.hierarchy_meta_name,
            meta_value=parent_title,
        )

        # Reconstruct Document objects with metadata
        documents = []
        metadatas = self.vector_store.get(where={self.hierarchy_meta_name: parent_title})["metadatas"]
        ids = self.vector_store.get(where={self.hierarchy_meta_name: parent_title})["ids"]

        for chunk, meta, doc_id in zip(chunks, metadatas, ids):
            documents.append(Document(page_content=chunk, metadata={**meta, "id": doc_id}))

        self.logger.debug(
            msg=f"Expanded to section '{parent_title}': {len(documents)} chunks retrieved",
            tag=self.name,
        )

        return documents

    def _get_relevant_strings(
        self, doc_infos: List[Tuple[Document, float]], retrieve_id: str = ""
    ) -> List[str]:
        """Extract content strings from documents."""
        # Return original content, not rich_text
        contents = [doc.metadata.get("content", doc.page_content) for doc, _ in doc_infos]
        return contents

    def retrieve_contents_by_query(
        self, query: str, retrieve_id: str = "", **kwargs
    ) -> List[str]:
        """
        Retrieve contents using hierarchical strategy.

        Args:
            query: The query string
            retrieve_id: Optional identifier for logging
            **kwargs: Additional arguments (retrieve_k, etc.)

        Returns:
            List of retrieved content strings
        """
        # Initial retrieval
        retrieve_k = kwargs.get("retrieve_k", self.retrieve_k * 2)  # Retrieve more for expansion analysis
        # 16个文档
        initial_docs = self._get_doc_and_score_with_query(query, retrieve_k=retrieve_k)

        if len(initial_docs) == 0:
            return []

        # Check if expansion should be triggered
        parent_to_expand = self._check_expansion_criteria(initial_docs)

        if parent_to_expand:
            # Expand to full section
            expanded_docs = self._expand_to_section(parent_to_expand)

            # Sort by similarity to query (re-score expanded results)
            if len(expanded_docs) > 0:
                from langchain_core.embeddings import Embeddings
                from pikerag.utils.config_loader import load_embedding_func

                embedding_config = self._retriever_config.get("vector_store", {}).get("embedding_setting", {})
                embedding = load_embedding_func(
                    module_path=embedding_config.get("module_path", None),
                    class_name=embedding_config.get("class_name", None),
                    **embedding_config.get("args", {}),
                )

                query_embedding = embedding.embed_query(query)
                scored_docs = []

                for doc in expanded_docs:
                    doc_embedding = embedding.embed_query(doc.page_content)
                    import numpy as np
                    score = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    scored_docs.append((doc, score))

                # Sort by score
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                # Limit to retrieve_k
                final_docs = scored_docs[:retrieve_k]
            else:
                final_docs = initial_docs[:retrieve_k]
        else:
            # Use initial results, limited to retrieve_k
            final_docs = initial_docs[:retrieve_k]

        return self._get_relevant_strings(final_docs, retrieve_id)

    def retrieve_contents(
        self, qa: "BaseQaData", retrieve_id: str = ""
    ) -> List[str]:
        """
        Retrieve contents for a QA pair.

        Args:
            qa: QA data object with question attribute
            retrieve_id: Optional identifier for logging

        Returns:
            List of retrieved content strings
        """
        return self.retrieve_contents_by_query(qa.question, retrieve_id)


class HierarchicalChunkRetrieverWithMeta(HierarchicalChunkRetriever):
    """
    Extended hierarchical retriever that also supports metadata-based filtering.

    Combines:
    - Hierarchical expansion for section-based retrieval
    - Metadata filtering for additional constraints
    """

    name: str = "HierarchicalChunkRetrieverWithMeta"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        assert "meta_name" in self._retriever_config, f"meta_name must be specified to use {self.name}"
        self._meta_name = self._retriever_config["meta_name"]

    def _get_doc_and_score_with_query(
        self, query: str, retrieve_k: Optional[int] = None, score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Override to add metadata filtering if specified."""
        doc_infos = super()._get_doc_and_score_with_query(query, retrieve_k, score_threshold)

        # Apply metadata filter if specified
        if hasattr(self, "_meta_name") and self._meta_name:
            filter_value = self._retriever_config.get("meta_value")
            if filter_value is not None:
                doc_infos = [
                    (doc, score)
                    for doc, score in doc_infos
                    if doc.metadata.get(self._meta_name) == filter_value
                ]

        return doc_infos
