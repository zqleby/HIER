# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Literal, Tuple

import jsonlines
import pickle
from langchain_core.documents import Document

from pikerag.workflows.common import GenerationQaData


# Used in tagging
def load_chunks_from_jsonl(jsonl_chunk_path: str) -> List[Document]:
    with jsonlines.open(jsonl_chunk_path, "r") as reader:
        chunk_dicts = [obj for obj in reader]

    chunks: List[Document] = [
        Document(
            page_content=chunk_dict["content"],
            metadata={"chunk_id": chunk_dict["chunk_id"], "title": chunk_dict["title"]},
        )
        for chunk_dict in chunk_dicts
    ]
    return chunks


# Used in tagging
def save_chunks_to_jsonl(tagged_chunks: List[Document], dump_path: str) -> None:
    with jsonlines.open(dump_path, "w") as writer:
        for chunk in tagged_chunks:
            chunk_dict = chunk.metadata
            chunk_dict["content"] = chunk.page_content
            writer.write(chunk_dict)
    return


# Used in tagging
def load_chunks_from_pkl(filepath: str) -> List[Document]:
    with open(filepath, "rb") as fin:
        chunks = pickle.load(fin)
    return chunks


# Used in tagging
def save_chunks_to_pkl(chunks: List[Document], filepath: str) -> None:
    with open(filepath, "wb") as fout:
        pickle.dump(chunks, fout)
    return


# Used in QA
def load_testing_suite(filepath: str) -> List[GenerationQaData]:
    testing_suite = []
    with jsonlines.open(filepath, "r") as reader:
        for qa in reader:
            # TODO: update GenerationQaData definition
            metadata = qa["metadata"]
            metadata["id"] = qa["id"]
            metadata["question_type"] = qa["question_type"]
            testing_suite.append(
                GenerationQaData(
                    question=qa["question"],
                    answer_labels=[str(label) for label in qa["answer_labels"]],
                    metadata=qa["metadata"],
                )
            )
    return testing_suite


# Used in QA
def load_ids_and_chunks(filepath: str, atom_tag: str="atom_questions") -> Tuple[List[str], List[Document]]:
    chunk_ids: List[str] = []
    chunk_docs: List[Document] = []
    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            chunk_ids.append(chunk_dict["chunk_id"])
            chunk_docs.append(
                Document(
                    # TODO: check whether to use "content" only of the concatenate of "title" and "content"
                    page_content=chunk_dict["content"],
                    # page_content=f"Title: {chunk_dict['title']}. Content: {chunk_dict['content']}",
                    metadata={
                        "id": chunk_dict["chunk_id"],
                        "title": chunk_dict["title"],
                       # f"{atom_tag}_str": "\n".join(chunk_dict[atom_tag])  # TODO: allow missing
                    }
                )
            )
    return chunk_ids, chunk_docs


# Used in QA
def load_ids_and_atoms(filepath: str, atom_tag: str) -> Tuple[Literal[None], List[Document]]:
    atom_docs: List[Document] = []
    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            for atom in chunk_dict[atom_tag]:
                atom = atom.strip()
                if len(atom) > 0:
                    atom_docs.append(
                        Document(page_content=atom, metadata={"source_chunk_id": chunk_dict["chunk_id"]})
                    )
    return None, atom_docs


# =============================================================================
# Custom Electricity RAG: Multi-Aspect Indexing Functions
# =============================================================================

def load_ids_and_chunks_with_rich_text(
    filepath: str,
    title_prefix: str = "[层级]",
    content_prefix: str = "[正文]",
) -> Tuple[List[str], List[Document]]:
    """
    Load chunks and create rich_text for multi-aspect indexing.

    The rich_text format is: `{title_prefix} {title} {content_prefix} {content}`
    This enhances title keywords (like "极保护", "自诊断") in similarity calculation.

    Args:
        filepath: Path to the chunks jsonl file
        title_prefix: Prefix for title section (default: "[层级]")
        content_prefix: Prefix for content section (default: "[正文]")

    Returns:
        Tuple of (chunk_ids, chunk_docs)
    """
    chunk_ids: List[str] = []
    chunk_docs: List[Document] = []

    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            chunk_id = chunk_dict["chunk_id"]
            title = chunk_dict["title"]
            content = chunk_dict["content"]

            # Create rich_text for enhanced embedding
            rich_text = f"{title_prefix} {title} {content_prefix} {content}"

            chunk_ids.append(chunk_id)
            chunk_docs.append(
                Document(
                    page_content=rich_text,
                    metadata={
                        "id": chunk_id,
                        "title": title,
                        "content": content,  # Store original content for retrieval
                        "rich_text": rich_text,
                    }
                )
            )

    return chunk_ids, chunk_docs


def load_ids_and_chunks_with_hierarchy(
    filepath: str,
    title_prefix: str = "[层级]",
    content_prefix: str = "[正文]",
) -> Tuple[List[str], List[Document]]:
    """
    Load chunks with hierarchical structure information.

    Parses title path to extract hierarchy levels for section-based retrieval.
    Title format example: "DPS-500G-PPR极保护系统技术说明书 > 1. 概述 > 1.2 性能特点 > 1.2.1 高性能硬件架构"

    Args:
        filepath: Path to the chunks jsonl file
        title_prefix: Prefix for title section
        content_prefix: Prefix for content section

    Returns:
        Tuple of (chunk_ids, chunk_docs)
    """
    chunk_ids: List[str] = []
    chunk_docs: List[Document] = []

    def parse_title_hierarchy(title: str) -> dict:
        """Parse title string into hierarchical components."""
        parts = title.split(" > ")
        return {
            "root": parts[0] if len(parts) > 0 else "",
            "section_1": parts[1] if len(parts) > 1 else "",
            "section_2": parts[2] if len(parts) > 2 else "",
            "section_3": parts[3] if len(parts) > 3 else "",
            "full_title": title,
        }

    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            chunk_id = chunk_dict["chunk_id"]
            title = chunk_dict["title"]
            content = chunk_dict["content"]
            # Extract entities if present
            entities = chunk_dict.get("entities", "")

            hierarchy = parse_title_hierarchy(title)

            # Create rich_text for embedding
            rich_text = f"{title_prefix} {title} {content_prefix} {content}"

            chunk_ids.append(chunk_id)
            chunk_docs.append(
                Document(
                    page_content=rich_text,
                    metadata={
                        "id": chunk_id,
                        "title": title,
                        "content": content,
                        "rich_text": rich_text,
                        # Hierarchical metadata for section-based retrieval
                        "root": hierarchy["root"],
                        "section_1": hierarchy["section_1"],
                        "section_2": hierarchy["section_2"],
                        "section_3": hierarchy["section_3"],
                        "parent_title": " > ".join(title.split(" > ")[:-1]) if " > " in title else title,
                        # Entity metadata for entity-based retrieval
                        "entities": entities,
                    }
                )
            )

    return chunk_ids, chunk_docs
