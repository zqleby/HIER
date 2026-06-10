#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the hierarchical RAG data loading.

This script verifies:
1. Rich text creation with title + content
2. Hierarchy parsing for parent title extraction
3. Document metadata structure
"""

import sys
import json

sys.path.insert(0, r"D:\xj\rag\PIKE-RAG-main")

from pikerag.utils.data_protocol_utils import load_ids_and_chunks_with_hierarchy


def test_data_loading():
    """Test the hierarchical data loading function."""
    print("=" * 60)
    print("Testing Hierarchical Data Loading")
    print("=" * 60)

    filepath = r"D:\xj\rag\PIKE-RAG-main\data\electricity\chunks_output.jsonl"

    # Load chunks with hierarchy
    chunk_ids, chunk_docs = load_ids_and_chunks_with_hierarchy(
        filepath=filepath,
        title_prefix="[层级]",
        content_prefix="[正文]",
    )

    print(f"\nLoaded {len(chunk_ids)} chunks")
    print(f"First 5 chunk IDs: {chunk_ids[:5]}")

    # Show sample documents
    print("\n" + "-" * 60)
    print("Sample Documents:")
    print("-" * 60)

    for i, doc in enumerate(chunk_docs[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {doc.metadata['id']}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Root: {doc.metadata['root']}")
        print(f"Section 1: {doc.metadata['section_1']}")
        print(f"Section 2: {doc.metadata['section_2']}")
        print(f"Parent Title: {doc.metadata['parent_title']}")
        print(f"\nRich Text (first 200 chars):")
        print(doc.page_content[:200] + "...")

    # Verify parent_title extraction
    print("\n" + "-" * 60)
    print("Parent Title Extraction Verification:")
    print("-" * 60)

    parent_titles = set(doc.metadata["parent_title"] for doc in chunk_docs)
    print(f"Unique parent titles: {len(parent_titles)}")
    print("\nSample parent titles:")
    for pt in list(parent_titles)[:5]:
        print(f"  - {pt}")

    # Count chunks per parent section
    print("\n" + "-" * 60)
    print("Chunks per Parent Section (top 10):")
    print("-" * 60)

    from collections import Counter
    parent_counts = Counter(doc.metadata["parent_title"] for doc in chunk_docs)
    for pt, count in parent_counts.most_common(10):
        print(f"  {count:3d} chunks: {pt}")

    print("\n" + "=" * 60)
    print("Data Loading Test PASSED!")
    print("=" * 60)

    return True


def test_rich_text_format():
    """Verify the rich text format is correct."""
    print("\n" + "=" * 60)
    print("Testing Rich Text Format")
    print("=" * 60)

    filepath = r"D:\xj\rag\PIKE-RAG-main\data\electricity\chunks_output.jsonl"
    _, chunk_docs = load_ids_and_chunks_with_hierarchy(filepath)

    # Check format: "[层级] {title} [正文] {content}"
    for doc in chunk_docs[:5]:
        rich_text = doc.page_content
        metadata = doc.metadata

        # Verify rich_text contains title and content
        assert metadata["title"] in rich_text, "Title not found in rich_text"
        assert metadata["content"] in rich_text, "Content not found in rich_text"
        assert "[层级]" in rich_text, "Title prefix not found"
        assert "[正文]" in rich_text, "Content prefix not found"

        # Verify metadata
        assert "root" in metadata, "root field missing"
        assert "section_1" in metadata, "section_1 field missing"
        assert "parent_title" in metadata, "parent_title field missing"

    print("All rich_text format checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_data_loading()
        test_rich_text_format()
        print("\n[SUCCESS] All tests passed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
