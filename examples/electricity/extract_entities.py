# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Entity Extraction Preprocessing Script

Extract entities from each chunk and add them to the metadata.
This prepares data for Entity-Enhanced Hierarchical Retriever.

Usage:
    python examples/electricity/extract_entities.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--batch-size BATCH_SIZE]
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def simple_entity_extraction(content: str, title: str) -> str:
    """Simple regex-based entity extraction as fallback."""
    entities = set()

    # Chinese technical terms: 设备名称、技术参数等
    chinese_terms = re.findall(r'[\w]+(?:技术|系统|装置|设备|参数|规范|标准|协议)', content)
    entities.update([t.strip() for t in chinese_terms])

    # Model numbers: DPS-500G-PPR, etc.
    model_patterns = re.findall(r'[A-Z]{2,}[-\d]*[A-Z]*', content)
    entities.update(model_patterns)

    # GB/T standards
    standards = re.findall(r'GB/T \d+[\.\d-]*', content)
    entities.update(standards)

    # Technical terms with English
    english_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:System|Device|Module|Bus|Interface|Protocol|Standard|Method)\b', content)
    entities.update(english_terms)

    # Extract key terms from title
    title_terms = re.findall(r'[\w]+', title)
    if title_terms:
        entities.update([t for t in title_terms[-5:] if len(t) > 2])

    return ','.join(list(entities)[:20])  # Limit to 20 entities


def extract_entities_with_openai(content: str, title: str, client: OpenAI, model: str) -> str:
    """Extract entities from a chunk using OpenAI API."""
    prompt = f"""从以下技术文档中提取所有实体（人名、地名、机构名、设备名、技术术语、产品型号、标准规范、专有名词等）。
只返回逗号分隔的实体列表，不要其他内容。

标题: {title}
内容: {content[:1500]}

实体:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        entities_str = response.choices[0].message.content.strip()
        # Parse entities
        entities = [e.strip() for e in entities_str.split(',') if e.strip() and len(e.strip()) > 1]
        return ','.join(entities)
    except Exception as e:
        print(f"OpenAI API extraction failed: {e}")
        return ""


def process_chunk(chunk: dict, client, model: str, use_api: bool) -> dict:
    """Process a single chunk to extract entities."""
    content = chunk.get("content", "")
    title = chunk.get("title", "")

    if use_api and client:
        entities = extract_entities_with_openai(content, title, client, model)
        if not entities:
            entities = simple_entity_extraction(content, title)
    else:
        entities = simple_entity_extraction(content, title)

    # Add entities to chunk
    chunk["entities"] = entities

    return chunk


def main():
    parser = argparse.ArgumentParser(description="Extract entities from chunks")
    parser.add_argument("--input", type=str, default="data/electricity/chunks_output.jsonl",
                       help="Input file path")
    parser.add_argument("--output", type=str, default="data/electricity/chunks_with_entities.jsonl",
                       help="Output file path")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of chunks to process in parallel")
    parser.add_argument("--use-api", action="store_true", default=True,
                       help="Use OpenAI API for entity extraction")
    parser.add_argument("--model", type=str, default="qwen-plus",
                       help="API model name")

    args = parser.parse_args()

    # Initialize OpenAI client
    client = None
    if args.use_api:
        api_key = ""
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment. Using simple extraction.")
            args.use_api = False
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"Using API: {base_url} with model: {args.model}")

    # Read input file
    print(f"Reading input file: {args.input}")
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Total chunks: {len(chunks)}")

    # Process chunks
    print(f"Extracting entities...")
    processed_chunks = []

    if args.use_api and client:
        # Use parallel processing for API extraction
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {executor.submit(process_chunk, chunk, client, args.model, True): chunk for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                processed_chunks.append(future.result())
    else:
        # Sequential simple extraction
        for chunk in tqdm(chunks, desc="Extracting"):
            processed_chunks.append(process_chunk(chunk, None, args.model, False))

    # Sort by original order (by chunk_id)
    processed_chunks.sort(key=lambda x: x.get("chunk_id", ""))

    # Write output file
    print(f"Writing output file: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for chunk in processed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Print statistics
    total_entities = sum(len(c.get("entities", "").split(',')) for c in processed_chunks if c.get("entities"))
    print(f"Extracted {total_entities} entities from {len(processed_chunks)} chunks")
    print(f"Average entities per chunk: {total_entities / len(processed_chunks):.2f}")

    # Show sample
    print("\nSample output:")
    for i, chunk in enumerate(processed_chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.get('chunk_id', 'N/A')}")
        print(f"  Title: {chunk.get('title', 'N/A')[:60]}...")
        print(f"  Entities: {chunk.get('entities', 'N/A')}")


if __name__ == "__main__":
    main()
