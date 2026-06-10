"""
多视角 Chunk 生成器

为每个 chunk 生成多种表示：
1. 原始视角：直接使用原始文本
2. 摘要视角：LLM 生成简洁摘要
3. 问答视角：LLM 生成问答对
4. 关键词视角：提取关键词
"""

import json
import re
from typing import List, Dict
from tqdm import tqdm


class MultiViewChunkGenerator:
    """
    为 chunk 生成多视角表示
    """

    # 摘要 Prompt
    SUMMARY_PROMPT = """请为以下电力技术文档生成一段简洁的摘要（50字以内）：

{document}

摘要："""

    # 问答对 Prompt
    QA_PROMPT = """基于以下文档，生成一个可能的问题和它的答案。

文档：{document}

请以 JSON 格式输出：
{{
    "question": "生成的问题",
    "answer": "基于文档的答案"
}}

只返回 JSON，不要其他内容。"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def generate_views(self, chunk: Dict, chunk_id: str) -> List[Dict]:
        """
        为 chunk 生成多种视角
        """
        original_content = chunk.get("content", "")
        title = chunk.get("title", "")
        parent_title = chunk.get("parent_title", "")

        views = []

        # 视角1: 原始文本
        views.append({
            "id": f"{chunk_id}_original",
            "view": "original",
            "content": original_content,
            "metadata": {
                "chunk_id": chunk_id,
                "title": title,
                "parent_title": parent_title,
                "view_type": "original"
            }
        })

        # 视角2: 生成摘要
        summary = self._generate_summary(original_content)
        if summary:
            views.append({
                "id": f"{chunk_id}_summary",
                "view": "summary",
                "content": summary,
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": title,
                    "parent_title": parent_title,
                    "view_type": "summary"
                }
            })

        # 视角3: 生成问答对
        qa = self._generate_qa(original_content)
        if qa:
            views.append({
                "id": f"{chunk_id}_qa",
                "view": "qa",
                "content": f"Q: {qa['question']}\nA: {qa['answer']}",
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": title,
                    "parent_title": parent_title,
                    "view_type": "qa"
                }
            })

        # 视角4: 关键词
        keywords = self._extract_keywords(original_content)
        views.append({
            "id": f"{chunk_id}_keywords",
            "view": "keywords",
            "content": " ".join(keywords),
            "metadata": {
                "chunk_id": chunk_id,
                "title": title,
                "parent_title": parent_title,
                "view_type": "keywords"
            }
        })

        return views

    def _generate_summary(self, content: str) -> str:
        """生成摘要"""
        try:
            prompt = self.SUMMARY_PROMPT.format(document=content[:2000])
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate_content_with_messages(
                messages, model="qwen-plus-latest", temperature=0.3
            )
            return response.strip()
        except Exception:
            return None

    def _generate_qa(self, content: str) -> Dict:
        """生成问答对"""
        try:
            prompt = self.QA_PROMPT.format(document=content[:2000])
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate_content_with_messages(
                messages, model="qwen-plus-latest", temperature=0.5
            )

            # 解析 JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                qa = json.loads(response[start:end])
                return qa
        except Exception:
            pass
        return None

    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单提取名词和技术术语
        patterns = [
            r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*',  # 英文术语
            r'[\u4e00-\u9fa5]{2,}',  # 中文词
        ]

        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            # 过滤停用词
            stopwords = {'的', '是', '在', '和', '与', '或', '为', '了', '以及', '以及'}
            keywords.extend([m for m in matches if m not in stopwords])

        # 去重并返回前10个
        return list(set(keywords))[:10]


def process_chunks_with_views(
    input_path: str,
    output_path: str,
    llm_client
):
    """
    处理 chunks，生成多视角版本
    """
    from pikerag.utils.data_protocol_utils import load_chunks_from_jsonl

    # 加载原始 chunks
    chunks = load_chunks_from_jsonl(input_path)

    # 生成多视角
    generator = MultiViewChunkGenerator(llm_client)
    all_views = []

    for i, chunk in enumerate(tqdm(chunks, desc="生成多视角")):
        views = generator.generate_views(chunk, str(i))
        all_views.extend(views)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for view in all_views:
            f.write(json.dumps(view, ensure_ascii=False) + '\n')

    print(f"生成 {len(all_views)} 个视角表示（原 {len(chunks)} chunks）")


if __name__ == "__main__":
    from pikerag.llm_client import StandardOpenAIClient
    import sys

    llm = StandardOpenAIClient()
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/electricity/chunks_output.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/electricity/chunks_multi_view.jsonl"

    process_chunks_with_views(input_path, output_path, llm)
