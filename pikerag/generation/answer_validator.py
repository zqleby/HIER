"""
答案验证器 - 创新点

使用 LLM 验证答案是否：
1. 与检索到的文档一致
2. 正确回答了问题
3. 没有产生幻觉
"""

import json
from typing import List, Dict


class AnswerValidator:
    """
    答案验证器

    在生成答案后，验证答案的质量
    """

    VALIDATION_PROMPT = """你是一个答案质量评估专家。

请根据检索到的文档，评估以下答案的质量。

问题：{question}

检索到的文档：
{documents}

模型生成的答案：
{answer}

评估维度：
1. 一致性：答案是否与文档内容一致？（是/否）
2. 准确性：答案是否正确回答了问题？（是/否）
3. 完整性：答案是否包含了所有重要信息？（是/部分/否）
4. 幻觉检测：答案中是否有文档中没有的信息？（是/否）

请以 JSON 格式输出评估结果：
{{
    "consistent": true/false,
    "accurate": true/false,
    "complete": "是/部分/否",
    "has_hallucination": true/false,
    "quality_score": 0.0-1.0,
    "feedback": "简短反馈"
}}

只返回 JSON。"""

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def validate(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[str]
    ) -> Dict:
        """
        验证答案质量
        """
        # 截断过长的文档
        docs_text = "\n\n".join([
            doc[:500] for doc in retrieved_documents[:3]
        ])

        prompt = self.VALIDATION_PROMPT.format(
            question=question,
            documents=docs_text,
            answer=answer
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate_content_with_messages(
                messages, model="qwen-plus-latest", temperature=0
            )

            # 解析 JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(response[start:end])
                return result

        except Exception as e:
            if self.logger:
                self.logger.warning(f"答案验证失败: {e}")
            return {"error": str(e)}

        return {"error": "解析失败"}

    def quick_validate(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[str]
    ) -> bool:
        """
        快速验证（只返回是否通过）
        """
        result = self.validate(question, answer, retrieved_documents)
        score = result.get("quality_score", 0)
        return score >= 0.7


class SelfRefinementGenerator:
    """
    自反思答案生成器

    如果验证发现问题，自动改进答案
    """

    REFINE_PROMPT = """以下答案未通过质量检查，请改进。

原始问题：{question}

原始答案：{answer}

验证反馈：{feedback}

请根据反馈，生成一个改进后的答案。

要求：
1. 纠正错误的或不准确的信息
2. 确保答案与检索到的文档一致
3. 确保完整回答问题

改进后的答案："""

    def __init__(self, llm_client, validator: AnswerValidator = None, logger=None):
        self.llm = llm_client
        self.validator = validator or AnswerValidator(llm_client, logger)
        self.logger = logger

    def generate_with_validation(
        self,
        question: str,
        initial_answer: str,
        retrieved_documents: List[str],
        max_iterations: int = 2
    ) -> Dict:
        """
        带验证的答案生成
        """
        current_answer = initial_answer
        validation = None

        for iteration in range(max_iterations):
            # 验证
            validation = self.validator.validate(
                question,
                current_answer,
                retrieved_documents
            )

            # 如果质量合格
            score = validation.get("quality_score", 0)
            if score >= 0.8:
                if self.logger:
                    self.logger.debug(
                        msg=f"答案验证通过 (迭代 {iteration})",
                        tag="SelfRefinement"
                    )
                return {
                    "answer": current_answer,
                    "valid": True,
                    "score": score,
                    "iterations": iteration,
                    "validation": validation
                }

            # 如果需要改进
            feedback = validation.get("feedback", "")
            if feedback:
                if self.logger:
                    self.logger.debug(
                        msg=f"改进答案 (迭代 {iteration}): {feedback}",
                        tag="SelfRefinement"
                    )
                current_answer = self._refine_answer(
                    question,
                    current_answer,
                    feedback
                )
            else:
                break

        # 返回最终结果
        return {
            "answer": current_answer,
            "valid": score >= 0.6,
            "score": score,
            "iterations": max_iterations,
            "validation": validation
        }

    def _refine_answer(self, question: str, answer: str, feedback: str) -> str:
        """改进答案"""
        prompt = self.REFINE_PROMPT.format(
            question=question,
            answer=answer,
            feedback=feedback
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate_content_with_messages(
                messages, model="qwen-plus-latest", temperature=0.3
            )
            return response.strip()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"答案改进失败: {e}")
            return answer


class EnsembleValidator:
    """
    集成验证器

    多个验证器投票
    """

    def __init__(self, llm_clients: List, logger=None):
        self.validators = [AnswerValidator(client, logger) for client in llm_clients]
        self.logger = logger

    def validate(
        self,
        question: str,
        answer: str,
        retrieved_documents: List[str]
    ) -> Dict:
        """
        多验证器验证
        """
        results = []
        for validator in self.validators:
            result = validator.validate(question, answer, retrieved_documents)
            results.append(result)

        # 投票
        valid_count = sum(1 for r in results if r.get("quality_score", 0) >= 0.7)
        avg_score = sum(r.get("quality_score", 0) for r in results) / len(results)

        return {
            "votes": f"{valid_count}/{len(results)}",
            "avg_score": avg_score,
            "details": results
        }


def validate_and_refine(
    question: str,
    answer: str,
    retrieved_documents: List[str],
    llm_client,
    enable_refinement: bool = True,
    logger=None
) -> Dict:
    """
    便捷函数：验证并可能改进答案
    """
    validator = AnswerValidator(llm_client, logger)

    if enable_refinement:
        refiner = SelfRefinementGenerator(llm_client, validator, logger)
        return refiner.generate_with_validation(
            question, answer, retrieved_documents, max_iterations=2
        )
    else:
        validation = validator.validate(question, answer, retrieved_documents)
        return {
            "answer": answer,
            "valid": validation.get("quality_score", 0) >= 0.7,
            "score": validation.get("quality_score", 0),
            "validation": validation
        }
