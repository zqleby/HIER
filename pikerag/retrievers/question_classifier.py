"""
问题类型分类器

根据问题自动识别类型，并选择最优检索策略
"""

from typing import Dict, List


class QuestionTypeClassifier:
    """
    问题类型分类器

    类型：
    - FACT: 事实型问题（"什么是变压器？"）
    - COMPARISON: 对比型问题（"变压器和电抗器的区别？"）
    - PROCEDURE: 流程型问题（"如何安装变压器？"）
    - CAUSE: 因果型问题（"短路故障的原因？"）
    - DEFINITION: 定义型问题（"解释一下功率因数"）
    """

    CLASSIFY_PROMPT = """请判断以下电力领域问题的类型。

问题类型定义：
- FACT: 事实型问题，询问某个概念、设备或现象是什么
- COMPARISON: 对比型问题，询问两个事物的区别或联系
- PROCEDURE: 流程型问题，询问操作步骤或流程
- CAUSE: 因果型问题，询问原因或影响
- DEFINITION: 定义型问题，解释某个术语

请只返回类型名称，不要其他内容。

问题：{question}

类型："""

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def classify(self, question: str) -> str:
        """判断问题类型"""
        try:
            prompt = self.CLASSIFY_PROMPT.format(question=question)
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.generate_content_with_messages(
                messages, model="qwen-plus-latest", temperature=0
            )

            type_name = response.strip().upper()
            valid_types = {'FACT', 'COMPARISON', 'PROCEDURE', 'CAUSE', 'DEFINITION'}

            result = type_name if type_name in valid_types else 'FACT'

            if self.logger:
                self.logger.debug(
                    msg=f"问题类型分类: '{question[:50]}...' -> {result}",
                    tag="QuestionClassifier"
                )

            return result

        except Exception as e:
            if self.logger:
                self.logger.warning(f"问题分类失败: {e}")
            return 'FACT'

    def get_retrieval_strategy(self, question_type: str) -> Dict:
        """
        根据问题类型返回最优检索策略

        Returns:
            {
                "preferred_views": ["qa", "summary"],  # 优先使用的视角
                "retrieve_k": 10,                      # 检索数量
                "expand_section": True,                # 是否扩展章节
            }
        """
        strategies = {
            'FACT': {
                "preferred_views": ["qa", "summary", "original"],
                "retrieve_k": 8,
                "expand_section": True,
                "description": "事实型：使用问答对和摘要检索"
            },
            'COMPARISON': {
                "preferred_views": ["original", "summary"],
                "retrieve_k": 12,
                "expand_section": True,
                "description": "对比型：扩大检索范围找对比信息"
            },
            'PROCEDURE': {
                "preferred_views": ["original", "keywords"],
                "retrieve_k": 10,
                "expand_section": True,
                "description": "流程型：使用原文和关键词检索"
            },
            'CAUSE': {
                "preferred_views": ["qa", "summary", "keywords"],
                "retrieve_k": 8,
                "expand_section": False,
                "description": "因果型：聚焦因果关系信息"
            },
            'DEFINITION': {
                "preferred_views": ["summary", "qa"],
                "retrieve_k": 6,
                "expand_section": False,
                "description": "定义型：简洁信息优先"
            }
        }

        return strategies.get(question_type, strategies['FACT'])

    def get_system_prompt(self, question_type: str) -> str:
        """
        根据问题类型返回系统提示词
        """
        prompts = {
            'FACT': "请提供准确、简洁的事实信息。",
            'COMPARISON': "请详细对比两者的区别和联系。",
            'PROCEDURE': "请提供完整的操作步骤和注意事项。",
            'CAUSE': "请分析原因和可能的影响。",
            'DEFINITION': "请给出清晰、准确的定义解释。"
        }
        return prompts.get(question_type, prompts['FACT'])


class HybridQuestionClassifier:
    """
    混合问题分类器

    结合规则和 LLM 进行问题分类
    """

    # 规则关键词
    KEYWORDS = {
        'COMPARISON': ['区别', '不同', '相比', '比较', '与...不同', '和...区别', '差异'],
        'PROCEDURE': ['如何', '怎么', '步骤', '流程', '操作', '安装', '维修', '处理'],
        'CAUSE': ['原因', '为什么', '为何', '由于', '导致', '引起', '影响', '结果'],
        'DEFINITION': ['定义', '解释', '什么是', '意思', '含义', '概念'],
        'FACT': ['什么', '哪些', '哪个', '是谁', '是啥']
    }

    def __init__(self, llm_classifier: QuestionTypeClassifier):
        self.llm_classifier = llm_classifier

    def classify(self, question: str) -> str:
        """先用规则，规则不确定时再用 LLM"""

        # 先用规则判断
        for qtype, keywords in self.KEYWORDS.items():
            for kw in keywords:
                if kw in question:
                    return qtype

        # 规则无法判断，使用 LLM
        return self.llm_classifier.classify(question)

    def classify_with_confidence(self, question: str) -> Dict:
        """
        带置信度的分类
        """
        # 规则匹配
        rule_match = None
        rule_score = 0.0

        for qtype, keywords in self.KEYWORDS.items():
            for kw in keywords:
                if kw in question:
                    rule_match = qtype
                    rule_score = 0.8  # 规则匹配置信度
                    break

        # LLM 分类
        llm_type = self.llm_classifier.classify(question)

        # 综合判断
        if rule_match == llm_type:
            return {
                "type": rule_match,
                "confidence": 0.9,
                "source": "both"
            }
        else:
            # 两者不一致，以 LLM 为准
            return {
                "type": llm_type,
                "confidence": 0.6,
                "source": "llm",
                "rule_hint": rule_match
            }
