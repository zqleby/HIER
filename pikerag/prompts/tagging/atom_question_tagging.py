# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate


DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的人工智能助手，擅长内容理解和提问。"


atom_question_tagging_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
你的任务是尽可能多地提取与给定内容相关且可以回答的问题。请尽量保持多样性，避免提取重复或相似的问题。确保你的问题包含必要的实体名称，并避免使用它、他、她、他们、公司、该人等代词。
# Output Format
将你的答案逐行输出，每个问题另起一行，不使用项目符号或数字。
# Content
{content}

# Output:
""".strip()),
    ],
    input_variables=["content"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AtomQuestionParser(BaseContentParser):
    def encode(self, content: str, **kwargs) -> Tuple[str, dict]:
        title = kwargs.get("title", None)
        if title is not None:
            content = f"Title: {title}. Content: {content}"
        return content, {}

    def decode(self, content: str, **kwargs) -> List[str]:
        questions = content.split("\n")
        questions = [question.strip() for question in questions if len(question.strip()) > 0]
        return questions


atom_question_tagging_protocol = CommunicationProtocol(
    template=atom_question_tagging_template,
    parser=AtomQuestionParser(),
)
