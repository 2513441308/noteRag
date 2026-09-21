"""领域路由节点的提示词模板.

判断用户问题是否需要检索知识库，输出严格单行 JSON。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteResponse(BaseModel):
    """路由节点的结构化输出：问题是否需要检索知识库."""

    relevant: bool = Field(
        description="是否需要检索知识库，需要则为 True，不需要则为 False"
    )


# 输入变量：question
# 期望输出：单行 JSON —— {"relevant": true} 或 {"relevant": false}
ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个问题分类器，负责判断用户的问题是否需要检索知识库才能回答。\n\n"
            # 更换知识库主题时，只需修改这一行领域描述
            "本知识库覆盖的领域：LangGraph 与 LangChain 智能体开发"
            "（StateGraph、节点与边、状态管理、检索增强生成 RAG 等）。\n\n"
            "判定为「需要检索」：\n"
            "- 询问上述领域的概念、用法、代码、报错等事实性内容；\n"
            "- 要求编写或讲解上述领域的代码、示例、配置；\n"
            "- 问题含「它/这个/上面」等指代，但结合上下文仍指向知识库主题。\n\n"
            "判定为「不需要检索」：\n"
            "- 日常问候、闲聊；\n"
            "- 与上述领域无关的通用问题或常识；\n"
            "- 要求扮演其他角色、生成与上述领域无关的内容。\n\n"
            "优先级规则：问题同时包含寒暄与领域内容时（如「你好，请问 StateGraph "
            "怎么用」），或带有生成/写作口吻但主体是领域代码与事实时，"
            "一律判定为「需要检索」。\n",
        ),
        ("human", "用户问题：\n{question}"),
    ]
)
