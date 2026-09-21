"""通用问答节点的提示词模板.

路由判定为「无需检索知识库」后，由通用助手直接回答寒暄与领域外问题。
历史对话以原生消息列表经 ``MessagesPlaceholder`` 直接注入。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 输入变量：
#   summary  - 历史会话摘要（可能为占位符）
#   history  - 已完成轮次的消息列表（Human/AI 交替，可为空）
#   question - 本轮用户原始问题
COMMON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个友好的通用问答助手。用户的问题与知识库覆盖领域"
            "（LangGraph / LangChain 智能体开发）无关，由你直接作答。\n\n"
            "回答规则：\n"
            "1. 自然回应日常寒暄；通用技术问题可用你的通用知识认真回答；\n"
            "2. 不要声称自己检索或查阅过知识库，不要编造来源编号、文档名或链接；\n"
            "3. 不确定或超出能力范围时诚实说明，不要强行作答；\n"
            "4. 使用与用户一致的语言，简洁友好。",
        ),
        ("system", "历史会话摘要（可能为空）：\n{summary}"),
        # 已完成的历史轮次；首轮为空，optional 允许缺省
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "用户问题：\n{question}"),
    ]
)
