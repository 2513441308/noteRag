"""提示词模板变量渲染的离线单元测试.

验证摘要长度/条数由模板变量注入（不硬编码），以及生成模板的
MessagesPlaceholder 对空/非空历史都能正确渲染。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.prompts.generate import GENERATE_PROMPT
from agent.prompts.summarize import SUMMARY_PROMPT


def test_summary_prompt_limits_are_variables() -> None:
    """300/8 不再硬编码，max_chars/max_items 由调用方注入。"""
    prompt_value = SUMMARY_PROMPT.format_prompt(
        summary="（无）",
        dialogue="用户：你好\n助手：你好",
        max_chars=200,
        max_items=5,
    )

    system_text = str(prompt_value.messages[0].content)
    assert "200 字以内" in system_text
    assert "不超过 5 条" in system_text
    assert "300 字" not in system_text  # 旧硬编码已移除
    assert "{max_chars}" not in system_text
    assert "{max_items}" not in system_text


def test_generate_prompt_empty_history_renders_without_placeholder() -> None:
    """首轮 history 为空列表时模板正常渲染，历史位置不产生消息。"""
    prompt_value = GENERATE_PROMPT.format_prompt(
        summary="（无）",
        context="（无）",
        history=[],
        question="StateGraph 是什么？",
    )

    messages = prompt_value.messages
    # 规则 system + 摘要/文档 system + 本轮 human，无历史消息
    assert len(messages) == 3
    assert messages[-1].content == "StateGraph 是什么？"


def test_generate_prompt_injects_native_history_messages() -> None:
    """非空 history 以原生消息插入到 system 与本轮问题之间。"""
    history = [
        HumanMessage(content="上一轮问题"),
        AIMessage(content="上一轮回答"),
    ]

    prompt_value = GENERATE_PROMPT.format_prompt(
        summary="（无）",
        context="[1]\n文档正文",
        history=history,
        question="追问",
    )

    messages = prompt_value.messages
    assert messages[2] is history[0]
    assert messages[3] is history[1]
    assert messages[-1].content == "追问"
    assert "[1]\n文档正文" in str(messages[1].content)
