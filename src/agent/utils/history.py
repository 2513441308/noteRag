"""对话历史渲染工具：把 LangChain 消息列表渲染为纯文本.

供查询重构与最终生成等节点复用，避免历史拼装逻辑各处复制。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

# LangChain 消息 type -> 历史文本中的角色标签；
# 未知角色统一按「助手」呈现，避免工具/系统消息被误标
_ROLE_LABELS = {
    "human": "用户",
    "ai": "助手",
    "system": "系统",
    "tool": "工具",
}


def render_messages(messages: list[Any], limit: int) -> str:
    """把消息列表渲染为带角色标签的纯文本历史.

    只取最近 ``limit`` 条；非文本内容（如多模态片段、纯工具调用的
    空 AI 消息）跳过。SystemMessage 标注为「系统」、ToolMessage
    标注为「工具」，不再一律冒充助手发言。

    Args:
        messages: LangChain 消息对象（BaseMessage）列表。
        limit: 最多渲染最近多少条消息；非正数时返回空历史占位。

    Returns:
        渲染后的历史文本；没有可用历史时返回 ``（无）``。
    """
    if limit <= 0:
        return "（无）"
    lines: list[str] = []
    for message in messages[-limit:]:
        if not isinstance(message, BaseMessage):
            continue
        content = message.content
        if not isinstance(content, str) or not content.strip():
            continue
        role = _ROLE_LABELS.get(message.type, "助手")
        lines.append(f"{role}：{content.strip()}")
    return "\n".join(lines) or "（无）"
