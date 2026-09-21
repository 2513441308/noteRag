"""消息 token 估算工具：基于 tiktoken 粗估对话历史占用.

仅用于判断何时触发历史压缩，不需要与模型供应商的计费口径精确一致；
数值稳定、离线可算、对同一输入单调即可。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage

# 每条消息的固定结构开销（角色标记等，OpenAI 口径约 3~4 token/条）
_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _get_encoder() -> Any:
    """懒加载 cl100k_base 编码器，只初始化一次（编码表已随包内置）."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str) -> int:
    """估算单段文本的 token 数.

    Args:
        text: 待估算文本。

    Returns:
        非负 token 数；空文本为 0。
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def estimate_messages_tokens(messages: list[Any]) -> int:
    """估算消息列表的总 token 数（含每条消息的固定结构开销）.

    Args:
        messages: LangChain 消息对象列表；非字符串 content（如多模态片段）
            按其字符串形式降级估算，非 BaseMessage 对象跳过。

    Returns:
        估算总 token 数。
    """
    total = 0
    for message in messages:
        if not isinstance(message, BaseMessage):
            continue
        content = message.content
        total += _PER_MESSAGE_OVERHEAD
        total += count_text_tokens(
            content if isinstance(content, str) else str(content)
        )
    return total
