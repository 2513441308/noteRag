"""对话历史压缩节点：超阈值时把旧的一半消息增量摘要后删除.

挂载在回答节点之后、END 之前，由条件边根据 :func:`should_summarize`
决定是否执行。触发条件（满足任一）：

1. 消息条数超过 ``Config.SUMMARY_TRIGGER_MESSAGES``；
2. 历史估算 token 超过 ``Config.SUMMARY_TRIGGER_TOKENS``。

触发后按**对话轮**为最小单位切分：一轮从一个 HumanMessage 开始、
到下一个 HumanMessage 之前结束，普通问答轮即 [Human, AI]，工具
调用轮则是 [Human, AI(tool_calls), ToolMessage..., AI]，切点始终
对齐整轮边界，绝不会把一轮对话（尤其工具调用组）拆成两半。
较旧的 ``Config.SUMMARY_COMPRESS_RATIO``（默认一半）轮次经
``SUMMARY_PROMPT`` 与已有摘要增量合并写回 ``summary`` 后，用
``RemoveMessage`` 从 messages 通道删除，较新的轮次原样保留。
SystemMessage 属于稳定上下文，不计入轮次、永不压缩删除
（add_messages 是追加语义，删除只能走 RemoveMessage；节点返回的
无 id 消息入通道时由 reducer 自动补 id）。

安全保证：摘要 LLM 调用失败或返回空白时**不删除任何消息**，避免
「摘要没生成、历史却被删」的不可逆数据丢失。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import RemoveMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.config import Config
from agent.llm.llm_client import create_chat_model
from agent.prompts.summarize import SUMMARY_PROMPT
from agent.state.state import MainState
from agent.utils.history import render_messages
from agent.utils.token_utils import estimate_messages_tokens

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_summary_chain() -> Any:
    """懒加载摘要链：prompt | 模型 | 字符串解析，只构建一次."""
    llm = create_chat_model()
    return SUMMARY_PROMPT | llm | StrOutputParser()


def should_summarize(state: MainState) -> bool:
    """判断本轮回答后是否需要压缩历史（条数或 token 任一超阈值）.

    Args:
        state: 当前图状态。

    Returns:
        需要压缩返回 True，否则 False。
    """
    messages = list(state.get("messages", []))
    if len(messages) > Config.SUMMARY_TRIGGER_MESSAGES:
        return True
    return estimate_messages_tokens(messages) > Config.SUMMARY_TRIGGER_TOKENS


def _split_stale_and_kept(messages: list[Any]) -> tuple[list[Any], list[Any]]:
    """按对话轮切分：旧的 ratio 轮进摘要，其余原样保留.

    一轮 = 从一个 HumanMessage 到下一个 HumanMessage 之前的全部消息：
    普通问答轮为 [Human, AI]；工具调用轮天然是
    [Human, AI(tool_calls), ToolMessage..., AI]，作为原子单元整体
    压缩或保留，不会把工具调用组劈成两半（否则残留的 tool_call_id
    找不到对应 ToolMessage，会导致后续 LLM 请求非法）。

    SystemMessage 属于稳定上下文（如持久系统指令），不计入轮次、
    永不进 stale，也就永远不会被 RemoveMessage 删除；第一个
    HumanMessage 之前的悬挂消息（状态续写等异常形态）同样不压缩。

    stale 轮数按 ``SUMMARY_COMPRESS_RATIO`` 向下取整，保证新旧两侧
    各至少一轮；比例非法（<=0 或 >=1）时按 0.5 兜底。

    Args:
        messages: 当前全部历史消息。

    Returns:
        (stale, kept) 二元组，stale 为待摘要删除的旧轮消息；
        没有用户消息、不足两轮或 stale 不含可压缩消息时返回
        ([], messages)。
    """
    first_human = next(
        (
            i
            for i, message in enumerate(messages)
            if getattr(message, "type", None) == "human"
        ),
        None,
    )
    if first_human is None:
        return [], list(messages)

    # 首个用户消息之前的前导消息（SystemMessage / 悬挂消息）一律保留
    head = list(messages[:first_human])
    turns: list[list[Any]] = []
    for message in messages[first_human:]:
        if getattr(message, "type", None) == "human" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)

    if len(turns) < 2:
        return [], list(messages)

    ratio = Config.SUMMARY_COMPRESS_RATIO
    if not 0 < ratio < 1:
        ratio = 0.5
    stale_turns = max(1, min(int(len(turns) * ratio), len(turns) - 1))

    stale_raw = [message for turn in turns[:stale_turns] for message in turn]
    # 防御：SystemMessage 即使混在旧轮区间也不压缩（不删除、不进摘要）
    stale = [message for message in stale_raw if not isinstance(message, SystemMessage)]
    if not stale:
        return [], list(messages)
    rescued_system = [
        message for message in stale_raw if isinstance(message, SystemMessage)
    ]
    kept = (
        head
        + rescued_system
        + [message for turn in turns[stale_turns:] for message in turn]
    )
    return stale, kept


def node_summarize(state: MainState) -> dict[str, object]:
    """把旧消息增量合并进摘要，并从历史中删除这些旧消息.

    Args:
        state: 当前图状态，需包含 ``messages``。

    Returns:
        成功时返回 ``summary``（新摘要）与 ``messages``（RemoveMessage
        列表）；无可压缩内容、摘要失败或返回空白时返回空字典，历史原样
        保留，下一轮再尝试。
    """
    messages = list(state.get("messages", []))
    stale, _kept = _split_stale_and_kept(messages)
    if not stale:
        logger.debug("消息过少，跳过压缩。")
        return {}

    dialogue = render_messages(stale, limit=len(stale))
    try:
        new_summary = str(
            _get_summary_chain().invoke(
                {
                    "summary": state.get("summary") or "（无）",
                    "dialogue": dialogue,
                    "max_chars": Config.SUMMARY_MAX_CHARS,
                    "max_items": Config.SUMMARY_MAX_ITEMS,
                }
            )
        ).strip()
    except Exception:
        # 先成功拿到摘要才能删历史：任何异常都放弃本轮压缩
        logger.exception("摘要生成失败，本轮跳过压缩，历史保持不变。")
        return {}

    if not new_summary:
        logger.warning("摘要结果为空，本轮跳过压缩，历史保持不变。")
        return {}

    removals = [
        RemoveMessage(id=message.id)
        for message in stale
        if getattr(message, "id", None) is not None
    ]
    if len(removals) != len(stale):
        # 理论上经 add_messages 通道的消息都有 id；缺 id 意味着无法精确
        # 删除，宁可不压缩也不能残留半删状态
        logger.error("存在无 id 的历史消息，无法安全删除，本轮跳过压缩。")
        return {}

    logger.info("压缩 %d 条旧消息为 %d 字摘要。", len(stale), len(new_summary))
    return {"summary": new_summary, "messages": removals}
