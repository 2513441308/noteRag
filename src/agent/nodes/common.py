"""通用节点：回答无需检索知识库的问题."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm.llm_client import create_chat_model
from agent.prompts.common import COMMON_PROMPT
from agent.state.state import MainState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_common_chain() -> Any:
    """懒加载通用问答链：prompt | 模型 | 字符串解析，只构建一次."""
    llm = create_chat_model()
    return COMMON_PROMPT | llm | StrOutputParser()


def node_common(state: MainState) -> dict[str, object]:
    """回答与知识库无关的通用问题，写入 ``answer`` 并追加完整对话轮次.

    历史以原生消息列表经 MessagesPlaceholder 全量注入，摘要缺失时给
    占位符；历史规模统一由 summarize 节点压缩控制，本节点不截断。
    生成完成后把本轮的 ``HumanMessage`` 与 ``AIMessage`` 成对追加到
    ``messages``，使通用问答分支与 RAG 分支的多轮历史行为保持一致。

    Args:
        state: 当前图状态，需包含 ``question``。

    Returns:
        包含 ``answer``、空 ``sources``（通用回答无知识库引用）与
        ``messages``（本轮 Human + AI 两条消息）的更新。
    """
    question = state["question"]
    history = list(state.get("messages", []))
    response = str(
        _get_common_chain().invoke(
            {
                "summary": state.get("summary") or "（无）",
                "history": history,
                "question": question,
            }
        )
    ).strip()
    logger.info("通用问答分支生成回答 %d 字。", len(response))
    return {
        "answer": response,
        "sources": [],
        "messages": [HumanMessage(content=question), AIMessage(content=response)],
    }
