"""检索词重构节点：结合对话历史把含指代的追问改写为自包含的检索查询.

改写时额外注入一份从向量库 metadata 聚合的知识库章节术语表，引导改写
用词向文档真实标题对齐（「分支判断」→「条件边」）。术语表只拉
metadata、缓存复用，并在知识库写操作后由 admin 接口清缓存刷新。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from agent.config import Config
from agent.llm.llm_client import create_chat_model
from agent.prompts.rewrite import REWRITE_PROMPT
from agent.retrieval.bm25 import load_distinct_headers
from agent.state.state import MainState
from agent.utils.history import render_messages

logger = logging.getLogger(__name__)

# 术语表最多注入多少条标题路径：只给 LLM 词汇参考，过多既费 token 又引噪声
_GLOSSARY_LIMIT = 150
_GLOSSARY_EMPTY = "（无）"


@lru_cache(maxsize=1)
def _get_rewrite_chain() -> Any:
    """懒加载改写链：prompt | 模型 | 字符串解析，只构建一次."""
    llm = create_chat_model()
    return REWRITE_PROMPT | llm | StrOutputParser()


@lru_cache(maxsize=1)
def _get_kb_glossary() -> str:
    """从向量库聚合去重的章节标题路径，渲染为改写提示词的术语表.

    与 BM25 一样仅支持 Chroma（需从底层集合导出 metadata）；集合为空、
    非 Chroma 后端或任何取数异常都返回 ``（无）`` 占位，术语表只是词汇
    参考，绝不能因为它让改写节点失败。

    Returns:
        每行一个 ``- 标题路径`` 的术语文本；无可用标题时返回占位符。
    """
    try:
        # 复用 rag 节点缓存的同一个向量库单例；懒导入避免节点间顶层耦合
        from agent.nodes.rag import _get_vectorstore_from_config

        if Config.VECTORSTORE_PROVIDER.lower() != "chroma":
            return _GLOSSARY_EMPTY
        headers = load_distinct_headers(
            _get_vectorstore_from_config(), limit=_GLOSSARY_LIMIT
        )
    except Exception:
        logger.warning("聚合知识库术语表失败，改写将不带术语参考。", exc_info=True)
        return _GLOSSARY_EMPTY
    if not headers:
        return _GLOSSARY_EMPTY
    return "\n".join(f"- {path}" for path in headers)


def node_search_rebuild(state: MainState) -> dict[str, object]:
    """结合对话历史把最新问题改写为自包含检索查询，写入 ``rewritten_query``.

    Args:
        state: 当前图状态，需包含 ``question``，可选 ``messages`` 历史。

    Returns:
        仅包含 ``rewritten_query`` 字段的状态更新字典；
        首轮无可用历史时不存在指代消解需求，直接回退原始问题并跳过 LLM 调用；
        模型输出为空时同样回退为原始问题，避免空查询进入检索环节。
    """
    question = state["question"]
    history = render_messages(state.get("messages", []), Config.REBUILD_RETRIEVAL_K)
    if history == "（无）":
        # 首轮对话：没有上文就没有「它/这个」等指代可消解，省一次 LLM 调用
        return {"rewritten_query": question}
    rewritten = str(
        _get_rewrite_chain().invoke(
            {
                "history": history,
                "question": question,
                "glossary": _get_kb_glossary(),
            }
        )
    ).strip()
    return {"rewritten_query": rewritten or question}
