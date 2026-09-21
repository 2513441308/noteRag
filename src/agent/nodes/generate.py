"""最终回答节点（RAG 生成）：依据精排文档生成带来源编号的中文回答.

文档按**来源级编号**组织 context：同一来源文件的多个 chunk 归入同一个
``[n]`` 分组，``n`` 即 rerank 节点产出的去重 ``sources`` 列表序号，
模型在句尾标注的 ``[n]`` 与 ``state["sources"]`` 天然一一对齐。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm.llm_client import create_chat_model
from agent.prompts.generate import GENERATE_PROMPT
from agent.state.state import MainState

logger = logging.getLogger(__name__)

_NO_SOURCE_HEADER = "[无来源]"


@lru_cache(maxsize=1)
def _get_generate_chain() -> Any:
    """懒加载生成链：prompt | 模型 | 字符串解析，只构建一次."""
    llm = create_chat_model()
    return GENERATE_PROMPT | llm | StrOutputParser()


def _source_order_key(doc: dict[str, Any]) -> int | None:
    """提取 chunk_index 用于组内恢复文档原序；缺失/非法返回 None."""
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        return None
    index = metadata.get("chunk_index")
    return index if isinstance(index, int) else None


def _order_contents(docs: list[dict[str, Any]]) -> list[str]:
    """同来源片段按 chunk_index 升序排列，无序号片段保持原序追加在后.

    重排输出按分数排序，会把同一文档前后相邻的论述打散；邻域扩展补回的
    兄弟块与命中块在这里统一恢复成文档原始叙事顺序。Python sort 稳定，
    序号相同（理论上不应出现）时保留传入次序。
    """
    indexed = [
        (int(doc["metadata"]["chunk_index"]), str(doc["page_content"]))
        for doc in docs
        if _source_order_key(doc) is not None
    ]
    indexed.sort(key=lambda item: item[0])
    ordered = [content for _index, content in indexed]
    ordered.extend(
        str(doc["page_content"]) for doc in docs if _source_order_key(doc) is None
    )
    return ordered


def _build_context(docs: list[dict[str, Any]]) -> str:
    """把精排文档按来源分组渲染为 ``[n]`` 编号的上下文文本.

    编号按来源在文档序列中的首次出现顺序分配（与 ``state["sources"]``
    的去重保序结果一致）；同一来源的多个 chunk 归入同一编号、以分隔线
    隔开，组内按 ``chunk_index`` 恢复文档原序。缺失 ``source`` 元数据的
    文档（生产链路上不应当出现）统一放入「无来源」分组，避免静默丢弃
    召回内容。

    Args:
        docs: state 中字典形态的 ``ranked_docs``。

    Returns:
        编号分组后的上下文文本；无有效内容时返回 ``（无）``。
    """
    blocks: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    sourceless: list[str] = []
    for doc in docs:
        content = str(doc.get("page_content", "")).strip()
        if not content:
            continue
        metadata = doc.get("metadata")
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if source:
            source = str(source)
            if source not in blocks:
                blocks[source] = []
                order.append(source)
            blocks[source].append(doc)
        else:
            sourceless.append(content)

    sections: list[str] = []
    for index, source in enumerate(order, start=1):
        contents = _order_contents(blocks[source])
        sections.append(f"[{index}]\n" + "\n---\n".join(contents))
    if sourceless:
        sections.append(f"{_NO_SOURCE_HEADER}\n" + "\n---\n".join(sourceless))
    if not sections:
        return "（无）"
    # 给出确定性的合法编号范围，防止模型引用不存在的编号（如只有 [1] 却标 [2]）
    if order:
        sections.append(
            f"（本次共 {len(order)} 个可用来源，合法引用编号仅为 [1]~[{len(order)}]，"
            "严禁使用其他编号）"
        )
    return "\n\n".join(sections)


def node_generate(state: MainState) -> dict[str, object]:
    """依据 ``ranked_docs`` 生成最终回答，写入 ``answer`` 并追加完整对话轮次.

    使用**原始** ``question``（而非改写后的检索词）作为用户问题，保证
    回答表述对齐用户原话；历史以原生消息列表经 MessagesPlaceholder **全量**
    注入，摘要缺失时给占位符。历史规模由 summarize 节点单一负责
    （超阈值即压缩为摘要 + 删除旧消息），此处不再二次截断，避免出现
    「旧消息已不在窗口、却尚未压缩进摘要」的信息空洞。

    生成完成后把本轮的 ``HumanMessage`` 与 ``AIMessage`` **成对**追加到
    ``messages``：进入节点的 messages 只含已完成的历史轮次，本轮用户问题
    此刻才入历史，因此查询重构等上游节点看到的历史天然不含在途问题。

    Args:
        state: 当前图状态，需包含 ``question`` 与 ``ranked_docs``。

    Returns:
        包含 ``answer`` 与 ``messages``（本轮 Human + AI 两条消息）的更新。
    """
    question = state["question"]
    history = list(state.get("messages", []))
    payload = {
        "summary": state.get("summary") or "（无）",
        "history": history,
        "context": _build_context(list(state.get("ranked_docs", []))),
        "question": question,
    }
    answer = str(_get_generate_chain().invoke(payload)).strip()
    source_count = len(state.get("sources", []))
    logger.info("生成回答 %d 字，依据来源 %d 个。", len(answer), source_count)
    return {
        "answer": answer,
        "messages": [HumanMessage(content=question), AIMessage(content=answer)],
    }
