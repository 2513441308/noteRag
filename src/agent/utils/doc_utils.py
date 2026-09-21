"""文档形态转换工具：LangChain ``Document`` 与 state 字典互转.

state 的 ``retrieved_docs`` / ``ranked_docs`` 声明为 ``list[dict]``，
而检索与重排组件吃的是 ``Document``，统一在本模块做两种形态的转换，
避免各处各写一份导致字段契约漂移。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def doc_to_dict(doc: Document) -> dict[str, Any]:
    """把 ``Document`` 转为可存入 state 的普通字典.

    Args:
        doc: 检索/重排产出的文档对象。

    Returns:
        包含 ``id`` / ``page_content`` / ``metadata`` 三个字段的字典，
        metadata 复制一份以免外部修改污染原文档。
    """
    return {
        "id": doc.id,
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata) if doc.metadata else {},
    }


def dict_to_doc(item: dict[str, Any]) -> Document:
    """把 state 中的文档字典还原为 ``Document``.

    Args:
        item: 含 ``id`` / ``page_content`` / ``metadata`` 的字典，
            缺字段时按空值兜底（兼容旧 state 数据）。

    Returns:
        LangChain ``Document`` 对象。
    """
    raw_id = item.get("id")
    raw_metadata = item.get("metadata")
    return Document(
        id=raw_id if isinstance(raw_id, str) else None,
        page_content=str(item.get("page_content", "")),
        metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
    )
