"""Document/dict 转换工具的离线单元测试."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.utils.doc_utils import dict_to_doc, doc_to_dict


def test_doc_to_dict_shape_and_metadata_copy() -> None:
    """转换后为三字段字典，且 metadata 是副本，不与原文档共享。"""
    doc = Document(id="x1", page_content="正文", metadata={"source": "/a.md"})

    result = doc_to_dict(doc)
    result["metadata"]["source"] = "/changed.md"  # type: ignore[index]

    assert set(result) == {"id", "page_content", "metadata"}
    assert result["id"] == "x1"
    assert result["page_content"] == "正文"
    assert doc.metadata["source"] == "/a.md"  # 原文档未被污染


def test_roundtrip_preserves_fields() -> None:
    """Document -> dict -> Document 往返后字段保持一致。"""
    doc = Document(
        id="x2",
        page_content="含 rerank 元数据",
        metadata={"source": "/b.md", "rerank_rank": 1},
    )

    restored = dict_to_doc(doc_to_dict(doc))

    assert restored.id == "x2"
    assert restored.page_content == "含 rerank 元数据"
    assert restored.metadata == {"source": "/b.md", "rerank_rank": 1}


def test_dict_to_doc_defaults_for_missing_fields() -> None:
    """缺 id/metadata 等字段时安全兜底，不抛异常。"""
    doc = dict_to_doc({"page_content": "只有正文"})

    assert doc.id is None
    assert doc.page_content == "只有正文"
    assert doc.metadata == {}


def test_dict_to_doc_tolerates_bad_metadata_and_id() -> None:
    """metadata 不是字典、id 不是字符串时按空值/None 处理。"""
    doc = dict_to_doc({"id": 123, "page_content": "x", "metadata": "not-a-dict"})

    assert doc.id is None
    assert doc.metadata == {}
