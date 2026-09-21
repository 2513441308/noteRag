"""MainState reducer 的离线单元测试."""

from __future__ import annotations

from agent.state.state import merge_docs


def test_merge_docs_first_write_none_left() -> None:
    """通道首次写入时 left 为 None，不应报错，原样返回右侧。"""
    assert merge_docs(None, [{"id": "a"}]) == [{"id": "a"}]


def test_merge_docs_concatenates() -> None:
    """两次写入按顺序拼接（模拟 BM25 + 向量分路写入）。"""
    first = merge_docs(None, [{"id": "a"}])
    second = merge_docs(first, [{"id": "b"}, {"id": "c"}])

    assert second == [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def test_merge_docs_both_none() -> None:
    """两侧都为空时返回空列表而非报错。"""
    assert merge_docs(None, None) == []
