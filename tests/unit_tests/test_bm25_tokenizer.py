"""normalized_tokenize 归一化分词的单元测试（纯函数，无模型无网络）."""

import pytest

from agent.retrieval.bm25 import chinese_tokenize, normalized_tokenize


def test_camel_case_expands_to_whole_and_parts() -> None:
    assert normalized_tokenize("StateGraph") == ["stategraph", "state", "graph"]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("stategraph", ["stategraph"]),
        ("STATEGRAPH", ["stategraph"]),
        # 交替大小写会触发驼峰边界，但整体小写形态仍保留（精确匹配不丢）
        ("StAtEgRaPh", ["stategraph", "st", "at", "eg", "ra", "ph"]),
    ],
)
def test_case_insensitive_whole_form(word: str, expected: list[str]) -> None:
    assert normalized_tokenize(word) == expected


def test_consecutive_capitals_boundary() -> None:
    # HTTPServer -> HTTP|Server，而不是把 HTTPServer 当一个不可拆整体
    assert normalized_tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_chinese_tokens_preserved() -> None:
    assert normalized_tokenize("条件边怎么写") == ["条件", "边", "怎么", "写"]


def test_mixed_chinese_english() -> None:
    tokens = normalized_tokenize("用 StateGraph 构建条件边")
    assert tokens == ["用", "stategraph", "state", "graph", "构建", "条件", "边"]


def test_snake_and_hyphen_normalize_to_same_terms() -> None:
    snake = normalized_tokenize("add_conditional_edges")
    hyphen = normalized_tokenize("add-conditional-edges")
    assert snake == hyphen == ["add", "conditional", "edges"]


def test_punctuation_tokens_dropped() -> None:
    tokens = normalized_tokenize("graph.compile() = result")
    assert "." not in tokens and "(" not in tokens and ")" not in tokens
    assert "=" not in tokens
    assert "compile" in tokens and "result" in tokens


def test_no_whitespace_tokens() -> None:
    assert all(token.strip() for token in normalized_tokenize("a    b\n\nc\t"))


def test_dedup_only_within_one_raw_token() -> None:
    # 同一个词重复出现两次是合法词频，不应去重；整体/子词的形态重复才去重
    tokens = normalized_tokenize("StateGraph StateGraph")
    assert tokens.count("stategraph") == 2
    assert tokens.count("state") == 2 and tokens.count("graph") == 2


def test_punctuation_glued_keeps_whole_form() -> None:
    # 中文句号粘在驼峰词尾部时，精确整体形态仍要保留
    tokens = normalized_tokenize("StateGraph。")
    assert "stategraph" in tokens and "state" in tokens and "graph" in tokens


def test_legacy_chinese_tokenizer_kept_for_reference() -> None:
    # 旧分词器行为对照：大小写敏感、标点保留（证明它确实已不适合英文语料）
    tokens = chinese_tokenize("StateGraph")
    assert "StateGraph" in tokens and "stategraph" not in tokens
