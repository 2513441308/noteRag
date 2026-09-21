"""config 列表型环境变量解析（_get_env_list）的单元测试."""

import pytest

from agent import config
from agent.config import _get_env_list

DEFAULT = ["\n\n", "\n", " ", ""]


def test_missing_key_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEPARATORS", raising=False)
    assert _get_env_list("SEPARATORS", DEFAULT) is DEFAULT


def test_empty_value_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEPARATORS", "   ")
    assert _get_env_list("SEPARATORS", DEFAULT) is DEFAULT


def test_comma_form_unescapes_newlines_and_keeps_empty_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 结尾逗号 -> 空字符串分隔符；字面 \n -> 真实换行
    monkeypatch.setenv("SEPARATORS", r"\n\n,\n, ,")
    assert _get_env_list("SEPARATORS", DEFAULT) == ["\n\n", "\n", " ", ""]


def test_comma_form_does_not_strip_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    # 空格分隔符要放在两段之间；整体 strip 只作用于整个字符串的首尾，
    # 因此 "a, b, " 尾部的 " ," 被整体 strip 后变成结尾逗号 -> 空串段
    monkeypatch.setenv("SEPARATORS", "a, ,b")
    assert _get_env_list("SEPARATORS", DEFAULT) == ["a", " ", "b"]
    monkeypatch.setenv("SEPARATORS", "a, b, ")
    assert _get_env_list("SEPARATORS", DEFAULT) == ["a", " b", ""]


def test_json_array_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEPARATORS", '["\\n\\n", "\\n", " "]')
    assert _get_env_list("SEPARATORS", DEFAULT) == ["\n\n", "\n", " "]


@pytest.mark.parametrize("bad_json", ["[1, 2]", '["ok", 3]', "["])
def test_malformed_json_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, bad_json: str
) -> None:
    monkeypatch.setenv("SEPARATORS", bad_json)
    assert _get_env_list("SEPARATORS", DEFAULT) is DEFAULT


def test_config_separators_default_shape() -> None:
    # 与 splitter 默认约定一致：段落/换行/空格/空串四个层级
    assert config.Config.SEPARATORS == ["\n\n", "\n", " ", ""]
