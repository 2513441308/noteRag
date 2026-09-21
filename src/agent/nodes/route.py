"""领域路由节点：判断用户问题是否需要检索知识库."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from agent.llm.llm_client import create_chat_model
from agent.prompts.route import ROUTE_PROMPT, RouteResponse
from agent.state.state import MainState


@lru_cache(maxsize=1)
def _get_route_chain() -> Any:
    """懒加载路由链：prompt | 结构化输出模型，只构建一次."""
    llm = create_chat_model()
    return ROUTE_PROMPT | llm.with_structured_output(RouteResponse)


def node_route(state: MainState) -> dict[str, object]:
    """判断问题是否与知识库领域相关，结果写入 ``is_relevant``.

    Args:
        state: 当前图状态，需包含 ``question``。

    Returns:
        仅包含 ``is_relevant`` 字段的状态更新字典。
    """
    response: RouteResponse = _get_route_chain().invoke({"question": state["question"]})
    return {"is_relevant": response.relevant}
