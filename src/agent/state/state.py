"""LangGraph 图的状态定义：InputState / MainState / OutputState 三层结构."""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict  # Python<3.12 下 pydantic 生成 schema 要求


# ---------------------- 1. 输入状态 InputState（外部输入，只读） ----------------------
class InputState(TypedDict):
    """仅接收外部输入，节点不修改这里的字段."""

    question: str


# ---------------------- 2. 主状态 MainState（流水线中间全部数据） ----------------------
def merge_docs(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """合并多路检索文档：BM25 + 向量检索结果.

    通道首次被写入时 left 为 None，必须兜底为空列表，否则 ``None + list`` 报错。

    Args:
        left: 已有文档列表，首次写入为 None。
        right: 本次新增的文档列表。

    Returns:
        合并后的文档列表。
    """
    return (left or []) + (right or [])


class MainState(TypedDict):
    """叠加所有中间过程变量，节点读写."""

    question: str  # 原始用户问题
    is_relevant: bool | None  # 领域路由判断：是否LangGraph相关问题
    rewritten_query: str | None  # 查询改写后的检索词
    retrieved_docs: Annotated[
        list[dict[str, Any]], merge_docs
    ]  # 原始召回文档（BM25+向量合并）
    ranked_docs: list[
        dict[str, Any]
    ]  # rerank重排后的文档（dict 形态，见 utils/doc_utils.py）
    sources: list[str]  # 引用来源列表
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史，多轮上下文
    summary: str | None  # 摘要，可选
    answer: str  # 最终回答


# ---------------------- 3. 输出状态 OutputState（最终返回给外部） ----------------------
class OutputState(TypedDict):
    """最终对外输出，只保留需要返回的字段，隐藏内部中间变量."""

    answer: str  # 最终回答
    sources: list[str]  # 引用来源列表
