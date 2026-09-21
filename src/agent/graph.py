"""RAG Agent 的 LangGraph 工作流编排.

链路：

    START -> route
      ├─ is_relevant=true  -> searchrebuild -> rag -> rerank -> generate ─┐
      └─ is_relevant=false -> common ─────────────────────────────────────┤
                                                                          ↓
                                       条件边 should_summarize?
                                          ├─ true  -> summarize -> END
                                          └─ false -> END

输出经 OutputState 收窄，只对外暴露 ``answer`` 与 ``sources``。
``build_graph`` 抽成工厂便于测试期重建；模块级 ``graph`` 为默认实例。
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph

from agent.checkpointer.checkpointer import get_checkpointer
from agent.nodes.common import node_common
from agent.nodes.generate import node_generate
from agent.nodes.rag import node_rag
from agent.nodes.rerank import node_rerank
from agent.nodes.route import node_route
from agent.nodes.searchrebuild import node_search_rebuild
from agent.nodes.summarize import node_summarize, should_summarize
from agent.state.state import InputState, MainState, OutputState

# 条件边返回的节点名常量
_RAG = "searchrebuild"
_COMMON = "common"
_SUMMARIZE = "summarize"


def _after_route(state: MainState) -> str:
    """路由分叉：领域相关走 RAG 检索链路，否则走通用问答."""
    return _RAG if state.get("is_relevant") else _COMMON


def _after_answer(state: MainState) -> str:
    """回答后分叉：历史超阈值则先压缩，否则直接结束."""
    return _SUMMARIZE if should_summarize(state) else END


def build_graph() -> CompiledStateGraph[MainState, Any, InputState, OutputState]:
    """组装并编译 RAG Agent 工作流.

    Returns:
        编译后的可执行图（CompiledStateGraph / Pregel 实例），
        泛型参数依次为状态类型、上下文、输入状态、输出状态。
    """
    builder = StateGraph(MainState, input_schema=InputState, output_schema=OutputState)

    builder.add_node("route", node_route)
    builder.add_node(_COMMON, node_common)
    builder.add_node(_RAG, node_search_rebuild)
    builder.add_node("rag", node_rag)
    builder.add_node("rerank", node_rerank)
    builder.add_node("generate", node_generate)
    builder.add_node(_SUMMARIZE, node_summarize)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _after_route,
        {_RAG: _RAG, _COMMON: _COMMON},
    )

    # RAG 主链路：改写 -> 双路召回 -> 精排 -> 生成
    builder.add_edge(_RAG, "rag")
    builder.add_edge("rag", "rerank")
    builder.add_edge("rerank", "generate")

    # 两条回答支路汇合后按历史规模决定是否压缩
    builder.add_conditional_edges(
        "generate", _after_answer, {_SUMMARIZE: _SUMMARIZE, END: END}
    )
    builder.add_conditional_edges(
        _COMMON, _after_answer, {_SUMMARIZE: _SUMMARIZE, END: END}
    )
    builder.add_edge(_SUMMARIZE, END)

    # langgraph dev/平台托管时（LANGSMITH_LANGGRAPH_API_VARIANT 已设置），
    # 持久化由平台自动注入 checkpointer，自带会被 local_dev 直接拒绝；
    # 独立运行/pytest 时才使用配置中的 checkpointer
    checkpointer = (
        None if os.getenv("LANGSMITH_LANGGRAPH_API_VARIANT") else get_checkpointer()
    )
    return builder.compile(name="RAG Agent", checkpointer=checkpointer)


graph = build_graph()
