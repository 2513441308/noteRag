import pytest

from agent import graph

pytestmark = [pytest.mark.anyio, pytest.mark.langsmith, pytest.mark.integration]


async def test_agent_simple_passthrough() -> None:
    inputs = {"question": "什么是 StateGraph？"}
    config = {"configurable": {"thread_id": "integration-smoke"}}
    res = await graph.ainvoke(inputs, config=config)
    assert res is not None
    assert "answer" in res
