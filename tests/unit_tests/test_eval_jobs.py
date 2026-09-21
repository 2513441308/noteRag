"""检索评估后台任务注册表与 /admin/eval HTTP 契约的离线测试.

注册表层注入脚本 runner（立即成功/抛错/阻塞），不加载任何真实模型；
HTTP 层用 Starlette TestClient，monkeypatch 默认 runner 后验证
202/409/404 与轮询响应结构。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from agent import eval_jobs
from agent.admin_server import app


@pytest.fixture(autouse=True)
def _isolate_jobs() -> Iterator[None]:
    """每个测试前后清空任务注册表，避免 running 任务串场导致 409。"""
    eval_jobs.reset_registry()
    yield
    eval_jobs.reset_registry()


def _wait_for(
    job_id: str,
    predicate: Any,
    *,
    timeout: float = 5.0,
) -> eval_jobs.EvalJob:
    """轮询任务状态直到 predicate 成立或超时。"""
    deadline = time.monotonic() + timeout
    while True:
        job = eval_jobs.get_job(job_id)
        assert job is not None
        if predicate(job):
            return job
        assert time.monotonic() < deadline, f"等待任务 {job_id} 状态超时"
        time.sleep(0.01)


# ---------------- 注册表层 ----------------


def test_start_job_runs_runner_and_records_summary() -> None:
    summary = {"total": 1, "metrics": {"hit@3": 1.0}}
    job_id = eval_jobs.start_evaluation_job(lambda: dict(summary))

    assert eval_jobs.latest_job_id() == job_id
    job = _wait_for(job_id, lambda j: j.status == "done")
    assert job.summary == summary
    assert job.error is None
    assert job.finished_at is not None
    assert job.started_at <= job.finished_at


def test_start_job_captures_exception_as_error_state() -> None:
    def _boom() -> dict[str, Any]:
        raise RuntimeError("模型炸了")

    job_id = eval_jobs.start_evaluation_job(_boom)
    job = _wait_for(job_id, lambda j: j.status == "error")
    assert job.summary is None
    assert job.error == "RuntimeError: 模型炸了"
    assert job.finished_at is not None


def test_start_job_rejects_while_one_running() -> None:
    release = threading.Event()
    started = threading.Event()

    def _blocking() -> dict[str, Any]:
        started.set()
        release.wait(5.0)
        return {"total": 0}

    first_id = eval_jobs.start_evaluation_job(_blocking)
    assert started.wait(2.0)
    with pytest.raises(eval_jobs.EvalJobBusy):
        eval_jobs.start_evaluation_job(lambda: {})

    release.set()
    _wait_for(first_id, lambda j: j.status == "done")
    # 上一个结束后可以再触发
    second_id = eval_jobs.start_evaluation_job(lambda: {"ok": True})
    _wait_for(second_id, lambda j: j.status == "done")


def test_latest_job_id_none_initially() -> None:
    assert eval_jobs.latest_job_id() is None
    assert eval_jobs.get_job("nope") is None


def test_registry_trims_old_finished_jobs() -> None:
    last_id = ""
    for index in range(eval_jobs._JOB_KEEP + 3):
        last_id = eval_jobs.start_evaluation_job(lambda i=index: {"i": i})
        _wait_for(last_id, lambda j: j.status in {"done", "error"})
    assert len(eval_jobs._job_order) <= eval_jobs._JOB_KEEP
    assert eval_jobs.latest_job_id() == last_id


def test_job_to_dict_is_json_serializable() -> None:
    job_id = eval_jobs.start_evaluation_job(lambda: {"total": 1})
    _wait_for(job_id, lambda j: j.status == "done")
    payload = eval_jobs.get_job(job_id).to_dict()  # type: ignore[union-attr]
    json.dumps(payload, ensure_ascii=False)
    assert payload["job_id"] == job_id
    assert payload["summary"] == {"total": 1}


# ---------------- HTTP 契约层 ----------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient + 默认 runner 置为脚本 runner（不加载真实模型）。"""
    state: dict[str, Any] = {"runner": lambda: {"total": 0}}
    monkeypatch.setattr(eval_jobs, "_default_runner", lambda: state["runner"]())
    with TestClient(app) as client_ctx:
        client_ctx.state = state  # type: ignore[attr-defined]
        yield client_ctx


def _wait_http_done(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(500):
        res = client.get(f"/admin/eval/retrieval/{job_id}")
        assert res.status_code == 200
        body = res.json()
        if body["status"] in {"done", "error"}:
            return body
        time.sleep(0.01)
    raise AssertionError("HTTP 轮询任务结束超时")


def test_http_start_returns_202_and_poll_completes(client: TestClient) -> None:
    client.state["runner"] = lambda: {"total": 20, "metrics": {"hit@3": 0.95}}  # type: ignore[attr-defined]

    res = client.post("/admin/eval/retrieval")
    assert res.status_code == 202
    body = res.json()
    assert body["status"] == "running"
    job_id = body["job_id"]

    result = _wait_http_done(client, job_id)
    assert result["status"] == "done"
    assert result["summary"]["total"] == 20
    assert result["error"] is None
    assert result["started_at"] and result["finished_at"]


def test_http_start_conflicts_while_running(client: TestClient) -> None:
    release = threading.Event()
    started = threading.Event()

    def _blocking() -> dict[str, Any]:
        started.set()
        release.wait(5.0)
        return {"total": 0}

    client.state["runner"] = _blocking  # type: ignore[attr-defined]
    first = client.post("/admin/eval/retrieval")
    assert first.status_code == 202
    assert started.wait(2.0)

    conflict = client.post("/admin/eval/retrieval")
    assert conflict.status_code == 409
    assert "已有" in conflict.json()["detail"]

    release.set()
    _wait_http_done(client, first.json()["job_id"])


def test_http_get_missing_job_404(client: TestClient) -> None:
    res = client.get("/admin/eval/retrieval/does-not-exist")
    assert res.status_code == 404
    assert "不存在" in res.json()["detail"]


def test_http_latest_alias(client: TestClient) -> None:
    # 无任何任务时 latest 也是 404
    assert client.get("/admin/eval/retrieval/latest").status_code == 404

    client.state["runner"] = lambda: {"total": 7}  # type: ignore[attr-defined]
    created = client.post("/admin/eval/retrieval").json()["job_id"]
    result = _wait_http_done(client, "latest")
    assert result["job_id"] == created
    assert result["summary"]["total"] == 7


def test_http_error_runner_surfaces_error_state(client: TestClient) -> None:
    def _boom() -> dict[str, Any]:
        raise ValueError("黄金集不见了")

    client.state["runner"] = _boom  # type: ignore[attr-defined]
    job_id = client.post("/admin/eval/retrieval").json()["job_id"]
    result = _wait_http_done(client, job_id)
    assert result["status"] == "error"
    assert result["summary"] is None
    assert "ValueError" in result["error"]
