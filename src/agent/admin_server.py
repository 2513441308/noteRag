"""调试用管理路由：网页查看/热更新 Agent 配置.

通过 ``langgraph.json`` 的 ``http.app`` 挂载到 langgraph dev 同一端口，
路由前缀 ``/admin``。仅适用于本地开发调试（dev 服务器绑 127.0.0.1）。

- GET  /admin/config 读取当前有效配置（密钥脱敏）及表单 schema
- POST /admin/config 保存：运行时覆盖 + 写回 .env + 重置组件缓存（即时生效）
- POST /admin/reset-caches 单独重置组件缓存

知识库文件级管理（源文档目录 ↔ Chroma 双写，写完热刷检索索引）：
- GET    /admin/kb/documents        列出文档与索引状态
- GET    /admin/kb/document?path=   读取文本文件正文
- POST   /admin/kb/document         新建文本文件 {path, content}
- PUT    /admin/kb/document         更新文本文件 {path, content}
- POST   /admin/kb/upload           上传/替换 {path, content_base64, overwrite}
- DELETE /admin/kb/document?path=   删除文件及其切片
- POST   /admin/kb/reindex?path=    单文件重新索引
- POST   /admin/kb/prune            清理源文件已删的游离切片

检索层离线评估（本地模型，跑数分钟，后台任务 + 轮询）：
- POST /admin/eval/retrieval        触发一次黄金集评估（运行中再触发返回 409）
- GET  /admin/eval/retrieval/{id}   查询任务状态与报告（id=latest 取最近一次）
"""

from __future__ import annotations

import importlib
import logging
from functools import partial
from typing import Any

from anyio import to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent import eval_jobs
from agent.config import (
    ConfigError,
    apply_runtime_overrides,
    get_hot_config,
)
from agent.knowledge_admin import KbAdmin, KbError

logger = logging.getLogger(__name__)


# 含 @lru_cache 组件的模块：配置变更后需清空，强制下次调用按新 Config 重建。
# 刻意不含 checkpointer（清缓存会丢失内存会话）与 token_utils（分词器与配置无关）。
_CACHE_MODULES = (
    "agent.llm.llm_client",
    "agent.nodes.route",
    "agent.nodes.common",
    "agent.nodes.generate",
    "agent.nodes.searchrebuild",
    "agent.nodes.summarize",
    "agent.nodes.rag",
    "agent.nodes.rerank",
)


def reset_runtime_caches() -> int:
    """清空所有配置相关的实例/链路缓存，返回清理的缓存项数量."""
    cleared = 0

    for module_name in _CACHE_MODULES:
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            clear = getattr(attr, "cache_clear", None)
            if callable(clear):
                clear()
                cleared += 1

    # 嵌入模型实例 dict 缓存
    from agent.embedding.embedding_factory import _INSTANCE_CACHE as embed_cache

    cleared += len(embed_cache)
    embed_cache.clear()

    # 重排模型实例 dict 缓存
    from agent.retrieval.reranker import _reranker_cache

    cleared += len(_reranker_cache)
    _reranker_cache.clear()

    logger.info("已重置 %d 个配置相关缓存", cleared)
    return cleared


async def read_config(_request: Request) -> JSONResponse:
    """当前有效配置 + 表单 schema（密钥仅返回脱敏标记）."""
    return JSONResponse(get_hot_config())


def _normalize_value(value: Any) -> str:
    """JSON 值统一转成配置层接受的字符串表单形式."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


async def write_config(request: Request) -> JSONResponse:
    """校验 → 运行时覆盖 → 写回 .env → 重置缓存，全部成功才返回."""
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "请求体不是合法 JSON"}, status_code=400)

    raw = body.get("values") if isinstance(body, dict) else None
    if not isinstance(raw, dict) or not raw:
        return JSONResponse({"detail": "values 必须是非空对象"}, status_code=400)

    try:
        values = {str(k): _normalize_value(v) for k, v in raw.items()}
        applied = apply_runtime_overrides(values)
    except ConfigError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    cleared = reset_runtime_caches()
    return JSONResponse({"ok": True, "applied": applied, "caches_cleared": cleared})


async def reset_caches(_request: Request) -> JSONResponse:
    """单独触发缓存重置（外部手改 .env 后可手动点）."""
    return JSONResponse({"ok": True, "caches_cleared": reset_runtime_caches()})


# ---------------------------------------------------------------------------
# 知识库管理
# ---------------------------------------------------------------------------

# 懒加载单例：首次知识库请求时才按 Config 构造（嵌入模型在实际入库时才加载）
_kb_singleton: KbAdmin | None = None


def _get_kb() -> KbAdmin:
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = KbAdmin.from_config()
    return _kb_singleton


def _refresh_retrieval_caches() -> None:
    """知识库写操作后热刷检索缓存.

    BM25 是从集合导出的内存快照，必须清空让其下次问答重建；向量检索器每次
    invoke 都实时查库（且与管理端共用同一 Chroma 实例），一并清空代价极小。
    查询改写的章节术语表同样派生自集合 metadata，一并刷新。
    刻意不触碰 LLM 链与嵌入缓存——知识库变更与它们无关。
    """
    from agent.nodes import rag, searchrebuild

    rag._get_bm25_retriever.cache_clear()
    rag._get_vector_retriever.cache_clear()
    searchrebuild._get_kb_glossary.cache_clear()


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise KbError("请求体不是合法 JSON。", status=400) from exc
    if not isinstance(body, dict):
        raise KbError("请求体必须是 JSON 对象。", status=400)
    return body


def _kb_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, KbError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)
    logger.exception("知识库管理接口异常")
    return JSONResponse(
        {"detail": f"服务器内部错误：{type(exc).__name__}"}, status_code=500
    )


# 说明：知识库操作包含 Chroma/SQLite、磁盘与本地嵌入等重阻塞同步调用，
# 绝不能在事件循环线程直接执行，否则触发 BlockingError。端点仅负责 async 的
# 请求体解析，真正的业务一律经 anyio.to_thread 丢到线程池运行。


async def kb_list(_request: Request) -> JSONResponse:
    """列出知识库全部文档及其索引状态、游离切片清单."""
    try:
        result = await to_thread.run_sync(_get_kb().list_documents)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - 统一映射为 JSON 错误
        return _kb_error(exc)


async def kb_read(request: Request) -> JSONResponse:
    """读取文本类文件正文（二进制返回 415）."""
    try:
        path = request.query_params["path"]
    except KeyError:
        return JSONResponse({"detail": "缺少 path 查询参数。"}, status_code=400)
    try:
        result = await to_thread.run_sync(partial(_get_kb().read_text, path))
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_create(request: Request) -> JSONResponse:
    """在线新建文本文件并入库（同名返回 409）."""
    body = await _json_body(request)
    try:
        result = await to_thread.run_sync(
            partial(
                _get_kb().create_text,
                str(body.get("path", "")),
                str(body.get("content", "")),
            )
        )
        _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_update(request: Request) -> JSONResponse:
    """覆写文本文件并按 source 替换其全部切片."""
    body = await _json_body(request)
    try:
        result = await to_thread.run_sync(
            partial(
                _get_kb().update_text,
                str(body.get("path", "")),
                str(body.get("content", "")),
            )
        )
        _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_upload(request: Request) -> JSONResponse:
    """上传/替换任意受支持文件（base64 正文），随后热刷检索索引."""
    body = await _json_body(request)
    try:
        result = await to_thread.run_sync(
            partial(
                _get_kb().upload,
                str(body.get("path", "")),
                str(body.get("content_base64", "")),
                overwrite=bool(body.get("overwrite", False)),
            )
        )
        _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_delete(request: Request) -> JSONResponse:
    """删除源文件并清除其全部切片."""
    try:
        path = request.query_params["path"]
    except KeyError:
        return JSONResponse({"detail": "缺少 path 查询参数。"}, status_code=400)
    try:
        result = await to_thread.run_sync(partial(_get_kb().delete, path))
        _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_reindex(request: Request) -> JSONResponse:
    """对单个已存在文件重新切分嵌入."""
    try:
        path = request.query_params["path"]
    except KeyError:
        return JSONResponse({"detail": "缺少 path 查询参数。"}, status_code=400)
    try:
        result = await to_thread.run_sync(partial(_get_kb().reindex, path))
        _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


async def kb_prune(_request: Request) -> JSONResponse:
    """清理源文件已不存在的游离切片."""
    try:
        result = await to_thread.run_sync(_get_kb().prune_orphans)
        if result["count"]:
            _refresh_retrieval_caches()
        return JSONResponse({"ok": True, **result})
    except Exception as exc:  # noqa: BLE001
        return _kb_error(exc)


# ---------------------------------------------------------------------------
# 检索层离线评估（后台任务 + 轮询；评估本身在 daemon 线程里跑 CPU）
# ---------------------------------------------------------------------------


async def eval_retrieval_start(_request: Request) -> JSONResponse:
    """触发一次黄金集检索评估，立即 202 返回 job_id.

    评估固定使用本地模型、耗时数分钟，实际执行发生在
    :mod:`agent.eval_jobs` 的 daemon 线程中；已有任务运行时返回 409。
    """
    try:
        job_id = eval_jobs.start_evaluation_job()
    except eval_jobs.EvalJobBusy:
        return JSONResponse(
            {"detail": "已有检索评估任务在运行，请等待本次完成后再触发。"},
            status_code=409,
        )
    return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)


async def eval_retrieval_status(request: Request) -> JSONResponse:
    """查询评估任务状态与报告；路径 id 为 ``latest`` 时取最近一次."""
    job_id = request.path_params["job_id"]
    if job_id == "latest":
        job_id = eval_jobs.latest_job_id()
        if job_id is None:
            return JSONResponse({"detail": "还没有任何评估任务。"}, status_code=404)
    job = eval_jobs.get_job(job_id)
    if job is None:
        return JSONResponse({"detail": f"评估任务不存在：{job_id}"}, status_code=404)
    return JSONResponse(job.to_dict())


app = Starlette(
    routes=[
        Route("/admin/config", read_config, methods=["GET"]),
        Route("/admin/config", write_config, methods=["POST"]),
        Route("/admin/reset-caches", reset_caches, methods=["POST"]),
        Route("/admin/kb/documents", kb_list, methods=["GET"]),
        Route("/admin/kb/document", kb_read, methods=["GET"]),
        Route("/admin/kb/document", kb_create, methods=["POST"]),
        Route("/admin/kb/document", kb_update, methods=["PUT"]),
        Route("/admin/kb/document", kb_delete, methods=["DELETE"]),
        Route("/admin/kb/upload", kb_upload, methods=["POST"]),
        Route("/admin/kb/reindex", kb_reindex, methods=["POST"]),
        Route("/admin/kb/prune", kb_prune, methods=["POST"]),
        Route("/admin/eval/retrieval", eval_retrieval_start, methods=["POST"]),
        Route(
            "/admin/eval/retrieval/{job_id}",
            eval_retrieval_status,
            methods=["GET"],
        ),
    ]
)
