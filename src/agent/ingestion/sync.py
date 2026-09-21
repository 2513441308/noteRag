"""文档目录 ↔ Chroma 向量库增量同步.

BM25 索引在图进程启动时从向量库全量重建，因此"知识库更新"只需保证
Chroma 集合与文档目录一致，BM25 无需单独维护。本模块以文件**原始字节的
SHA-256** 为指纹做三路 diff：

- **新增**：目录里有、集合里没有的文件 → 切分并嵌入；
- **更新**：两侧都有但 hash 变化的文件 → 删旧 chunk 后重新切分嵌入；
- **删除**：集合里有、目录里已不存在的文件 → 删除其全部 chunk；
- **不变**：hash 相同的文件完全跳过（同步幂等）。

``--rebuild`` 则把整个集合删除后全量重建，作为索引异常时的逃生舱。

命令行用法（默认 dry-run，只打印 diff 不动库）::

    python -m agent.ingestion.sync docs/                # 预览变更
    python -m agent.ingestion.sync docs/ --apply        # 执行增量同步
    python -m agent.ingestion.sync docs/ --rebuild      # 预览全量重建
    python -m agent.ingestion.sync docs/ --rebuild --apply  # 执行全量重建

同步后正在运行的图进程需重启才会重建 BM25 索引；下次启动则自动生效。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from agent.config import ROOT_DIR, Config
from agent.embedding.embedding_factory import create_embedding_model
from agent.ingestion.loader import (
    _SUPPORT_SUFFIX,
    _sanitize_metadata,
    load_document,
)
from agent.ingestion.markdown_splitter import split_documents
from agent.ingestion.vectorstore import _vectorstores, get_vectorstore

logger = logging.getLogger(__name__)

# chunk metadata 中存放文件内容指纹的字段名
CONTENT_HASH_KEY = "content_hash"
# 读文件算 hash 时的块大小（1 MiB）
_HASH_BLOCK_SIZE = 1 << 20


@dataclass
class SyncPlan:
    """一次同步的三路 diff 结果（元素均为已归一化的绝对路径）."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    # 集合中缺少 source / content_hash 元数据的历史 chunk 数（仅供审计提示）
    orphan_chunks: int = 0
    hash_missing_files: int = 0

    @property
    def changed(self) -> int:
        """新增 + 更新 + 删除的文件总数."""
        return len(self.added) + len(self.updated) + len(self.deleted)


@dataclass
class SyncResult:
    """同步执行结果（dry-run 时 embedded_* 为 0）."""

    plan: SyncPlan
    applied: bool
    rebuilt: bool
    scanned_files: int = 0
    embedded_files: int = 0
    embedded_chunks: int = 0
    failed_files: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    """计算文件原始字节的 SHA-256（二进制文档如 PDF 同样适用）."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_source(source: str) -> str:
    """把任意形态的 source 路径归一化为绝对路径字符串（同步主键）."""
    return str(Path(source).resolve())


def scan_documents(
    paths: str | Path | list[str | Path],
    *,
    recursive: bool = True,
) -> dict[str, str]:
    """扫描文件/目录，返回 ``{文件绝对路径: 内容 SHA-256}``.

    只纳入 loader 支持的后缀（与 :func:`agent.ingestion.loader.load_dir`
    同一套判定）；单个文件读取失败只告警跳过，不中断整体扫描。

    Args:
        paths: 文件或目录路径，可混合传入。
        recursive: 目录是否递归扫描子目录。

    Returns:
        归一化绝对路径到 SHA-256 的映射；路径不存在时跳过并告警。
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            files.extend(
                p
                for p in iterator
                if p.is_file() and p.suffix.lower() in _SUPPORT_SUFFIX
            )
        elif path.is_file():
            files.append(path)
        else:
            logger.warning(f"同步扫描跳过不存在的路径：{path}")

    hashes: dict[str, str] = {}
    for path in files:
        key = _normalize_source(str(path))
        try:
            hashes[key] = _sha256_file(path)
        except OSError as exc:
            logger.warning(f"跳过无法读取的文件：{path} - {type(exc).__name__}: {exc}")
    return hashes


def _get_raw_collection(store: VectorStore) -> Collection:
    """取出 Chroma 底层集合（metadata 读取 / 按 source 删除需要）.

    Raises:
        向量库实现未暴露 ``_collection``（如 PGVector）时抛出 TypeError。
    """
    collection = getattr(store, "_collection", None)
    if collection is None:
        raise TypeError(
            f"暂不支持对 {type(store).__name__} 做增量同步，当前仅支持 Chroma。"
        )
    # getattr 动态取值静态类型为 Any，运行时已确认是 Chroma Collection
    return cast(Collection, collection)


def read_store_index(store: VectorStore) -> tuple[dict[str, str | None], int]:
    """从集合读取文件索引：``{source 绝对路径: content_hash}``.

    同一文件的所有 chunk 共享同一个 hash；缺失 ``content_hash`` 的历史
    chunk 记为 None（首次同步会被当作"更新"自愈）；缺失 ``source`` 的
    游离 chunk 无法归属，只计数不动它（可用 --rebuild 彻底清理）。

    Returns:
        (路径->hash 或 None, 游离 chunk 数)。
    """
    collection = _get_raw_collection(store)
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    index: dict[str, str | None] = {}
    orphan_chunks = 0
    for metadata in metadatas:
        # Chroma metadata 值是 str/int/float/bool/列表的宽联合类型，
        # source/content_hash 入库时均写入字符串，这里显式收窄
        source = (metadata or {}).get("source")
        if not isinstance(source, str) or not source:
            orphan_chunks += 1
            continue
        key = _normalize_source(source)
        raw_hash = (metadata or {}).get(CONTENT_HASH_KEY)
        file_hash = raw_hash if isinstance(raw_hash, str) else None
        # 同文件多 chunk：任一 chunk 带 hash 即采用（正常情况下全部相同）
        if index.get(key) is None and file_hash is not None:
            index[key] = file_hash
        else:
            index.setdefault(key, file_hash)
    return index, orphan_chunks


def compute_plan(
    fs_hashes: dict[str, str],
    store_index: dict[str, str | None],
    *,
    orphan_chunks: int = 0,
) -> SyncPlan:
    """对比文件系统指纹与集合索引，产出新增/更新/删除/不变清单.

    Args:
        fs_hashes: 文件系统文件路径 -> 内容 SHA-256。
        store_index: 向量库文件路径 -> 内容 SHA-256 或 None。
        orphan_chunks: 游离 chunk 数。

    Returns:
        SyncPlan: 新增/更新/删除/不变清单。
    """
    fs_files = set(fs_hashes)
    store_files = set(store_index)

    plan = SyncPlan(orphan_chunks=orphan_chunks)
    plan.added = sorted(fs_files - store_files)
    plan.deleted = sorted(store_files - fs_files)
    for path in sorted(fs_files & store_files):
        store_hash = store_index[path]
        if store_hash is None:
            plan.updated.append(path)
            plan.hash_missing_files += 1
        elif store_hash != fs_hashes[path]:
            plan.updated.append(path)
        else:
            plan.unchanged.append(path)
    return plan


def _resolve_store_config(
    provider: str | None,
    collection_name: str | None,
    persist_directory: str | Path | None,
) -> tuple[str, str, Path]:
    """解析向量库连接配置（与 loader.ingest_documents 完全一致的回落规则）.

    Args:
        provider: 向量库提供者，如 "chroma" 或 "postgres"。
        collection_name: 向量库集合名。
        persist_directory: 向量库持久化目录。

    Returns:
        三元组 (提供者, 集合名, 目录)。
    """
    provider = (provider or Config.VECTORSTORE_PROVIDER).lower()
    if provider != "chroma":
        raise NotImplementedError("sync 目前仅支持 Chroma，postgres 暂不支持增量同步。")
    collection_name = collection_name or Config.VECTORSTORE_COLLECTION_NAME
    persist = Path(persist_directory or Config.VECTORSTORE_PATH)
    if not persist.is_absolute():
        persist = ROOT_DIR / persist
    return provider, collection_name, persist


def _index_files(
    files: Sequence[str],
    *,
    fs_hashes: dict[str, str],
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
    code_max_chars: int,
    encoding: str,
) -> tuple[int, int, list[str]]:
    """对给定文件执行 加载→切分→打指纹→写入，返回 (成功文件数, chunk 数, 失败清单).

    Args:
        files: 待处理文件路径列表。
        fs_hashes: 文件系统文件路径 -> 内容 SHA-256。
        store: 向量库实例。
        chunk_size: 切分 chunk 大小。
        chunk_overlap: 切分 chunk 重叠大小。
        code_max_chars: 代码块最大字符数。
        encoding: 文件编码。

    Returns:
        三元组 (成功文件数, chunk 数, 失败清单)。
    """
    all_chunks: list[Document] = []
    failed: list[str] = []
    for file_path in files:
        try:
            docs = load_document(Path(file_path), encoding=encoding)
            chunks = split_documents(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                code_max_chars=code_max_chars,
            )
            file_hash = fs_hashes[file_path]
            for chunk in chunks:
                # source 强制写归一化绝对路径，保证跨工作目录主键稳定
                metadata = dict(chunk.metadata or {})
                metadata["source"] = file_path
                metadata[CONTENT_HASH_KEY] = file_hash
                chunk.metadata = _sanitize_metadata(metadata)
            all_chunks.extend(chunks)
        except Exception as exc:  # 单文件失败不影响其余文件入库
            logger.warning(
                f"跳过入库失败的文件：{file_path} - {type(exc).__name__}: {exc}"
            )
            failed.append(file_path)

    if all_chunks:
        store.add_documents(all_chunks)
    return len(files) - len(failed), len(all_chunks), failed


def _apply_incremental(
    store: VectorStore,
    plan: SyncPlan,
    fs_hashes: dict[str, str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    code_max_chars: int,
    encoding: str,
) -> tuple[int, int, list[str]]:
    """应用增量变更：先删 更新/删除 文件的旧 chunk，再嵌入 新增/更新 文件."""
    collection = _get_raw_collection(store)
    for path in plan.deleted + plan.updated:
        collection.delete(where={"source": path})
    logger.info(
        f"已从集合删除 {len(plan.deleted) + len(plan.updated)} 个文件的旧 chunk"
        f"（删除 {len(plan.deleted)} / 更新替换 {len(plan.updated)}）"
    )

    changed = sorted(set(plan.added) | set(plan.updated))
    return _index_files(
        changed,
        fs_hashes=fs_hashes,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        code_max_chars=code_max_chars,
        encoding=encoding,
    )


def _rebuild_collection(
    provider: str,
    collection_name: str,
    persist: Path,
    embedding_model: Embeddings,
    fs_hashes: dict[str, str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    code_max_chars: int,
    encoding: str,
) -> tuple[VectorStore, int, int, list[str]]:
    """删除整个集合后重建：返回 (新 store, 成功文件数, chunk 数, 失败清单).

    Args:
        provider: 向量库提供者，如 "chroma" 或 "postgres"。
        collection_name: 向量库集合名。
        persist: 向量库持久化目录。
        embedding_model: 嵌入模型实例。
        fs_hashes: 文件系统文件路径 -> 内容 SHA-256。
        chunk_size: 切分 chunk 大小。
        chunk_overlap: 切分 chunk 重叠大小。
        code_max_chars: 代码块最大字符数。
        encoding: 文件编码。

    Returns:
        四元组 (新 store, 成功文件数, chunk 数, 失败清单)。
    """
    store = get_vectorstore(
        collection_name,
        embedding_model,
        provider=provider,
        persist_directory=persist,
    )
    client = getattr(store, "_client", None)
    if client is None:
        raise TypeError("Chroma 实例未暴露 _client，无法执行集合级重建。")

    client.delete_collection(collection_name)
    # vectorstore 工厂有实例缓存，旧对象的集合句柄已失效，必须剔除后重建
    _vectorstores.pop((provider, collection_name, str(persist)), None)
    logger.warning(f"已删除集合 {collection_name}，开始全量重建")

    fresh_store = get_vectorstore(
        collection_name,
        embedding_model,
        provider=provider,
        persist_directory=persist,
    )
    indexed, chunk_count, failed = _index_files(
        sorted(fs_hashes),
        fs_hashes=fs_hashes,
        store=fresh_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        code_max_chars=code_max_chars,
        encoding=encoding,
    )
    return fresh_store, indexed, chunk_count, failed


def sync_documents(
    paths: str | Path | list[str | Path],
    *,
    apply: bool = False,
    rebuild: bool = False,
    recursive: bool = True,
    encoding: str = "utf-8",
    collection_name: str | None = None,
    provider: str | None = None,
    persist_directory: str | Path | None = None,
    embedding_model: Embeddings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    code_max_chars: int | None = None,
    allow_empty: bool = False,
) -> SyncResult:
    """文档目录与 Chroma 集合同步的统一入口（默认 dry-run，不改库）.

    Args:
        paths: 待同步的文件/目录路径（与 ingest_documents 同形）。
        apply: False 只计算并返回 diff；True 才真正写入/删除。
        rebuild: True 时忽略增量 diff，删除整个集合后全量重建
            （仍受 apply 控制：不加 apply 只预览）。
        recursive: 目录是否递归扫描。
        encoding: 文本类文件加载编码。
        collection_name: 集合名，默认 Config.VECTORSTORE_COLLECTION_NAME。
        provider: 向量库类型，目前仅支持 chroma。
        persist_directory: chroma 持久化目录，默认 Config.VECTORSTORE_PATH
            （相对路径按项目根 ROOT_DIR 解析，与工作目录无关）。
        embedding_model: 可注入的嵌入模型；None 时按 Config 创建。
        chunk_size: 切分字符数，默认 Config.CHUNK_SIZE。
        chunk_overlap: chunk 重叠字符数，默认 Config.CHUNK_OVERLAP。
        code_max_chars: 代码块保留上限，默认 Config.CODE_MAX_CHARS。
        allow_empty: 扫描到 0 个文件时是否允许继续。默认拒绝，防止路径
            写错（如工作目录不对）导致集合被误清空。

    Returns:
        SyncResult: diff 计划与（执行时的）嵌入统计。

    Raises:
        RuntimeError: 扫描结果为空且未显式允许；或 apply 时有文件全部失败。
        NotImplementedError: provider 不是 chroma。
    """
    fs_hashes = scan_documents(paths, recursive=recursive)
    if not fs_hashes and not allow_empty:
        raise RuntimeError(
            "在给定路径下没有扫描到任何受支持的文档，已中止同步"
            "（防止路径写错误清空向量库）；确需清空请显式传 allow_empty。"
        )

    provider, collection_name, persist = _resolve_store_config(
        provider, collection_name, persist_directory
    )
    if embedding_model is None:
        embedding_model = create_embedding_model()
    chunk_size = Config.CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = Config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    code_max_chars = Config.CODE_MAX_CHARS if code_max_chars is None else code_max_chars

    store = get_vectorstore(
        collection_name,
        embedding_model,
        provider=provider,
        persist_directory=persist,
    )

    if rebuild:
        # 全量重建视角：集合内现存文件全部属于"将被替换"
        store_index, orphan_chunks = read_store_index(store)
        plan = SyncPlan(
            added=sorted(fs_hashes),
            deleted=sorted(store_index),
            orphan_chunks=orphan_chunks,
        )
        if not apply:
            logger.info("dry-run（--rebuild 未加 --apply）：未改动集合")
            return SyncResult(
                plan=plan, applied=False, rebuilt=True, scanned_files=len(fs_hashes)
            )
        _, indexed, chunk_count, failed = _rebuild_collection(
            provider,
            collection_name,
            persist,
            embedding_model,
            fs_hashes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            code_max_chars=code_max_chars,
            encoding=encoding,
        )
        return SyncResult(
            plan=plan,
            applied=True,
            rebuilt=True,
            scanned_files=len(fs_hashes),
            embedded_files=indexed,
            embedded_chunks=chunk_count,
            failed_files=failed,
        )

    store_index, orphan_chunks = read_store_index(store)
    plan = compute_plan(fs_hashes, store_index, orphan_chunks=orphan_chunks)

    if not apply:
        logger.info("dry-run（未加 --apply）：未改动集合")
        return SyncResult(
            plan=plan, applied=False, rebuilt=False, scanned_files=len(fs_hashes)
        )

    indexed, chunk_count, failed = _apply_incremental(
        store,
        plan,
        fs_hashes,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        code_max_chars=code_max_chars,
        encoding=encoding,
    )
    return SyncResult(
        plan=plan,
        applied=True,
        rebuilt=False,
        scanned_files=len(fs_hashes),
        embedded_files=indexed,
        embedded_chunks=chunk_count,
        failed_files=failed,
    )


def _display_path(path: str) -> str:
    """报告中尽量显示相对当前工作目录的短路径，显示不了再用绝对路径."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if not rel.startswith("..") else path


def render_report(result: SyncResult) -> str:
    """把同步结果渲染成人类可读的中文报告（dry-run / apply 通用）."""
    plan = result.plan
    mode = "全量重建" if result.rebuilt else "增量同步"
    action = "已执行" if result.applied else "dry-run（未改动向量库）"
    lines = [
        f"===== 知识库同步报告 | {mode} | {action} =====",
        f"扫描文件 {result.scanned_files} 个："
        f"新增 {len(plan.added)} / 更新 {len(plan.updated)} / "
        f"删除 {len(plan.deleted)} / 不变 {len(plan.unchanged)}",
    ]

    def _dump(title: str, paths: list[str]) -> None:
        if not paths:
            return
        lines.append(f"[{title}] {len(paths)} 个")
        lines.extend(f"  - {_display_path(p)}" for p in paths)

    _dump("新增", plan.added)
    _dump("更新", plan.updated)
    _dump("删除", plan.deleted)

    if result.applied:
        lines.append(
            f"实际嵌入 {result.embedded_files} 个文件、{result.embedded_chunks} 个 chunk"
        )
    if result.failed_files:
        _dump("失败（已跳过）", result.failed_files)
    if plan.hash_missing_files:
        lines.append(
            f"提示：{plan.hash_missing_files} 个文件是旧版入库（无内容指纹），"
            "本次已按更新处理并重嵌入"
        )
    if plan.orphan_chunks:
        lines.append(
            f"提示：集合中有 {plan.orphan_chunks} 个 chunk 缺少 source 元数据、"
            "无法归属文件，增量同步未动它们；如需彻底清理请用 --rebuild"
        )
    if not result.applied and plan.changed:
        lines.append(
            "确认无误后加 --apply 执行；索引异常时可用 --rebuild --apply 全量重建。"
        )
    if not result.applied and not plan.changed and not result.rebuilt:
        lines.append("文档目录与向量库已一致，无需操作。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``python -m agent.ingestion.sync``."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="python -m agent.ingestion.sync",
        description="文档目录与 Chroma 向量库增量同步（默认仅预览，加 --apply 执行）。",
    )
    parser.add_argument("paths", nargs="+", help="文档文件或目录（可多个）")
    parser.add_argument(
        "--apply", action="store_true", help="真正写入变更（缺省只做 dry-run）"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除整个集合后全量重建（仍需配合 --apply 才会执行）",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="目录不递归扫描子目录"
    )
    parser.add_argument("--encoding", default="utf-8", help="文本文件编码，默认 utf-8")
    parser.add_argument(
        "--collection",
        default=None,
        help="覆盖集合名（默认读 VECTORSTORE_COLLECTION_NAME）",
    )
    parser.add_argument(
        "--persist-directory",
        default=None,
        help="覆盖 chroma 持久化目录（默认读 VECTORSTORE_PATH，相对路径按项目根解析）",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="允许扫描结果为空（会导致集合内容被清空，慎用）",
    )
    args = parser.parse_args(argv)

    result = sync_documents(
        args.paths,
        apply=args.apply,
        rebuild=args.rebuild,
        recursive=not args.no_recursive,
        encoding=args.encoding,
        collection_name=args.collection,
        persist_directory=args.persist_directory,
        allow_empty=args.allow_empty,
    )
    # 命令行工具的人类可读报告天然应输出到 stdout，定点豁免项目的 print 禁令
    print(render_report(result))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
