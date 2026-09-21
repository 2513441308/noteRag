"""知识库网页管理服务：在源文档目录与 Chroma 集合之间做文件级 CRUD.

设计与 :mod:`agent.ingestion.sync` 一致——**源文档目录为准、向量库为派生**：
每个写操作都同时维护磁盘文件与其在集合中的切片，保持两侧一致。

- 切片共享入库时写入的 ``source``（文件绝对路径）与 ``content_hash`` 元数据，
  因此单文件更新 = 按 source 删旧切片 → 重新切分嵌入；
- 复用 sync 的 ``_index_files`` 完成 加载→切分→打指纹→写入 链路；
- 嵌入模型与向量库连接默认从 :class:`~agent.config.Config` 读取，也可显式
  注入（单元测试用 fake embedding + 临时目录全程离线）；
- 本模块**不触碰检索节点缓存**：BM25 是内存索引，由调用方（admin 路由）
  在写操作后清空 rag 节点缓存，使其下次问答时从集合重新导出建索引。

仅适用于本地开发：根目录之外的任何路径都会被拒绝（防路径穿越）。
"""

from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from agent.config import ROOT_DIR, Config
from agent.embedding.embedding_factory import create_embedding_model
from agent.ingestion.loader import _SUPPORT_SUFFIX
from agent.ingestion.sync import _index_files, _sha256_file
from agent.ingestion.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# 可在网页中直接编辑正文的文本类后缀；PDF 等二进制只能上传/替换/删除
EDITABLE_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".json", ".py"})

# 单次上传体积上限（20 MiB），与调试面板的本地用途相匹配
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class KbError(Exception):
    """业务错误：携带建议的 HTTP 状态码与面向用户的中文信息."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        """携带 HTTP ``status`` 与用户可见 ``message`` 初始化业务错误."""
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class KbAdmin:
    """知识库管理器：绑定一个源文档根目录与一个 Chroma 集合.

    嵌入模型留空时每次操作经工厂获取（工厂自带实例缓存），从而能跟随
    EMBEDDING_* 配置热更新；测试时注入 fake embedding 即可完全离线。
    """

    docs_dir: Path
    collection_name: str
    persist_directory: Path
    provider: str = "chroma"
    injected_embedding: Embeddings | None = None
    encoding: str = "utf-8"

    @classmethod
    def from_config(cls) -> KbAdmin:
        """按当前 ``Config`` 构造：解析知识库目录与 Chroma 连接（与 rag/sync 同规则）."""
        if Config.VECTORSTORE_PROVIDER.lower() != "chroma":
            raise KbError(
                "网页知识库管理目前仅支持 Chroma，"
                f"当前 VECTORSTORE_PROVIDER={Config.VECTORSTORE_PROVIDER}。",
                status=501,
            )
        docs_dir = Path(Config.KNOWLEDGE_DOCS_DIR)
        if not docs_dir.is_absolute():
            docs_dir = ROOT_DIR / docs_dir
        persist = Path(Config.VECTORSTORE_PATH)
        if not persist.is_absolute():
            persist = ROOT_DIR / persist
        return cls(
            docs_dir=docs_dir.resolve(),
            collection_name=Config.VECTORSTORE_COLLECTION_NAME,
            persist_directory=persist.resolve(),
            provider="chroma",
        )

    # ------------------------------------------------------------------
    # 内部基础设施
    # ------------------------------------------------------------------
    def _embedding(self) -> Embeddings:
        return self.injected_embedding or create_embedding_model()

    def _store(self) -> VectorStore:
        """获取与 rag 节点同一工厂缓存的向量库实例（同目录即同实例，写入互见）."""
        return get_vectorstore(
            self.collection_name,
            self._embedding(),
            provider=self.provider,
            persist_directory=self.persist_directory,
        )

    def _collection(self) -> Collection:
        """取出 Chroma 底层集合，非 Chroma 后端直接抛出 501 业务错误."""
        collection = getattr(self._store(), "_collection", None)
        if collection is None:
            raise KbError("当前向量库未暴露底层集合，无法做文件级管理。", status=501)
        # getattr 动态取值静态类型为 Any，运行时已确认是 Chroma Collection
        return cast(Collection, collection)

    def _resolve(self, rel_path: str) -> Path:
        """把相对路径安全解析到知识库根目录内，拒绝一切越界/非法形态.

        Raises:
            路径为空、绝对路径、含 NUL、跳出根目录或不是受支持文件时抛出 KbError。
        """
        if not rel_path or not rel_path.strip():
            raise KbError("文件路径不能为空。", status=400)

        rel = rel_path.strip().replace("\\", "/").lstrip("/")
        if "\x00" in rel or Path(rel_path).is_absolute() or rel.startswith("/"):
            raise KbError("只允许知识库目录内的相对路径。", status=400)

        candidate = self.docs_dir / rel
        # resolve() 会折叠所有 ../ 符号链接，再用相对判定确保无法越界
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.docs_dir)
        except ValueError as exc:
            raise KbError(
                f"路径越界，禁止访问知识库目录之外：{rel_path}", status=400
            ) from exc

        if resolved == self.docs_dir or resolved.suffix == "":
            raise KbError("必须指定一个具体的文件。", status=400)
        if resolved.suffix.lower() not in _SUPPORT_SUFFIX:
            raise KbError(
                f"不支持的文件类型 {resolved.suffix}，"
                f"支持：{', '.join(sorted(_SUPPORT_SUFFIX))}",
                status=400,
            )
        return resolved

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.docs_dir).as_posix()
        except ValueError:
            return str(path)

    def _index_one(self, abs_path: Path) -> int:
        """删旧切片后重新切分嵌入单个文件，返回新切片数."""
        abs_path = abs_path.resolve()
        fs_hash = _sha256_file(abs_path)
        collection = self._collection()
        collection.delete(where={"source": str(abs_path)})
        _, chunk_count, failed = _index_files(
            [str(abs_path)],
            fs_hashes={str(abs_path): fs_hash},
            store=self._store(),
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            code_max_chars=Config.CODE_MAX_CHARS,
            encoding=self.encoding,
        )
        if failed:
            raise KbError(
                f"文件已保存但索引失败（加载/切分阶段报错）：{abs_path.name}，"
                "请检查文件内容与格式。",
                status=422,
            )
        return chunk_count

    # ------------------------------------------------------------------
    # 查
    # ------------------------------------------------------------------
    def list_documents(self) -> dict[str, Any]:
        """列出源文件及其索引状态，并给出集合中的游离切片（孤儿）清单."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        # 文件系统：受支持后缀 → 元信息
        fs_files: dict[str, dict[str, Any]] = {}
        for path in sorted(self.docs_dir.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in _SUPPORT_SUFFIX):
                continue
            stat = path.stat()
            abs_path = str(path.resolve())
            fs_files[abs_path] = {
                "path": self._rel(path),
                "name": path.name,
                "suffix": path.suffix.lower(),
                "editable": path.suffix.lower() in EDITABLE_SUFFIXES,
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
                "content_hash": _sha256_file(path),
            }

        # 集合：按 source 聚合切片数与指纹
        data = self._collection().get(include=["metadatas"])
        index: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas") or []:
            # Chroma metadata 值是宽联合类型，source 入库时写入的是字符串
            source = (metadata or {}).get("source")
            if not isinstance(source, str) or not source:
                continue
            key = str(Path(source).resolve())
            entry = index.setdefault(key, {"chunks": 0, "hash": None})
            entry["chunks"] += 1
            if entry["hash"] is None:
                entry["hash"] = (metadata or {}).get("content_hash")

        documents = []
        for abs_path, info in fs_files.items():
            indexed = abs_path in index
            if not indexed:
                status = "unindexed"
            elif index[abs_path]["hash"] != info["content_hash"]:
                status = "modified"
            else:
                status = "synced"
            documents.append(
                {
                    **info,
                    "indexed": indexed,
                    "status": status,
                    "chunks": index.get(abs_path, {}).get("chunks", 0),
                }
            )

        # 孤儿：集合里有切片、源文件已不存在
        orphans = []
        for abs_path, entry in index.items():
            if abs_path in fs_files:
                continue
            orphans.append(
                {
                    "source": abs_path,
                    "rel": self._rel(Path(abs_path)),
                    "chunks": entry["chunks"],
                }
            )

        return {
            "root": str(self.docs_dir),
            "total_files": len(documents),
            "total_chunks": sum(e["chunks"] for e in index.values()),
            "documents": documents,
            "orphans": orphans,
            "editable_suffixes": sorted(EDITABLE_SUFFIXES),
            "supported_suffixes": sorted(_SUPPORT_SUFFIX),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }

    def read_text(self, rel_path: str) -> dict[str, Any]:
        """读取文本类文件正文；二进制文件拒绝在线编辑."""
        path = self._resolve(rel_path)
        if not path.exists():
            raise KbError(f"文件不存在：{self._rel(path)}", status=404)
        if path.suffix.lower() not in EDITABLE_SUFFIXES:
            raise KbError(
                f"{path.suffix} 为二进制/不支持在线编辑的类型，请用「上传替换」更新。",
                status=415,
            )
        try:
            content = path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as exc:
            raise KbError(
                f"文件不是合法的 {self.encoding} 文本，无法在线编辑。", status=422
            ) from exc
        return {"path": self._rel(path), "content": content, "editable": True}

    # ------------------------------------------------------------------
    # 增 / 改
    # ------------------------------------------------------------------
    def create_text(self, rel_path: str, content: str) -> dict[str, Any]:
        """在线新建文本文件并入库；同名文件已存在则冲突."""
        path = self._resolve(rel_path)
        if path.suffix.lower() not in EDITABLE_SUFFIXES:
            raise KbError(
                f"在线新建仅支持文本类型：{', '.join(sorted(EDITABLE_SUFFIXES))}",
                status=415,
            )
        if path.exists():
            raise KbError(f"文件已存在：{self._rel(path)}", status=409)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=self.encoding)
        chunks = self._index_one(path)
        return {"path": self._rel(path), "chunks": chunks, "action": "created"}

    def update_text(self, rel_path: str, content: str) -> dict[str, Any]:
        """覆写文本文件并按 source 替换其全部切片."""
        path = self._resolve(rel_path)
        if not path.exists():
            raise KbError(f"文件不存在：{self._rel(path)}", status=404)
        if path.suffix.lower() not in EDITABLE_SUFFIXES:
            raise KbError(
                f"{path.suffix} 不支持在线编辑，请用「上传替换」。", status=415
            )
        path.write_text(content, encoding=self.encoding)
        chunks = self._index_one(path)
        return {"path": self._rel(path), "chunks": chunks, "action": "updated"}

    def upload(
        self, rel_path: str, content_base64: str, *, overwrite: bool
    ) -> dict[str, Any]:
        """上传（或覆盖）任意受支持文件并入库，内容以 base64 传输（免 multipart 依赖）."""
        path = self._resolve(rel_path)
        if not content_base64:
            raise KbError("文件内容为空。", status=400)
        try:
            data = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise KbError("文件内容不是合法的 base64。", status=400) from exc
        if not data:
            raise KbError("文件内容为空。", status=400)
        if len(data) > MAX_UPLOAD_BYTES:
            raise KbError(
                f"文件过大：{len(data)} 字节，上限 {MAX_UPLOAD_BYTES} 字节。",
                status=413,
            )
        existed = path.exists()
        if existed and not overwrite:
            raise KbError(
                f"文件已存在：{self._rel(path)}（如需替换请确认覆盖）", status=409
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        chunks = self._index_one(path)
        return {
            "path": self._rel(path),
            "chunks": chunks,
            "action": "replaced" if existed else "created",
        }

    # ------------------------------------------------------------------
    # 删 / 重建 / 清理
    # ------------------------------------------------------------------
    def delete(self, rel_path: str) -> dict[str, Any]:
        """删除源文件并按 source 清掉其全部切片（文件缺失也尽力清索引）."""
        path = self._resolve(rel_path)
        abs_path = str(path.resolve())
        existed = path.exists()
        if existed:
            path.unlink()
            self._remove_empty_parents(path.parent)
        self._collection().delete(where={"source": abs_path})
        return {"path": self._rel(path), "removed_file": existed}

    def _remove_empty_parents(self, parent: Path) -> None:
        """删除文件后向上回收空子目录，遇到知识库根或非空目录即停."""
        root = self.docs_dir
        while parent != root:
            try:
                parent.relative_to(root)
            except ValueError:
                break
            try:
                parent.rmdir()  # 仅空目录可删，非空抛 OSError
            except OSError:
                break
            parent = parent.parent

    def reindex(self, rel_path: str) -> dict[str, Any]:
        """对单个已存在文件重新切分嵌入（索引缺失/与旧版参数不一致时用）."""
        path = self._resolve(rel_path)
        if not path.exists():
            raise KbError(f"文件不存在：{self._rel(path)}", status=404)
        chunks = self._index_one(path)
        return {"path": self._rel(path), "chunks": chunks, "action": "reindexed"}

    def prune_orphans(self) -> dict[str, Any]:
        """删除源文件已不存在的游离切片（外部手动删文件后用于对齐索引）."""
        removed = []
        collection = self._collection()
        for orphan in self.list_documents()["orphans"]:
            collection.delete(where={"source": orphan["source"]})
            removed.append(orphan["rel"])
        return {"removed": removed, "count": len(removed)}
