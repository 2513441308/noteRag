"""知识库增量同步 sync 的单元测试.

用 DeterministicFakeEmbedding + 临时 Chroma 目录，全程离线、不加载 bge 权重。
覆盖 dry-run / 新增 / 幂等 / 更新（多 chunk 按 source 替换）/ 删除 /
空扫描防误删 / rebuild / 纯函数 compute_plan。
"""

from uuid import uuid4

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from agent.ingestion import sync
from agent.ingestion.sync import CONTENT_HASH_KEY, compute_plan, scan_documents
from agent.ingestion.vectorstore import get_vectorstore

# 足够长，确保切出多个 chunk（默认 chunk_size=1500）
_LONG_BODY = "StateGraph conditional edges workflow checkpointer. " * 80


@pytest.fixture
def env(tmp_path):
    """构造隔离的文档目录 / chroma 目录 / 唯一集合名."""

    class _Env:
        docs = tmp_path / "docs"
        chroma = tmp_path / "chroma"
        collection = f"sync_{uuid4().hex[:12]}"
        embeddings = DeterministicFakeEmbedding(size=32)

        def _store(self):
            return get_vectorstore(
                self.collection,
                self.embeddings,
                provider="chroma",
                persist_directory=self.chroma,
            )

        def run(self, paths=None, **kwargs):
            return sync.sync_documents(
                self.docs if paths is None else paths,
                persist_directory=self.chroma,
                collection_name=self.collection,
                embedding_model=self.embeddings,
                **kwargs,
            )

        def count(self) -> int:
            # 生产代码 sync 内部本来就用底层集合；测试经 getattr 访问私有属性
            return getattr(self._store(), "_collection").count()

        def doc_texts(self, name: str) -> list[str]:
            data = getattr(self._store(), "_collection").get(
                where={"source": str((self.docs / name).resolve())},
                include=["documents", "metadatas"],
            )
            return data["documents"]

    instance = _Env()
    instance.docs.mkdir()
    return instance


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_compute_plan_three_way_diff() -> None:
    plan = compute_plan(
        fs_hashes={"/a": "h1", "/b": "h2", "/c": "h3"},
        store_index={"/a": "h1", "/b": "OLD", "/d": "h4"},
    )
    assert plan.added == ["/c"]
    assert plan.updated == ["/b"]
    assert plan.deleted == ["/d"]
    assert plan.unchanged == ["/a"]
    assert plan.changed == 3


def test_compute_plan_missing_hash_treated_as_update() -> None:
    plan = compute_plan(fs_hashes={"/a": "h1"}, store_index={"/a": None})
    assert plan.updated == ["/a"]
    assert plan.hash_missing_files == 1


def test_scan_recursive_skips_unsupported_suffix(env) -> None:
    (env.docs / "keep.md").write_text("# hi", encoding="utf-8")
    (env.docs / "skip.bin").write_text("binary", encoding="utf-8")
    sub = env.docs / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")

    hashes = scan_documents(env.docs)
    names = {path.rsplit("/", 1)[-1] for path in hashes}
    assert "keep.md" in names and "nested.txt" in names
    assert all(not path.endswith(".bin") for path in hashes)


# ---------------------------------------------------------------------------
# 增量同步主链路
# ---------------------------------------------------------------------------


def test_dry_run_previews_without_writing(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nhello", encoding="utf-8")
    result = env.run()
    assert result.applied is False
    assert len(result.plan.added) == 1
    assert env.count() == 0  # 库完全没动


def test_initial_apply_embeds_with_hash_metadata(env) -> None:
    (env.docs / "a.md").write_text(
        f"# A\n\n{_LONG_BODY}\nOLD_ALPHA_MARKER", encoding="utf-8"
    )
    result = env.run(apply=True)

    assert result.embedded_files == 1
    assert result.embedded_chunks >= 2  # 长文档多 chunk
    store = get_vectorstore(
        env.collection, env.embeddings, provider="chroma", persist_directory=env.chroma
    )
    metas = getattr(store, "_collection").get(include=["metadatas"])["metadatas"]
    assert all(meta[CONTENT_HASH_KEY] for meta in metas)
    assert all(meta["source"].startswith("/") for meta in metas)  # 归一化绝对路径


def test_sync_is_idempotent(env) -> None:
    (env.docs / "a.md").write_text(f"# A\n\n{_LONG_BODY}", encoding="utf-8")
    env.run(apply=True)

    preview = env.run()
    assert preview.plan.changed == 0
    assert len(preview.plan.unchanged) == 1

    applied = env.run(apply=True)
    assert applied.embedded_files == 0 and applied.embedded_chunks == 0


def test_update_replaces_all_chunks_without_duplication(env) -> None:
    (env.docs / "a.md").write_text(
        f"# A\n\n{_LONG_BODY}\nOLD_MARKER_XYZ", encoding="utf-8"
    )
    env.run(apply=True)
    before = env.count()
    assert before >= 2

    (env.docs / "a.md").write_text(
        f"# A v2\n\n{_LONG_BODY.replace('checkpointer', 'persistence')}\nNEW_MARKER_XYZ",
        encoding="utf-8",
    )
    plan = env.run()
    assert [p.endswith("a.md") for p in plan.plan.updated]

    env.run(apply=True)
    texts = env.doc_texts("a.md")
    assert len(texts) == before  # 旧多 chunk 全删、新 chunk 补齐，无重复残留
    assert all("OLD_MARKER_XYZ" not in text for text in texts)
    assert any("NEW_MARKER_XYZ" in text for text in texts)
    assert env.count() == before


def test_delete_removes_its_chunks(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nkeep me", encoding="utf-8")
    (env.docs / "b.md").write_text("# B\n\ndelete me", encoding="utf-8")
    env.run(apply=True)
    assert env.count() == 2

    (env.docs / "b.md").unlink()
    env.run(apply=True)
    assert env.doc_texts("b.md") == []
    assert len(env.doc_texts("a.md")) == 1


def test_added_file_embedded_incrementally(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nfirst", encoding="utf-8")
    env.run(apply=True)
    (env.docs / "c.md").write_text("# C\n\nRunnableConfig config.", encoding="utf-8")

    result = env.run(apply=True)
    assert len(result.plan.added) == 1
    assert len(env.doc_texts("c.md")) == 1
    assert result.failed_files == []


def test_empty_scan_is_rejected_and_store_untouched(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nx", encoding="utf-8")
    env.run(apply=True)

    with pytest.raises(RuntimeError, match="没有扫描到"):
        env.run(paths=env.docs.parent / "no_such_dir", apply=True)
    assert env.count() == 1  # 拒绝时库未被清空


def test_allow_empty_clears_store_incrementally(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nx", encoding="utf-8")
    env.run(apply=True)
    assert env.count() == 1

    empty_dir = env.docs.parent / "empty"
    empty_dir.mkdir()
    result = env.run(paths=empty_dir, apply=True, allow_empty=True)
    assert result.plan.added == []
    assert len(result.plan.deleted) == 1
    assert env.count() == 0


def test_rebuild_makes_store_match_directory(env) -> None:
    (env.docs / "a.md").write_text("# A\n\nkeep alpha", encoding="utf-8")
    (env.docs / "b.md").write_text("# B\n\nwill be gone", encoding="utf-8")
    env.run(apply=True)
    (env.docs / "b.md").unlink()
    (env.docs / "c.md").write_text("# C\n\nnew gamma", encoding="utf-8")

    preview = env.run(rebuild=True)
    assert preview.applied is False and preview.rebuilt is True

    result = env.run(rebuild=True, apply=True)
    assert result.rebuilt is True
    assert env.count() == 2
    assert len(env.doc_texts("a.md")) == 1
    assert len(env.doc_texts("c.md")) == 1
    assert env.doc_texts("b.md") == []
