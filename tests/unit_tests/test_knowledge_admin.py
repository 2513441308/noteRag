"""知识库网页管理服务 KbAdmin 的单元测试.

用 DeterministicFakeEmbedding + 临时文档/Chroma 目录，全程离线、不加载 bge 权重。
覆盖：空目录列表、新建/重复建、读回、在线编辑切片替换、上传覆盖、删除、
重新索引、路径穿越/绝对路径/空路径/非法后缀、以及游离切片清理（prune）。
"""

import base64
from uuid import uuid4

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from agent.knowledge_admin import EDITABLE_SUFFIXES, KbAdmin, KbError


def _long(token: str) -> str:
    """足够长（>默认 chunk_size 1500），保证同一文件切出多个 chunk."""
    return (f"{token} " + "StateGraph conditional edges workflow checkpointer. " * 60).strip()


@pytest.fixture
def kb(tmp_path):
    docs = tmp_path / "kb"
    chroma = tmp_path / "chroma"
    docs.mkdir()
    admin = KbAdmin(
        docs_dir=docs.resolve(),
        collection_name=f"kb_{uuid4().hex[:12]}",
        persist_directory=chroma.resolve(),
        injected_embedding=DeterministicFakeEmbedding(size=32),
    )
    return admin


def _chunks_for(kb: KbAdmin, rel: str) -> list[str]:
    """直接查底层集合，返回某文件全部切片正文."""
    source = str((kb.docs_dir / rel).resolve())
    data = kb._collection().get(where={"source": source}, include=["documents"])
    return data["documents"]


def test_list_empty_dir(kb: KbAdmin) -> None:
    listing = kb.list_documents()
    assert listing["documents"] == []
    assert listing["orphans"] == []
    assert listing["total_chunks"] == 0
    # 受支持/可编辑后缀白名单也下发给前端
    assert ".md" in listing["editable_suffixes"]
    assert ".pdf" in listing["supported_suffixes"]


def test_create_then_list_synced(kb: KbAdmin) -> None:
    res = kb.create_text("notes/hello.md", _long("alpha"))
    assert res["action"] == "created"
    assert res["chunks"] >= 2  # 长文切多片

    listing = kb.list_documents()
    assert listing["total_files"] == 1
    doc = listing["documents"][0]
    assert doc["path"] == "notes/hello.md"
    assert doc["status"] == "synced"
    assert doc["chunks"] == res["chunks"]
    assert doc["editable"] is True
    assert (kb.docs_dir / "notes/hello.md").exists()


def test_create_duplicate_conflict(kb: KbAdmin) -> None:
    kb.create_text("a.md", "hello")
    with pytest.raises(KbError) as exc:
        kb.create_text("a.md", "again")
    assert exc.value.status == 409


def test_read_text_roundtrip(kb: KbAdmin) -> None:
    kb.create_text("b.txt", "你好，知识库")
    got = kb.read_text("b.txt")
    assert got["content"] == "你好，知识库"
    assert got["editable"] is True


def test_read_missing_is_404(kb: KbAdmin) -> None:
    with pytest.raises(KbError) as exc:
        kb.read_text("nope.md")
    assert exc.value.status == 404


def test_binary_not_editable(kb: KbAdmin) -> None:
    # 直接落一个假二进制（绕过入库），在线读取应被拒
    (kb.docs_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(KbError) as exc:
        kb.read_text("paper.pdf")
    assert exc.value.status == 415


def test_update_replaces_chunks(kb: KbAdmin) -> None:
    kb.create_text("doc.md", _long("OLD_UNIQUE_TOKEN"))
    kb.update_text("doc.md", _long("NEW_UNIQUE_TOKEN"))

    texts = _chunks_for(kb, "doc.md")
    joined = " ".join(texts)
    assert "NEW_UNIQUE_TOKEN" in joined
    assert "OLD_UNIQUE_TOKEN" not in joined
    # 更新后状态回到已同步
    doc = next(d for d in kb.list_documents()["documents"] if d["path"] == "doc.md")
    assert doc["status"] == "synced"


def test_update_missing_is_404(kb: KbAdmin) -> None:
    with pytest.raises(KbError) as exc:
        kb.update_text("x.md", "data")
    assert exc.value.status == 404


def test_upload_create_and_overwrite(kb: KbAdmin) -> None:
    payload = base64.b64encode(_long("uploaded").encode()).decode()

    created = kb.upload("data.txt", payload, overwrite=False)
    assert created["action"] == "created"
    assert created["chunks"] >= 2

    with pytest.raises(KbError) as exc:
        kb.upload("data.txt", payload, overwrite=False)
    assert exc.value.status == 409

    replaced = kb.upload("data.txt", payload, overwrite=True)
    assert replaced["action"] == "replaced"


def test_upload_rejects_bad_base64(kb: KbAdmin) -> None:
    with pytest.raises(KbError) as exc:
        kb.upload("a.txt", "@@not-base64@@", overwrite=False)
    assert exc.value.status == 400


def test_delete_removes_file_and_chunks(kb: KbAdmin) -> None:
    kb.create_text("doomed.md", _long("doomed"))
    assert _chunks_for(kb, "doomed.md")

    res = kb.delete("doomed.md")
    assert res["removed_file"] is True
    assert not (kb.docs_dir / "doomed.md").exists()
    assert _chunks_for(kb, "doomed.md") == []


def test_delete_prunes_empty_parent_dirs(kb: KbAdmin) -> None:
    kb.create_text("folder/sub/leaf.md", "some content here")
    assert (kb.docs_dir / "folder/sub/leaf.md").exists()
    kb.delete("folder/sub/leaf.md")
    # 文件删完后，逐层变空的子目录应一并回收（根目录保留）
    assert not (kb.docs_dir / "folder").exists()
    assert kb.docs_dir.exists()


def test_delete_keeps_nonempty_parent_dir(kb: KbAdmin) -> None:
    kb.create_text("keep/a.md", "a")
    kb.create_text("keep/b.md", "b")
    kb.delete("keep/a.md")
    # 目录里还有 b.md，不能把 keep 删掉
    assert (kb.docs_dir / "keep/b.md").exists()


def test_delete_missing_file_still_clears_index(kb: KbAdmin) -> None:
    kb.create_text("ghost.md", _long("ghost"))
    (kb.docs_dir / "ghost.md").unlink()  # 外部直接删文件，索引残留
    res = kb.delete("ghost.md")
    assert res["removed_file"] is False
    assert _chunks_for(kb, "ghost.md") == []


def test_reindex_restores_sync(kb: KbAdmin) -> None:
    kb.create_text("r.md", _long("one"))
    # 外部改动文件 → 列表显示已修改
    (kb.docs_dir / "r.md").write_text(_long("two"), encoding="utf-8")
    doc = next(d for d in kb.list_documents()["documents"] if d["path"] == "r.md")
    assert doc["status"] == "modified"

    res = kb.reindex("r.md")
    assert res["action"] == "reindexed"
    doc = next(d for d in kb.list_documents()["documents"] if d["path"] == "r.md")
    assert doc["status"] == "synced"


def test_reindex_missing_is_404(kb: KbAdmin) -> None:
    with pytest.raises(KbError) as exc:
        kb.reindex("nope.md")
    assert exc.value.status == 404


def test_prune_orphans(kb: KbAdmin) -> None:
    kb.create_text("keep.md", _long("keep"))
    kb.create_text("gone.md", _long("gone"))
    (kb.docs_dir / "gone.md").unlink()

    orphans = kb.list_documents()["orphans"]
    assert len(orphans) == 1
    gone_source = str((kb.docs_dir / "gone.md").resolve())
    gone_chunks = len(kb._collection().get(where={"source": gone_source})["ids"])
    before = kb._collection().count()

    result = kb.prune_orphans()
    assert result["count"] == 1
    assert kb._collection().count() == before - gone_chunks
    assert kb.list_documents()["orphans"] == []


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "   ",
        "../../../etc/passwd",
        "sub/../../escape.md",
        "/etc/passwd",
        "a.exe",
        "noext",
    ],
)
def test_unsafe_or_unsupported_paths_rejected(kb: KbAdmin, bad_path: str) -> None:
    with pytest.raises(KbError) as exc:
        kb.create_text(bad_path, "x")
    assert exc.value.status in (400,)


def test_editable_suffixes_constant() -> None:
    # 防止误把二进制类型加入在线编辑白名单
    assert ".pdf" not in EDITABLE_SUFFIXES
    assert {".md", ".txt", ".json", ".py"} <= EDITABLE_SUFFIXES
