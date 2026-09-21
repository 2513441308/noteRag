"""结构优先的 Markdown 文档切分器.

兼容普通 Markdown 与官网抓取的 MDX 文档（<CodeGroup>/<Tab> 深缩进代码块、
`````bash pip theme={...}````` 带属性的 info string、blockquote 内围栏等），
保证代码围栏与标题不被拦腰切断。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

# 词法原子块：形状随首元素变化
# ("heading", level: int, title: str) / ("code", lang: str, code: str) / ("text", para: str)
# 统一用可变长元组承载，按 kind 分支后再按位置取字段
Block = tuple[Any, ...]

# 代码块内部的"安全边界"行：空行 / 注释 / 顶层定义。切开超长代码块时只在这些位置下刀
_CODE_BOUNDARY_RE = re.compile(
    r"^\s*($|#|//|/\*|\*|\"\"\"|'''|--|@|def\s|class\s|async\s+def\s|import\s|from\s)"
)


class MarkdownCodeTextSplitter(TextSplitter):
    """结构优先的 Markdown 切分器，保证代码块与标题不被拦腰切断.

    切分策略（三级降级）：
      1. 词法解析：用围栏状态机逐行扫描，识别 heading / code(```围栏) / text 三类原子块；
         围栏内的 ``#`` 注释不会被误判为标题；
      2. 贪心打包：沿标题层级把原子块累积到 chunk_size 内，每个 chunk 携带标题路径；
         当前 chunk 放不下一个完整代码块时，先封存当前 chunk，代码块整体进入新 chunk；
         遇到 heading_boundary 级及以上的标题强制另起一块；
      3. 超长降级：普通文本按 段落→句子→字符 切分（带 overlap）；
         超过 code_max_chars 的代码块只在 空行/注释/定义 边界切分，每片自动补回围栏。

    注意：长度不超过 code_max_chars 的代码块始终整块保留，即使它单独超过 chunk_size
    （此时该 chunk 会偏大），这是为代码完整性有意做的权衡。
    """

    # 容忍任意缩进（MDX 的 <CodeGroup>/<Tab>/列表项内代码块）和 blockquote 前缀（> ）；
    # info string 允许任意属性，如 ```bash pip theme={"theme":...}，语言取第一个单词
    _FENCE_RE = re.compile(
        r"^[ \t]*(?:>[ \t]*)*(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$"
    )
    _HEADING_RE = re.compile(
        r"^[ \t]*(?:>[ \t]*)*(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$"
    )

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        code_max_chars: int = 3000,
        heading_boundary: int = 3,
        **kwargs: Any,
    ) -> None:
        """初始化切分器.

        Args:
            chunk_size: 普通文本 chunk 的目标字符数。
            chunk_overlap: 普通文本跨 chunk 的重叠字符数。
            code_max_chars: 代码块完整保留的字符上限，超过才在安全边界切分。
            heading_boundary: 几级及以上标题强制另起 chunk（3 表示 #/##/###）。
            **kwargs: 透传给基类 ``TextSplitter`` 的额外参数（如 separators）。
        """
        kwargs.setdefault("length_function", len)
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._max_chars = chunk_size
        self.code_max_chars = code_max_chars
        self.heading_boundary = heading_boundary
        # 普通文本超长时的降级切分（代码块不走这里，因此不会被它切坏）
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get(
                "separators", ["\n\n", "\n", "。", ". ", "? ", "！", "! ", " ", ""]
            ),
            length_function=len,
        )

    # ---- 第 1 级：词法解析 -------------------------------------------------

    @staticmethod
    def _strip_container_lead(line: str, lead: str) -> str:
        """剥离开围栏所在容器的前缀（缩进 / > 引用），保留代码自身缩进.

        例如围栏在 MDX 容器内缩进 7 空格，代码行为 11 空格时只剥 7 空格，
        使 Python 代码自身的 4 空格缩进得以保留。

        Args:
            line: 围栏内的原始代码行。
            lead: 开围栏标记之前的前缀，如 '       ' 或 '> ' 或 '  > > '。

        Returns:
            剥离容器前缀后的代码行。
        """
        rest = line
        # 1) 第一个 > 之前的前导空白（纯缩进容器在此一步完成）
        i = 0
        while i < len(lead) and lead[i] in " \t" and i < len(rest) and rest[i] in " \t":
            i += 1
        rest = rest[i:]
        # 2) 逐层剥 blockquote 标记 '>' 及其后一个可选空白
        for _ in range(lead.count(">")):
            qm = re.match(r"^>[ \t]?", rest)
            if not qm:
                break
            rest = rest[qm.end() :]
        # 3) 最后一个 '>' 之后的对齐空白（如 '> ' 中 > 后的空格）；
        #    纯缩进 lead 在步骤 1 已剥完，这里不能重复计算
        last_quote = lead.rfind(">")
        tail = len(lead) - last_quote - 1 if last_quote >= 0 else 0
        j = 0
        while j < tail and j < len(rest) and rest[j] in " \t":
            j += 1
        return rest[j:]

    def _parse_blocks(self, text: str) -> list[Block]:
        """把 Markdown 文本解析成原子块序列.

        Returns:
            原子块列表，元素形状为 ``("heading", level, title)`` /
            ``("code", lang, code)`` / ``("text", para)``。
        """
        blocks: list[Block] = []
        text_buf: list[str] = []

        def flush_text() -> None:
            if not text_buf:
                return
            para = "\n".join(text_buf).strip()
            text_buf.clear()
            if para:
                blocks.append(("text", para))

        in_fence = False
        fence_char: str = ""
        fence_len: int = 0
        lang = lead = ""
        code_buf: list[str] = []

        for raw in text.splitlines():
            m = self._FENCE_RE.match(raw)
            if m:
                marker = m.group("fence")
                if not in_fence:
                    flush_text()
                    in_fence = True
                    fence_char, fence_len = marker[0], len(marker)
                    lead = raw[: m.start("fence")]  # 容器前缀（缩进 / > 引用）
                    info_tokens = m.group("info").strip().split()
                    lang = info_tokens[0] if info_tokens else ""
                    code_buf = []
                elif marker[0] == fence_char and len(marker) >= fence_len:
                    blocks.append(("code", lang or "text", "\n".join(code_buf)))
                    in_fence = False
                    lang = lead = ""
                else:
                    # 围栏内出现的更短/异号围栏（如 markdown 教程示例），剥容器前缀后保留
                    code_buf.append(self._strip_container_lead(raw, lead))
                continue
            if in_fence:
                code_buf.append(self._strip_container_lead(raw, lead))
                continue
            hm = self._HEADING_RE.match(raw)
            if hm:
                flush_text()
                blocks.append(
                    ("heading", len(hm.group("hashes")), hm.group("title").strip())
                )
                continue
            if raw.strip() == "":
                flush_text()
            else:
                text_buf.append(raw)

        if in_fence:  # 文件结束仍未闭合的围栏，兜底按代码块处理
            blocks.append(("code", lang or "text", "\n".join(code_buf)))
        flush_text()
        return blocks

    # ---- 第 3 级：超长代码块在安全边界切分 ---------------------------------

    def _split_long_code(self, code: str) -> list[str]:
        """把超长代码块按 空行/注释/定义 边界切成不超过 code_max_chars 的片段."""
        if len(code) <= self.code_max_chars:
            return [code]

        lines = code.splitlines()
        pieces: list[str] = []
        buf: list[str] = []

        def push() -> None:
            if buf:
                pieces.append("\n".join(buf))
                buf.clear()

        for line in lines:
            too_long = (
                sum(len(x) + 1 for x in buf) + len(line) + 1 > self.code_max_chars
            )
            if too_long and buf:
                # 从缓冲末尾向前找最近的安全边界（至少保留第 1 行，避免切出空片）
                cut = 0
                for idx in range(len(buf) - 1, 0, -1):
                    if _CODE_BOUNDARY_RE.match(buf[idx]):
                        cut = idx
                        break
                if cut:
                    pieces.append("\n".join(buf[:cut]))
                    del buf[:cut]
                else:
                    push()
            buf.append(line)
        push()
        return pieces

    # ---- 第 2 级：贪心打包 -------------------------------------------------

    def _build(self, text: str) -> tuple[list[str], list[dict[str, Any]]]:
        """解析并打包，返回 (chunk 文本列表, 对应 metadata 列表)."""
        blocks = self._parse_blocks(text)
        headers: dict[int, str] = {}
        chunks: list[str] = []
        metas: list[dict[str, Any]] = []
        cur: list[str] = []
        cur_langs: set[str] = set()

        def header_path() -> str:
            return " > ".join(headers[k] for k in sorted(headers))

        def flush() -> None:
            if not cur:
                return
            path = header_path()
            prefix = f"[{path}]\n" if path else ""
            chunks.append(prefix + "\n\n".join(cur).strip())
            metas.append(
                {
                    "headers": path,
                    "has_code": bool(cur_langs),
                    "code_lang": ",".join(sorted(cur_langs)) or None,
                }
            )
            cur.clear()
            cur_langs.clear()

        def fits(fragment: str) -> bool:
            return len("\n\n".join(cur + [fragment])) <= self._max_chars

        for block in blocks:
            kind = block[0]
            if kind == "heading":
                _, level, title = block
                if level <= self.heading_boundary and cur:
                    flush()  # 章节边界：强制另起一块
                headers[level] = title
                for deeper in [k for k in headers if k > level]:
                    del headers[deeper]
                cur.append(f"{'#' * level} {title}")
            elif kind == "text":
                pieces = self._text_splitter.split_text(block[1])
                for piece in pieces:
                    if cur and not fits(piece):
                        flush()
                    cur.append(piece)
            else:  # code
                _, code_lang, code = block
                for piece in self._split_long_code(code):
                    wrapped = f"```{code_lang}\n{piece}\n```"
                    if cur and not fits(wrapped):
                        flush()  # 代码块整体进入新 chunk，绝不与前文硬挤
                    cur.append(wrapped)
                    cur_langs.add(code_lang)
        flush()
        return chunks, metas

    def split_text(self, text: str) -> list[str]:
        """切分纯文本，返回 chunk 字符串列表（TextSplitter 抽象接口）."""
        chunks, _ = self._build(text)
        return chunks

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """切分 Document 列表，保留 source 等原始 metadata 并补充标题路径等信息.

        Args:
            documents: load_document/load_dir 的返回结果，
                类型与基类 TextSplitter 保持一致（接受任意可迭代文档）。

        Returns:
            切分后的文档，metadata 新增 headers（标题路径）、
            has_code、code_lang、chunk_index。
        """
        out: list[Document] = []
        for doc in documents:
            chunks, metas = self._build(doc.page_content)
            for idx, (content, extra) in enumerate(zip(chunks, metas)):
                metadata = {**doc.metadata, **extra, "chunk_index": idx}
                out.append(Document(page_content=content, metadata=metadata))
        return out


def split_documents(
    documents: list[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    code_max_chars: int = 3000,
    separators: list[str] | None = None,
) -> list[Document]:
    """文档切分便捷入口：Markdown 走结构优先切分，其他类型走通用递归切分.

    Args:
        documents: 加载后的文档列表。
        chunk_size: 每个 chunk 的目标字符数。
        chunk_overlap: 普通文本 chunk 间重叠字符数。
        code_max_chars: 代码块完整保留的字符上限。
        separators: 非 Markdown 文档的递归切分分隔符；
            Markdown 始终走结构切分，不受此参数影响。

    Returns:
        切分后的文档列表。
    """
    md_splitter = MarkdownCodeTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        code_max_chars=code_max_chars,
    )
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", " ", ""],
        length_function=len,
    )
    out: list[Document] = []
    for doc in documents:
        source = str(doc.metadata.get("source", ""))
        if source.lower().endswith((".md", ".markdown")):
            out.extend(md_splitter.split_documents([doc]))
        else:
            out.extend(fallback.split_documents([doc]))
    return out
