"""通用 JSON 文档加载器.

不依赖 langchain-community 的 JSONLoader（它强制要求与数据结构强绑定的
``jq_schema``，且强依赖 jq 原生库），改用标准库实现，自动兼容对象根/数组根。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class GenericJSONLoader(BaseLoader):
    """通用 JSON 加载器，无需 jq_schema，也不依赖 jq 原生库.

    - 根为数组：每个数组元素生成一个 Document（适合多条记录的知识库导出）；
    - 根为对象/标量：整个文件生成一个 Document。
    """

    def __init__(self, file_path: str | Path, encoding: str = "utf-8") -> None:
        """初始化加载器.

        Args:
            file_path: JSON 文件路径。
            encoding: 文件编码，默认 utf-8。
        """
        self.file_path = Path(file_path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载 JSON 并逐元素产出 Document."""
        data = json.loads(self.file_path.read_text(encoding=self.encoding))
        source = str(self.file_path)
        items = data if isinstance(data, list) else [data]
        for seq, item in enumerate(items, start=1):
            yield Document(
                page_content=json.dumps(item, ensure_ascii=False, indent=2),
                metadata={"source": source, "seq_num": seq},
            )
