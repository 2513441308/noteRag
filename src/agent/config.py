"""全局配置：从环境变量 / .env 读取所有可调参数."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, TypeVar, final, overload

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 默认离线运行，避免本地模型加载时尝试联网；需要下载模型时在环境中显式置 0
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 项目根目录（config.py 位于 <根>/src/agent/config.py，需上溯三级）
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 显式记录 .env 是否命中：找不到文件时 python-dotenv 会静默返回 False，
# 导致所有配置悄悄回退默认值，这里打日志便于排查“配置为何没生效”
_env_path = ROOT_DIR / ".env"
_env_loaded = load_dotenv(_env_path)
logger.info(
    f"配置文件 {_env_path} 加载{'成功' if _env_loaded else '未找到，全部使用默认值'}"
)

T = TypeVar("T")


@overload
def _get_env(key: str, cast: type[bool], default: bool | None = None) -> bool: ...


@overload
def _get_env(key: str, cast: type[int], default: int) -> int: ...


@overload
def _get_env(key: str, cast: type[float], default: float) -> float: ...


@overload
def _get_env(key: str, cast: type[str], default: str) -> str: ...


@overload
def _get_env(key: str, cast: type[str], default: str | None = None) -> str | None: ...


@overload
def _get_env(key: str, cast: type[T], default: T) -> T: ...


def _get_env(key: str, cast: Any, default: Any = None) -> Any:
    """读取环境变量，自动清洗并转换为指定类型.

    读取环境变量，去除首尾空格。支持 str/int/float/bool 类型转换。
    bool类型特殊处理："true"/"1" 转为 True，"false"/"0" 转为 False。
    环境变量不存在或者为空字符串时返回默认值。类型转换失败时记录警告并返回默认值。

    Args:
        key: 环境变量名称。不能为空字符串。
        cast: 需要转换的目标类型，支持 str, int, float, bool。
        default: 当环境变量不存在时返回的默认值，默认为 None。

    Returns:
        转换完成的环境变量值；环境变量缺失且未给默认值时为 None。

    Raises:
        ValueError: 传入的 key 为空字符串。
    """
    if not key:
        raise ValueError("key 不能为空字符串")

    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default

    try:
        if cast is bool:
            if val.lower() in ("true", "1"):
                return True
            if val.lower() in ("false", "0"):
                return False
            raise ValueError(f"不合法的 bool 值: {val}")
        return cast(val)
    except (ValueError, TypeError):
        # bool 取值不合法、int/float 转换失败等都在此兜底：记录警告并返回默认值，
        # 不让非法配置中断进程（故本函数除“key 为空”外不会向调用方抛出异常）
        logger.warning(
            f"无法将环境变量值 {val} 转换为 {cast.__name__}，使用默认值 {default}"
        )
        return default


def _get_env_list(key: str, default: list[str]) -> list[str]:
    r"""读取字符串列表型环境变量（用于分隔符等配置）.

    支持两种 ``.env`` 写法：

    - 逗号分隔（推荐）：``SEPARATORS=\n\n,\n, ,``。字面 ``\n`` 会被
      还原为换行符；**空项保留**，因此结尾逗号可表达空字符串分隔符 ``""``；
      不对每段做 strip，以保留空格分隔符 ``" "``。
    - JSON 数组（兼容）：``SEPARATORS=["\n\n", "\n", " "]``。

    Args:
        key: 环境变量名称。
        default: 环境变量缺失或为空时返回的默认列表。

    Returns:
        解析后的字符串列表；JSON 格式非法或元素不全是字符串时记录警告并返回默认值。
    """
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()

    # 兼容 JSON 数组写法（最直觉但易写错的形式）
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"环境变量 {key} 不是合法 JSON 数组，使用默认值：{raw}")
            return default
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            logger.warning(
                f"环境变量 {key} 的 JSON 必须是字符串数组，使用默认值：{raw}"
            )
            return default
        return list(parsed)

    return [part.replace("\\n", "\n") for part in raw.split(",")]


# 运行时热更新覆盖：优先级高于类属性（即导入时从 .env/环境读到的值）。
# 网页「调试设置」保存后写入此处，无需重启进程即可让后续节点读到新值；
# 同时会持久化回 .env，使重启后仍然生效。
_RUNTIME_OVERRIDES: dict[str, Any] = {}


class _HotReloadMeta(type):
    """让 ``Config.XXX`` 优先返回运行时覆盖值，未覆盖时回退类属性."""

    def __getattribute__(cls, name: str) -> Any:
        # 直接查模块级 dict（不走属性访问，避免递归）
        if name in _RUNTIME_OVERRIDES:
            return _RUNTIME_OVERRIDES[name]
        return super().__getattribute__(name)


@final
class Config(metaclass=_HotReloadMeta):
    """Agent 全局配置容器（不允许继承）.

    类属性为导入时从 .env / 环境变量读取的默认值；
    ``HOT_RELOADABLE_KEYS`` 中的字段可被 :func:`apply_runtime_overrides`
    在运行时覆盖（热更新），读取时覆盖值优先。
    """

    # 大模型配置
    DEFAULT_LLM_MODEL = _get_env("DEFAULT_LLM_MODEL", str, "deepseek-chat")
    DEFAULT_LLM_API_KEY = _get_env("DEFAULT_LLM_API_KEY", str, None)
    DEFAULT_LLM_PROVIDER = _get_env("DEFAULT_LLM_PROVIDER", str, "deepseek")
    DEFAULT_LLM_TEMPERATURE = _get_env("DEFAULT_LLM_TEMPERATURE", float, 0.0)
    # OpenAI 兼容自定义网关地址：仅 openai/qwen 等兼容协议使用，
    # 留空时按 provider 取内置端点（deepseek/qwen）；deepseek 官方忽略此项
    DEFAULT_LLM_BASE_URL = _get_env("DEFAULT_LLM_BASE_URL", str, None)

    # 嵌入模型配置
    EMBEDDING_PROVIDER = _get_env("EMBEDDING_PROVIDER", str, "bge")
    EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", str, "")
    EMBEDDING_API_KEY = _get_env("EMBEDDING_API_KEY", str, None)
    EMBEDDING_BASE_URL = _get_env("EMBEDDING_BASE_URL", str, None)
    # 本地 BGE 编码批大小：查询通常只编码 1~6 条不受影响，
    # 入库大批量可显著提高 CPU 吞吐，内存紧张时调小
    EMBEDDING_BATCH_SIZE = _get_env("EMBEDDING_BATCH_SIZE", int, 16)

    # 文档切分配置
    CHUNK_SIZE = _get_env("CHUNK_SIZE", int, 1500)
    CHUNK_OVERLAP = _get_env("CHUNK_OVERLAP", int, 150)
    CODE_MAX_CHARS = _get_env("CODE_MAX_CHARS", int, 3000)
    SEPARATORS = _get_env_list("SEPARATORS", ["\n\n", "\n", " ", ""])

    # 知识库源文档目录：网页知识库管理与 ingestion sync 的根目录。
    # 相对路径按项目根 ROOT_DIR 解析；向量库是该目录的派生索引。
    KNOWLEDGE_DOCS_DIR = _get_env("KNOWLEDGE_DOCS_DIR", str, "docs/knowledge_base")

    # 向量数据库配置
    VECTORSTORE_PROVIDER = _get_env("VECTORSTORE_PROVIDER", str, "chroma")
    VECTORSTORE_PATH = _get_env("VECTORSTORE_PATH", str, "./chroma_db")
    VECTORSTORE_POSTGRES_HOST = _get_env("VECTORSTORE_POSTGRES_HOST", str, "localhost")
    VECTORSTORE_POSTGRES_PORT = _get_env("VECTORSTORE_POSTGRES_PORT", int, 5432)
    VECTORSTORE_POSTGRES_USER = _get_env("VECTORSTORE_POSTGRES_USER", str, "postgres")
    VECTORSTORE_POSTGRES_PASSWORD = _get_env(
        "VECTORSTORE_POSTGRES_PASSWORD", str, "postgres"
    )
    VECTORSTORE_POSTGRES_DB = _get_env("VECTORSTORE_POSTGRES_DB", str, "postgres")
    VECTORSTORE_COLLECTION_NAME = _get_env(
        "VECTORSTORE_COLLECTION_NAME", str, "rag_agent"
    )

    # 检索配置（双路召回：向量 + BM25）
    RETRIEVAL_K = _get_env("RETRIEVAL_K", int, 3)  # 向量召回条数
    BM25_K = _get_env("BM25_K", int, 5)  # BM25 召回条数

    # Rerank 重排配置（召回合并后精排取 top-k；支持本地 cross-encoder 与云端 API）
    RERANK_ENABLED = _get_env("RERANK_ENABLED", bool, True)
    # 提供方：local（本地模型）/ siliconflow（硅基流动，bge-reranker 免费档）/
    # dashscope（阿里云百炼 qwen3-rerank，兼容协议）
    RERANK_PROVIDER = _get_env("RERANK_PROVIDER", str, "local")
    # 模型名：local 时为 HF 模型名/本地路径；云端时为 API 模型名
    # （硅基流动的 BAAI/bge-reranker-v2-m3 与本地同名，切换 provider 无需改此项）
    RERANK_MODEL = _get_env("RERANK_MODEL", str, "BAAI/bge-reranker-v2-m3")
    RERANK_DEVICE = _get_env("RERANK_DEVICE", str, "cpu")  # 仅 local 使用
    RERANK_TOP_K = _get_env("RERANK_TOP_K", int, 3)  # 重排后送入生成的文档数
    # 重排熔断阈值：开启重排时，若最高分仍低于该值，判定知识库未命中、
    # 清空候选走「未找到」回复，避免「矮子里拔将军」硬答。None=关闭；
    # bge-reranker 输出是未归一化 logits，阈值需按真实流量的分数分布标定，
    # 不能跨模型照搬
    RERANK_SCORE_THRESHOLD = _get_env("RERANK_SCORE_THRESHOLD", float, None)
    # 邻域扩展窗口：命中 chunk 前后各补多少片同文档兄弟块进入生成上下文
    # （小块检索、带语境喂给生成）；0=关闭；仅 Chroma 后端生效
    RERANK_CONTEXT_WINDOW = _get_env("RERANK_CONTEXT_WINDOW", int, 1)
    # 云端 API 配置（provider=local 时忽略；BASE_URL 留空则按 provider 取默认端点）
    RERANK_API_KEY = _get_env("RERANK_API_KEY", str, None)
    RERANK_BASE_URL = _get_env("RERANK_BASE_URL", str, None)
    RERANK_API_TIMEOUT = _get_env("RERANK_API_TIMEOUT", float, 30.0)
    # 云端失败重试次数（仅网络异常/429/5xx，指数退避；0 关闭重试）
    RERANK_MAX_RETRIES = _get_env("RERANK_MAX_RETRIES", int, 2)

    # 对话历史压缩配置：条数或估算 token 任一超阈值即触发
    SUMMARY_TRIGGER_TOKENS = _get_env("SUMMARY_TRIGGER_TOKENS", int, 4000)
    SUMMARY_TRIGGER_MESSAGES = _get_env("SUMMARY_TRIGGER_MESSAGES", int, 10)
    # 触发后按问答对切分的压缩比例：旧的 ratio 进摘要，其余原样保留（0.5=一半）
    SUMMARY_COMPRESS_RATIO = _get_env("SUMMARY_COMPRESS_RATIO", float, 0.5)
    # 滚动摘要的长度上限（字数与条数），由摘要节点注入 prompt
    SUMMARY_MAX_CHARS = _get_env("SUMMARY_MAX_CHARS", int, 300)
    SUMMARY_MAX_ITEMS = _get_env("SUMMARY_MAX_ITEMS", int, 8)
    # 重构检索语句时携带的最近历史消息条数（改写只需近期语境，旧历史
    # 可能干扰指代消解，故此处独立截断；最终回答不截断，由 summarize 兜底）
    REBUILD_RETRIEVAL_K = _get_env("REBUILD_RETRIEVAL_K", int, 6)

    # Checkpointer 配置
    CHECKPOINTER_BACKEND = _get_env("CHECKPOINTER_BACKEND", str, "memory")
    CP_REDIS_URL = _get_env("CP_REDIS_URL", str, "redis://localhost:6379/0")
    CP_POSTGRES_URL = _get_env(
        "CP_POSTGRES_URL",
        str,
        "postgresql://postgres:postgres@localhost:5432/langgraph",
    )


# ---------------------------------------------------------------------------
# 热更新（网页「调试设置」）
# ---------------------------------------------------------------------------
# 允许在运行时调整的字段及其表单 schema。type 决定入参转换与前端控件：
#   str / secret / int / float / bool / select
# secret 字段 GET 时不回显明文，POST 传空串表示「不修改」。
HOT_FIELDS: dict[str, dict[str, Any]] = {
    # —— 对话模型 ——
    "DEFAULT_LLM_PROVIDER": {
        "group": "对话模型",
        "label": "服务商",
        "type": "select",
        "options": ["deepseek", "openai", "qwen"],
        "help": "deepseek/qwen 走 OpenAI 兼容协议",
    },
    "DEFAULT_LLM_MODEL": {
        "group": "对话模型",
        "label": "模型名",
        "type": "str",
        "help": "如 deepseek-chat、qwen-plus、gpt-4o-mini",
    },
    "DEFAULT_LLM_API_KEY": {
        "group": "对话模型",
        "label": "API Key",
        "type": "secret",
    },
    "DEFAULT_LLM_BASE_URL": {
        "group": "对话模型",
        "label": "网关地址 BASE_URL",
        "type": "str",
        "optional": True,
        "help": "OpenAI 兼容自定义网关；留空用服务商内置端点",
    },
    "DEFAULT_LLM_TEMPERATURE": {
        "group": "对话模型",
        "label": "Temperature",
        "type": "float",
        "help": "0~2，RAG 建议 0",
    },
    # —— 检索与重排 ——
    "RETRIEVAL_K": {
        "group": "检索与重排",
        "label": "向量召回条数",
        "type": "int",
    },
    "BM25_K": {
        "group": "检索与重排",
        "label": "BM25 召回条数",
        "type": "int",
    },
    "RERANK_ENABLED": {
        "group": "检索与重排",
        "label": "启用重排",
        "type": "bool",
    },
    "RERANK_PROVIDER": {
        "group": "检索与重排",
        "label": "重排提供方",
        "type": "select",
        "options": ["local", "siliconflow", "dashscope"],
    },
    "RERANK_MODEL": {
        "group": "检索与重排",
        "label": "重排模型名",
        "type": "str",
    },
    "RERANK_TOP_K": {
        "group": "检索与重排",
        "label": "重排后保留条数",
        "type": "int",
    },
    "RERANK_SCORE_THRESHOLD": {
        "group": "检索与重排",
        "label": "重排熔断阈值",
        "type": "float",
        "optional": True,
        "help": "最高分低于此值则判定未命中、直接答未找到；留空关闭。按真实分数分布标定",
    },
    "RERANK_CONTEXT_WINDOW": {
        "group": "检索与重排",
        "label": "邻域扩展窗口",
        "type": "int",
        "help": "命中片段前后各补几片同文档内容（防碎片化丢语境），0=关闭",
    },
    "RERANK_API_KEY": {
        "group": "检索与重排",
        "label": "重排 API Key",
        "type": "secret",
        "optional": True,
        "help": "云端重排需要，local 本地模型留空",
    },
    "RERANK_BASE_URL": {
        "group": "检索与重排",
        "label": "重排网关地址",
        "type": "str",
        "optional": True,
    },
    # —— 嵌入模型 ——
    "EMBEDDING_PROVIDER": {
        "group": "嵌入模型",
        "label": "提供方",
        "type": "select",
        "options": ["bge", "openai", "dashscope"],
        "help": "切换后向量维度可能变化，需与知识库构建时一致",
    },
    "EMBEDDING_MODEL": {
        "group": "嵌入模型",
        "label": "嵌入模型名",
        "type": "str",
        "optional": True,
        "help": "bge 本地可留空用默认 BAAI/bge-small-zh-v1.5",
    },
    "EMBEDDING_API_KEY": {
        "group": "嵌入模型",
        "label": "嵌入 API Key",
        "type": "secret",
        "optional": True,
    },
    "EMBEDDING_BASE_URL": {
        "group": "嵌入模型",
        "label": "嵌入网关地址",
        "type": "str",
        "optional": True,
    },
    # —— 历史压缩 ——
    "SUMMARY_TRIGGER_MESSAGES": {
        "group": "历史压缩",
        "label": "触发压缩的消息条数",
        "type": "int",
    },
    "SUMMARY_TRIGGER_TOKENS": {
        "group": "历史压缩",
        "label": "触发压缩的 token 数",
        "type": "int",
    },
    "SUMMARY_COMPRESS_RATIO": {
        "group": "历史压缩",
        "label": "压缩比例",
        "type": "float",
        "help": "0~1，触发后旧的该比例历史进摘要（0.5=一半）",
    },
    "REBUILD_RETRIEVAL_K": {
        "group": "历史压缩",
        "label": "改写时携带历史条数",
        "type": "int",
    },
}

_CASTS: dict[str, Any] = {
    "str": str,
    "secret": str,
    "int": int,
    "float": float,
}


class ConfigError(ValueError):
    """热更新配置校验失败（收集所有字段错误后一次性抛出）."""


def _mask_secret(value: str | None) -> str:
    """密钥脱敏：仅展示前 3 位与后 2 位."""
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}…{value[-2:]}"


def get_hot_config() -> dict[str, Any]:
    """返回热更新字段的当前有效值与表单 schema（密钥脱敏）."""
    fields = []
    for key, spec in HOT_FIELDS.items():
        value = getattr(Config, key)
        item = {
            "key": key,
            "group": spec["group"],
            "label": spec["label"],
            "type": spec["type"],
            "help": spec.get("help", ""),
            "optional": spec.get("optional", False),
            "options": spec.get("options"),
        }
        if spec["type"] == "secret":
            item["secret_set"] = bool(value)
            item["value"] = _mask_secret(value)
        else:
            item["value"] = value
        fields.append(item)
    return {"fields": fields}


def _coerce(key: str, raw: str) -> Any:
    """把表单字符串按字段类型转换；bool 与空值特判."""
    spec = HOT_FIELDS[key]
    ftype = spec["type"]

    if raw.strip() == "":
        if spec.get("optional") or ftype == "secret":
            return None
        raise ConfigError(f"{spec['label']} 不能为空")

    if ftype == "bool":
        low = raw.strip().lower()
        if low in ("true", "1", "on", "yes"):
            return True
        if low in ("false", "0", "off", "no"):
            return False
        raise ConfigError(f"{spec['label']} 必须是 true/false")

    if ftype == "select":
        options = spec["options"]
        val = raw.strip()
        if val not in options:
            raise ConfigError(f"{spec['label']} 必须是 {options} 之一")
        return val

    cast = _CASTS[ftype]
    try:
        return cast(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{spec['label']} 需要 {ftype} 类型，收到 {raw!r}") from exc


def _persist_env(updates: dict[str, Any]) -> None:
    """把更新按 ``KEY=VALUE`` 行 upsert 回 .env，其余行原样保留."""
    path = _env_path
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def _serialize(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    pending = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in line:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in pending:
            lines[i] = f"{key}={_serialize(pending.pop(key))}"

    if pending:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# 网页调试设置写入（热更新）")
        for key, value in pending.items():
            lines.append(f"{key}={_serialize(value)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_runtime_overrides(raw_values: dict[str, str]) -> list[str]:
    """校验并应用一批热更新：写入运行时覆盖 → 同步 os.environ → 持久化 .env.

    Args:
        raw_values: 表单原始字符串值。secret 字段传空串表示不修改；
            可选字符串字段传空串表示置空（None）。

    Returns:
        实际生效的字段名列表。

    Raises:
        ConfigError: 存在未知字段或类型校验失败时抛出，且不会应用任何更新。
    """
    unknown = sorted(set(raw_values) - set(HOT_FIELDS))
    if unknown:
        raise ConfigError(f"未知配置项：{', '.join(unknown)}")

    # 先全部转换，任一失败则整体不生效
    coerced: dict[str, Any] = {}
    errors: list[str] = []
    for key, raw in raw_values.items():
        try:
            coerced[key] = _coerce(key, raw)
        except ConfigError as exc:
            errors.append(str(exc))
    if errors:
        raise ConfigError("；".join(errors))

    applied = [
        k
        for k, v in coerced.items()
        if not (HOT_FIELDS[k]["type"] == "secret" and v is None)
    ]

    # 1) 运行时覆盖（secret 的 None=不修改，直接跳过）
    for key, value in coerced.items():
        if HOT_FIELDS[key]["type"] == "secret" and value is None:
            continue
        _RUNTIME_OVERRIDES[key] = value
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = (
                "true" if value is True else "false" if value is False else str(value)
            )

    # 2) 持久化（同样跳过未修改的密钥）
    _persist_env({k: coerced[k] for k in applied})
    logger.info("热更新配置已生效：%s", ", ".join(applied))
    return applied
