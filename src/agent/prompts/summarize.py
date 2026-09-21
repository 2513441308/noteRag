"""对话压缩节点的提示词模板.

基于已有摘要与新发生的对话做增量合并，生成滚动摘要。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# 输入变量：summary（已有摘要，首次为"（无）"）、dialogue（待压缩的旧对话文本）、
# max_chars（摘要字数上限）、max_items（摘要条数上限），上限由节点从 Config 注入。
# 期望输出：合并后的摘要正文
SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是对话压缩器。请把「已有摘要」和「新发生的对话」合并，生成一份更新后的"
            "中文摘要，供后续对话作为长期上下文使用。\n"
            "要求：\n"
            "- 增量合并：已有摘要中的有效信息必须保留，不得丢失；\n"
            "- 重点保留：关键实体（文件名、接口、配置名等）、用户的约束与偏好、"
            "已确认的结论、尚未解决的问题；\n"
            "- 忽略寒暄、重复与无关细节；\n"
            "- 摘要紧凑、分条陈述，不得编造对话中不存在的内容；\n"
            # 滚动摘要必须有长度上限，否则多轮对话后会无限膨胀挤占上下文
            "- 摘要总长控制在 {max_chars} 字以内、不超过 {max_items} 条；"
            "接近上限时合并同类条目，"
            "优先保留近期对话与尚未解决的问题，丢弃过时细节；\n"
            "- 只输出摘要正文，不要标题、不要解释。",
        ),
        (
            "human",
            "已有摘要：\n{summary}\n\n新发生的对话：\n{dialogue}",
        ),
    ]
)
