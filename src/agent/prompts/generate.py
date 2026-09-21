"""最终回答节点（RAG 生成）的提示词模板.

严格依据检索文档生成带来源编号标注的中文回答。历史对话以原生消息列表
经 ``MessagesPlaceholder`` 直接注入（保留角色结构），不拍平为文本。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 输入变量：
#   summary  - 历史会话摘要（可能为占位符）
#   context  - 按来源编号后的检索文档文本
#   history  - 已完成轮次的消息列表（Human/AI 交替，可为空）
#   question - 本轮用户原始问题
# 期望输出：带来源编号标注的中文回答
GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识库问答助手。请严格依据「检索到的文档」回答用户问题。\n\n"
            "回答规则：\n"
            "1. 只能使用文档中可核实的信息，不得依赖文档之外的内容进行猜测；\n"
            "2. 文档以 [n] 编号（n 从 1 开始连续编号），在引用其内容的句子末尾"
            "标注来源编号，如 [1][2]；\n"
            "   只能使用本次上下文中实际出现的编号，严禁编造未出现的编号"
            "（例如只有 [1] 时绝不允许出现 [2]）；未被任何编号覆盖的内容"
            "（如「无来源」分组）不得标注编号；\n"
            "3. 文档不足以回答时，直接说明知识库中没有找到相关内容，"
            "可建议用户补充信息，不要编造；\n"
            "4. 涉及代码时给出可运行的片段并简要说明；\n"
            "5. 多篇文档内容相互矛盾时，指出分歧所在而不是随意取舍；\n"
            "6. 不要在回答末尾罗列参考资料、文件名或链接清单，"
            "来源信息由系统通过引用编号单独提供；\n"
            "7. 使用与用户一致的语言（中文），条理清晰。",
        ),
        (
            "system",
            "历史会话摘要（可能为空）：\n{summary}\n\n检索到的文档：\n{context}",
        ),
        # 已完成的历史轮次；首轮为空，optional 允许缺省
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{question}"),
    ]
)
