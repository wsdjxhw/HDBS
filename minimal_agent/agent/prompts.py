"""System prompt 构建:把工具注册表渲染进 system prompt,并约定输出协议。"""
from agent.tools import ToolRegistry

# 格式修复指令:非法输出塞回 context,让 LLM 自己重出(有次数上限)
REPAIR_PROMPT = (
    "你上一轮的输出不符合协议(不是合法 JSON 或缺少必要字段)。"
    '请严格只输出一个 JSON 对象:调用工具用 {"thought":"...","action":{"name":"...","args":{...}}},'
    '给出最终答案用 {"thought":"...","answer":"..."}。不要输出任何其他文字。'
)


def build_system_prompt(registry: ToolRegistry) -> str:
    return f"""你是一个能调用工具的助手。

# 可用工具
{registry.schemas_text()}

# 输出协议(必须严格遵守)
每次回复只能是一个 JSON 对象,不要输出 JSON 以外的任何文字,不要使用 markdown 代码块。两种格式二选一:
1. 需要调用工具:
{{"thought": "简述你的思考", "action": {{"name": "工具名", "args": {{...}}}}}}
2. 已有足够信息,给出最终答案:
{{"thought": "简述你的思考", "answer": "给用户看的最终回答"}}

# 规则
- action.name 必须是"可用工具"列表中存在的名字,args 必须符合该工具的参数说明。
- 工具的执行结果会在下一轮以 [工具结果] 开头的消息提供。
- 信息足够时立即用 answer 回答,不要重复调用同一个工具。
- 请用中文回答。"""
