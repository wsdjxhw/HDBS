"""LLM 输出解析器。

协议约定(每轮输出二选一):
    {"thought": "...", "action": {"name": "weather", "args": {"city": "广州"}}}
    {"thought": "...", "answer": "最终答案文本"}

判定规则:有 answer 字段 -> 最终答案;有 action 字段 -> 工具调用;其他 -> 非法。
容错策略:只取第一个 { 到最后一个 } 之间的内容再 json.loads,
代码块围栏(```json)、前后废话一次解决;解析失败返回 invalid,由上层决定重试。
"""
import json
from dataclasses import dataclass

ACTION = "action"    # 这轮要调用工具
ANSWER = "answer"    # 这是最终答案,循环结束
INVALID = "invalid"  # 格式坏了,需要重试或兜底


@dataclass
class ParseResult:
    type: str                    # ACTION / ANSWER / INVALID
    thought: str = ""            # LLM 的思考过程,trace 日志素材
    action: dict | None = None   # {"name": ..., "args": {...}}
    answer: str | None = None
    raw: str = ""                # 原始输出,写日志用


def _extract_json(text: str) -> str | None:
    """切出第一个 { 到最后一个 } 之间的一段;找不到返回 None。"""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def parse_llm_output(text: str) -> ParseResult:
    result = ParseResult(type=INVALID, raw=text)

    snippet = _extract_json(text)
    if snippet is None:
        return result
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return result
    if not isinstance(data, dict):
        return result

    result.thought = str(data.get("thought", ""))

    # answer 优先:两个都有时按最终答案处理,别让循环多转
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        result.type = ANSWER
        result.answer = answer
        return result

    action = data.get("action")
    if isinstance(action, dict) and isinstance(action.get("name"), str) and action["name"].strip():
        result.type = ACTION
        # args 缺省时容忍为 {},协议对 LLM 友好一点
        result.action = {"name": action["name"].strip(), "args": action.get("args") or {}}
        return result

    return result
