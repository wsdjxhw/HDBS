"""Session 与 context 管理。

- Session: 一个会话窗口的全部历史(user / assistant / tool),多窗口 = 多 Session,天然隔离
- SessionManager: Session 的落盘与加载(sessions/*.json),进程重启也能接着聊
- compress_if_needed: 结构感知压缩(见函数 docstring)
- build_llm_messages: 组装最终发给 LLM 的消息(system + 压缩后历史)

压缩的设计原则(评审教训):
    压缩的对象是"结构化信息",不是"字符串"。历史里塞进任何残缺 JSON,
    都会污染 LLM 的 few-shot 示例(它模仿自己历史里的格式),并破坏其推理依据。
    因此:工具结果按 JSON 字段截断后**重新序列化**,进 context 的永远是合法 JSON。
"""
import json
import os
import re
import time
from dataclasses import dataclass, field, replace

USER, ASSISTANT, TOOL = "user", "assistant", "tool"

DEFAULT_MAX_CONTEXT_CHARS = 8000   # 发给 LLM 的历史文本预算(字符数)


@dataclass
class Message:
    role: str
    content: str
    meta: dict = field(default_factory=dict)   # 如 tool 消息记录工具名/执行结果,trace 用


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.messages: list[Message] = []
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at

    def append(self, role: str, content: str, **meta) -> Message:
        self.messages.append(Message(role, content, meta))
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return self.messages[-1]


class SessionManager:
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    @staticmethod
    def _safe_id(session_id: str) -> str:
        return re.sub(r"[^\w.-]", "_", session_id)

    def load(self, session_id: str) -> Session:
        path = os.path.join(self.base_dir, self._safe_id(session_id) + ".json")
        if not os.path.exists(path):
            return Session(session_id)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        session = Session(session_id)
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.messages = [
            Message(m["role"], m["content"], m.get("meta", {})) for m in data.get("messages", [])
        ]
        return session

    def save(self, session: Session):
        session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        path = os.path.join(self.base_dir, self._safe_id(session.id) + ".json")
        payload = {
            "id": session.id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [
                {"role": m.role, "content": m.content, "meta": m.meta} for m in session.messages
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _shrink_tool_content(content: str, per_value_limit: int = 60, max_keys: int = 4) -> str:
    """压缩工具结果:按 JSON 结构压缩,输出仍是合法 JSON;原文非 JSON 才退化为纯文本标注。"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return f"[已压缩] {_clip(content, 150)}"
    if isinstance(data, dict):
        kept = {}
        for i, (k, v) in enumerate(data.items()):
            if i >= max_keys:
                break
            kept[k] = _clip(v, per_value_limit) if isinstance(v, str) else v
        kept["_compressed"] = True
        return json.dumps(kept, ensure_ascii=False)
    return f"[已压缩] {_clip(content, 150)}"


def _shrink_assistant_content(content: str) -> str:
    """压缩 assistant 历史:它作为 few-shot 示例的价值在"格式正确",内容可丢。
    解析成功 -> 只留 thought 与动作标记(仍是合法协议 JSON);失败 -> 纯文本标注。"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return f"[已压缩] {_clip(content, 120)}"
    if isinstance(data, dict):
        kept = {"thought": _clip(str(data.get("thought", "")), 100), "_compressed": True}
        if "answer" in data:
            kept["answer"] = _clip(str(data["answer"]), 100)
        elif isinstance(data.get("action"), dict):
            kept["action"] = {"name": data["action"].get("name", "")}
        return json.dumps(kept, ensure_ascii=False)
    return f"[已压缩] {_clip(content, 120)}"


def _digest(messages: list[Message], limit: int = 800) -> str:
    """被丢弃轮次的规则式一行摘要:摘要层(旧)+ 原文层(新)的基础分层记忆。
    limit 是整条消息(含前缀)的总长上限,保证插回后不超预算。"""
    prefix = "[历史摘要] "
    lines = []
    for m in messages:
        if m.role == USER:
            lines.append(f"用户说:{_clip(m.content, 50)}")
        elif m.role == TOOL:
            lines.append(f"工具{m.meta.get('name', '?')}:{_clip(m.content, 40)}")
        else:
            lines.append(f"助手:{_clip(m.content, 40)}")
    return prefix + _clip(" | ".join(lines), max(0, limit - len(prefix) - 1))  # -1 给省略号


def compress_if_needed(messages: list[Message], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> list[Message]:
    """结构感知压缩,三步走,不动原对象(返回新列表,session 里保留完整历史):

    1. 从旧到新压缩 TOOL 消息(工具结果是最占空间、时效性最弱的)
    2. 从旧到新压缩 ASSISTANT 消息
    3. 仍超预算:从第二条起丢弃最旧轮次(第一条 user 通常是本次会话目标,必须保住),
       被丢弃的轮次以一行摘要替代,而不是无声消失
    """
    msgs = [replace(m, meta=dict(m.meta)) for m in messages]   # 深拷贝一层,绝不污染 session

    def total() -> int:
        return sum(len(m.content) for m in msgs)

    if total() <= max_chars:
        return msgs

    for m in msgs:  # step 1
        if total() <= max_chars:
            break
        if m.role == TOOL and not m.meta.get("shrunk"):
            m.content = _shrink_tool_content(m.content)
            m.meta["shrunk"] = True

    for m in msgs:  # step 2
        if total() <= max_chars:
            break
        if m.role == ASSISTANT and not m.meta.get("shrunk"):
            m.content = _shrink_assistant_content(m.content)
            m.meta["shrunk"] = True

    dropped = []
    digest_headroom = 200   # 给摘要消息预留的预算:丢弃到(预算-200)为止,插回摘要后才不超
    while total() > max_chars - digest_headroom and len(msgs) > 3:  # step 3
        dropped.append(msgs.pop(1))
    if dropped:
        remaining = max(0, max_chars - total())               # 摘要还能用多少预算
        msgs.insert(1, Message(USER, _digest(dropped, limit=min(800, remaining))))
    return msgs


def build_llm_messages(system_prompt: str, messages: list[Message], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> list[dict]:
    """system prompt + 压缩后的历史。

    tool 消息转成 user 角色并加 [工具结果] 前缀:本协议不使用原生 function calling,
    这样写对所有 OpenAI 兼容端点都合法,且 system prompt 里已向 LLM 说明该约定。
    """
    out = [{"role": "system", "content": system_prompt}]
    for m in compress_if_needed(messages, max_chars):
        if m.role == TOOL:
            out.append({"role": "user", "content": f"[工具结果] {m.content}"})
        else:
            out.append({"role": m.role, "content": m.content})
    return out
