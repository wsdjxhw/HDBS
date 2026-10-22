"""Session 与 context 管理。

- Session: 一个会话窗口的全部历史(user / assistant / tool),多窗口 = 多 Session,天然隔离
- SessionManager: Session 的落盘与加载(sessions/*.json),进程重启也能接着聊
- compress_if_needed: 基础压缩(题目只要求基础的,复杂的未实现)
- build_llm_messages: 组装最终发给 LLM 的消息(system + 压缩后历史)
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
    meta: dict = field(default_factory=dict)   # 如 tool 消息记录工具名,trace 用


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


def compress_if_needed(messages: list[Message], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> list[Message]:
    """基础压缩,三步走,不动原对象(返回新列表,session 里保留完整历史):

    1. 从旧到新截断 TOOL 消息内容(工具结果是最占空间、时效性最弱的)
    2. 从旧到新截断 ASSISTANT 消息内容(协议原文留个头即可)
    3. 从第二条起丢最旧的消息(第一条 user 通常是本次会话目标,必须保住)
    """
    msgs = [replace(m) for m in messages]

    def total() -> int:
        return sum(len(m.content) for m in msgs)

    if total() <= max_chars:
        return msgs

    for m in msgs:  # step 1
        if total() <= max_chars:
            break
        if m.role == TOOL and len(m.content) > 200:
            m.content = m.content[:200] + "…[已压缩]"

    for m in msgs:  # step 2
        if total() <= max_chars:
            break
        if m.role == ASSISTANT and len(m.content) > 300:
            m.content = m.content[:300] + "…[已压缩]"

    dropped = 0
    while total() > max_chars and len(msgs) > 3:  # step 3
        msgs.pop(1)
        dropped += 1
    if dropped:
        msgs.insert(1, Message(USER, f"[系统提示: 为控制上下文长度,已省略 {dropped} 条较早的对话]"))
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
