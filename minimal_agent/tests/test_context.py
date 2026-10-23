"""session 与 context 管理测试:多窗口隔离、结构感知压缩、落盘恢复。"""
import json
import time

from agent.context import (
    ASSISTANT,
    TOOL,
    USER,
    Session,
    SessionManager,
    build_llm_messages,
    compress_if_needed,
)


def test_session_isolation():
    """两个窗口两个 session,历史互不污染。"""
    s1, s2 = Session("window1"), Session("window2")
    s1.append(USER, "帮我查天气记待办")
    s2.append(USER, "帮我写周报")
    assert len(s1.messages) == 1 and len(s2.messages) == 1
    assert not any("周报" in m.content for m in s1.messages)


def test_compress_truncates_and_keeps_first_user():
    s = Session("c")
    s.append(USER, "这次会话的目标:查天气并记待办")           # 会话目标,压缩必须保住
    for i in range(20):
        s.append(TOOL, "x" * 500, name="search")
        s.append(ASSISTANT, '{"thought": "...", "action": {"name": "search", "args": {}}}')

    out = compress_if_needed(s.messages, max_chars=4000)
    assert sum(len(m.content) for m in out) <= 4000
    assert "会话的目标" in out[0].content                     # 第一条 user 保住了
    assert any("[已压缩]" in m.content for m in out)          # 非JSON内容被压缩时有标注
    assert any("[历史摘要]" in m.content for m in out)        # 被丢弃的轮次有摘要,不是无声消失


def test_compress_tool_output_stays_valid_json():
    """核心回归:JSON 工具结果压缩后仍是合法 JSON,绝不出残缺片段。"""
    s = Session("c")
    s.append(USER, "查天气")
    big = json.dumps({"city": "广州", "weather": "晴", "detail": "z" * 1000}, ensure_ascii=False)
    s.append(TOOL, big, name="weather")
    out = compress_if_needed(s.messages, max_chars=300)
    tool = [m for m in out if m.role == "tool"][0]
    data = json.loads(tool.content)                           # 能被解析 => 没有被剁残
    assert data["_compressed"] is True
    assert len(data["detail"]) <= 61                          # 超长字段被截短


def test_compress_assistant_keeps_protocol_shape():
    """assistant 历史压缩后仍是合法协议 JSON,只保留 thought/answer 摘要。"""
    s = Session("c")
    s.append(USER, "问")
    long_reply = json.dumps({"thought": "t" * 400, "answer": "a" * 400}, ensure_ascii=False)
    s.append(ASSISTANT, long_reply)
    out = compress_if_needed(s.messages, max_chars=200)
    asst = [m for m in out if m.role == "assistant"][0]
    data = json.loads(asst.content)
    assert "thought" in data
    assert len(data.get("answer", "")) <= 101


def test_compress_does_not_mutate_session():
    s = Session("c")
    s.append(TOOL, "y" * 2000)
    before_content, before_meta = len(s.messages[0].content), dict(s.messages[0].meta)
    compress_if_needed(s.messages, max_chars=100)
    assert len(s.messages[0].content) == before_content       # session 里保留完整原文
    assert s.messages[0].meta == before_meta                  # meta 也不被污染


def test_compress_noop_when_short():
    s = Session("c")
    s.append(USER, "hi")
    out = compress_if_needed(s.messages, max_chars=8000)
    assert out == s.messages


def test_build_llm_messages_shape():
    s = Session("c")
    s.append(USER, "广州天气?")
    s.append(TOOL, '{"city": "广州"}', name="weather")
    msgs = build_llm_messages("SYSTEM_PROMPT", s.messages)
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYSTEM_PROMPT"
    assert msgs[1] == {"role": "user", "content": "广州天气?"}
    assert msgs[2]["content"].startswith("[工具结果]")         # tool -> user 加标记
    assert msgs[2]["role"] == "user"


def test_session_persistence_roundtrip(tmp_path):
    sm = SessionManager(base_dir=str(tmp_path))
    s = sm.load("window1")
    s.append(USER, "帮我查天气记待办")
    s.append(ASSISTANT, '{"answer": "好的"}')
    sm.save(s)

    time.sleep(0.05)  # 确保 updated_at 有变化意义,防止同秒写入
    s2 = sm.load("window1")
    assert s2.id == "window1"
    assert [m.content for m in s2.messages] == [m.content for m in s.messages]

    fresh = sm.load("window-other")
    assert len(fresh.messages) == 0                            # 不存在的 session 返回新会话
