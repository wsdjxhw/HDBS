"""runtime 测试:用 FakeLLM 跑完整循环,不依赖真实 API。"""
import json

from agent.context import Session
from agent.runtime import Agent
from agent.tools import default_registry

ACTION_JSON = json.dumps(
    {"thought": "需要查天气", "action": {"name": "weather", "args": {"city": "广州"}}},
    ensure_ascii=False,
)
ANSWER_JSON = json.dumps({"thought": "信息够了", "answer": "广州今天晴。"}, ensure_ascii=False)


class FakeLLM:
    """按剧本回复的假模型,并记录每次实际发出的 messages。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages):
        self.sent.append(messages)
        return self.replies.pop(0)


def make_agent(replies, session_id="t"):
    llm = FakeLLM(replies)
    return Agent(default_registry(), llm, Session(session_id)), llm


def test_tool_then_answer():
    """主流程:调工具 -> 结果塞回 -> 最终答案。"""
    agent, _ = make_agent([ACTION_JSON, ANSWER_JSON])
    out = agent.run("广州天气怎么样?")
    assert out == "广州今天晴。"
    roles = [m.role for m in agent.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.session.messages[2].meta["name"] == "weather"   # trace 信息在


def test_followup_sees_history():
    """追问(纯对话):第二轮请求里带着第一轮的完整历史。"""
    agent, llm = make_agent([ACTION_JSON, ANSWER_JSON, ANSWER_JSON])
    agent.run("广州天气?")
    out = agent.run("那要带伞吗?")
    assert out == "广州今天晴。"
    second_run = llm.sent[2]
    assert any("[工具结果]" in m["content"] for m in second_run)  # 带着工具结果的追问
    assert any("带伞" in m["content"] for m in second_run)


def test_invalid_then_repair_then_answer():
    """格式坏了 -> 修复指令塞回 -> 下一轮恢复正常。"""
    agent, _ = make_agent(["这不是JSON", ANSWER_JSON])
    out = agent.run("hi")
    assert out == "广州今天晴。"
    assert any("不符合协议" in m.content for m in agent.session.messages)


def test_invalid_exceeds_repairs_fallback():
    """修复超限 -> 兜底返回最后一轮原文,程序不崩。"""
    agent, _ = make_agent(["垃圾1", "垃圾2", "垃圾3"])
    out = agent.run("hi")
    assert out == "垃圾3"


def test_rounds_limit():
    """轮次上限:LLM 一直要调工具时强制结束。"""
    agent, _ = make_agent([ACTION_JSON] * 10)
    agent.max_rounds = 3
    out = agent.run("查天气")
    assert "最大轮次" in out


def test_llm_error_friendly_message():
    """LLM 网络异常:返回友好提示而不是崩掉。"""
    class BoomLLM:
        def chat(self, messages):
            raise ConnectionError("no net")

    agent = Agent(default_registry(), BoomLLM(), Session("t"))
    out = agent.run("hi")
    assert "调用模型失败" in out


def test_two_windows_independent():
    """笔试场景:窗口1查天气记待办,窗口2写周报,互不影响。"""
    a1, _ = make_agent([ACTION_JSON, ANSWER_JSON], "window1")
    a2, _ = make_agent([ANSWER_JSON], "window2")
    a1.run("查天气记待办")
    a2.run("帮我写周报")
    assert len(a1.session.messages) == 4
    assert len(a2.session.messages) == 2
    assert not any("周报" in m.content for m in a1.session.messages)
    assert not any("[工具结果]" in m.content for m in a2.session.messages)


def test_context_sent_with_system_prompt():
    """每次请求都带 system prompt,且包含全部工具的 schema。"""
    agent, llm = make_agent([ANSWER_JSON])
    agent.run("hi")
    first = llm.sent[0]
    assert first[0]["role"] == "system"
    assert "weather" in first[0]["content"] and "calculator" in first[0]["content"]
