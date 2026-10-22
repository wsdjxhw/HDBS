"""parser 测试:各种正常/异常输出。"""
from agent.parser import ACTION, ANSWER, INVALID, parse_llm_output

ACTION_JSON = '{"thought": "查天气", "action": {"name": "weather", "args": {"city": "广州"}}}'
ANSWER_JSON = '{"thought": "信息够了", "answer": "广州今天晴。"}'


def test_action():
    pr = parse_llm_output(ACTION_JSON)
    assert pr.type == ACTION
    assert pr.action["name"] == "weather"
    assert pr.action["args"] == {"city": "广州"}
    assert pr.thought == "查天气"


def test_answer():
    pr = parse_llm_output(ANSWER_JSON)
    assert pr.type == ANSWER
    assert pr.answer == "广州今天晴。"


def test_answer_wins_over_action():
    """两个字段都有时按最终答案走,让循环尽早退出。"""
    pr = parse_llm_output('{"answer": "好了", "action": {"name": "weather", "args": {}}}')
    assert pr.type == ANSWER


def test_fenced_json():
    """最常见坏法:套 markdown 代码块。"""
    pr = parse_llm_output(f"```json\n{ANSWER_JSON}\n```")
    assert pr.type == ANSWER


def test_json_with_chatter():
    """第二种坏法:前后夹废话。"""
    pr = parse_llm_output(f"好的,这是我的回复:\n{ACTION_JSON}\n希望对你有帮助")
    assert pr.type == ACTION


def test_no_json():
    assert parse_llm_output("我不会输出 JSON").type == INVALID


def test_broken_json():
    assert parse_llm_output('{"thought": ').type == INVALID


def test_non_object_json():
    assert parse_llm_output("[1, 2, 3]").type == INVALID


def test_action_missing_name():
    assert parse_llm_output('{"action": {"args": {}}}').type == INVALID


def test_action_args_default_empty():
    """args 缺省容忍为 {},协议对 LLM 宽松一点。"""
    pr = parse_llm_output('{"action": {"name": "weather"}}')
    assert pr.type == ACTION
    assert pr.action["args"] == {}
