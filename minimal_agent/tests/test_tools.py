"""工具与注册表测试:正常执行 + 各种错误路径都返回错误字符串而不是抛异常。"""
import json

from agent.tools import default_registry


def test_calculator_add():
    r = default_registry().execute("calculator", {"operation": "add", "num1": 2, "num2": 3})
    assert json.loads(r)["result"] == 5


def test_calculator_divide_by_zero():
    """除零 -> ZeroDivisionError 被 registry 兜底,变成错误字符串。"""
    r = default_registry().execute("calculator", {"operation": "divide", "num1": 1, "num2": 0})
    assert "error" in json.loads(r)


def test_unknown_tool():
    r = default_registry().execute("weathr", {"city": "广州"})   # 拼错的工具名
    data = json.loads(r)
    assert "unknown tool" in data["error"]
    assert "weather" in data["available"]          # 把可用列表告诉 LLM,让它自己改


def test_missing_param_caught_by_validation():
    """缺必填参数:执行前被 schema 校验拦下,错误里带上参数名和期望。"""
    r = default_registry().execute("weather", {})
    data = json.loads(r)
    assert "参数校验失败" in data["error"]
    assert "city" in data["error"]
    assert "expect" in data                                   # 把期望 schema 告诉 LLM


def test_typo_param_name_caught():
    r = default_registry().execute("weather", {"ciy": "广州"})   # 参数名打错
    assert "缺少必填参数 'city'" in json.loads(r)["error"]


def test_wrong_type_caught():
    r = default_registry().execute("weather", {"city": 123})     # 类型错
    assert "应为 string" in json.loads(r)["error"]


def test_bool_is_not_number():
    """Python 的 bool 是 int 子类,校验必须显式排除,否则 num1=true 能混过去。"""
    r = default_registry().execute("calculator", {"operation": "add", "num1": True, "num2": 1})
    assert "应为 number" in json.loads(r)["error"]


def test_unsupported_operation():
    r = default_registry().execute("calculator", {"operation": "pow", "num1": 2, "num2": 3})
    assert "不支持的操作" in json.loads(r)["error"]


def test_schemas_text_contains_all_tools():
    text = default_registry().schemas_text()
    for name in ("calculator", "weather", "search"):
        assert name in text
    assert "参数" in text
