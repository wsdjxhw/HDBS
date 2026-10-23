"""工具模块:基类 Tool、三个具体工具、注册表 ToolRegistry。

设计要点:
- 每个工具 = name + description + parameters(参数 schema,LLM 全靠它决定怎么传参)
- 执行层"永不抛异常":工具不存在 / 参数不匹配 / 内部崩溃,统一返回错误字符串,
  由 runtime 塞回 context,让 LLM 下一轮自己修正——错误也是上下文。
"""
import json


class Tool:
    def __init__(self, name, description, parameters):
        self.name = name                  # 注册名,统一小写;LLM 输出的 action.name 要和它逐字匹配
        self.description = description
        self.parameters = parameters      # 参数 schema: {参数名: {type, description}}

    def execute(self, **kwargs) -> str:
        raise NotImplementedError

    def schema_text(self) -> str:
        """渲染成给 LLM 看的文本(工具注册机制的最后一公里:对象 -> 文本)。"""
        return (
            f"- {self.name}: {self.description}\n"
            f"  参数: {json.dumps(self.parameters, ensure_ascii=False)}"
        )


class Calculator(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="简易计算工具,支持加减乘除。",
            parameters={
                "operation": {"type": "string", "description": "算术操作: add / subtract / multiply / divide"},
                "num1": {"type": "number", "description": "第一个数字"},
                "num2": {"type": "number", "description": "第二个数字"},
            },
        )

    def execute(self, operation, num1, num2) -> str:
        a, b = float(num1), float(num2)
        ops = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b,   # 除零时抛 ZeroDivisionError,由 registry 兜底转成错误字符串
        }
        if operation not in ops:
            return json.dumps(
                {"error": f"不支持的操作 '{operation}'", "可选": ["add", "subtract", "multiply", "divide"]},
                ensure_ascii=False,
            )
        return json.dumps(
            {"operation": operation, "num1": a, "num2": b, "result": ops[operation]},
            ensure_ascii=False,
        )


class Weather(Tool):
    """Mock 天气工具(演示用,返回固定假数据)。"""

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询指定城市今天的天气情况。",
            parameters={
                "city": {"type": "string", "description": "要查询天气的城市名称"}
            },
        )

    def execute(self, city) -> str:
        return json.dumps(
            {"city": city, "weather": "晴", "temp": "22~28℃", "wind": "东风2级", "advice": "适合出门"},
            ensure_ascii=False,
        )


class Search(Tool):
    """Mock 搜索工具(演示用,返回假结果)。"""

    def __init__(self):
        super().__init__(
            name="search",
            description="在互联网上搜索指定关键词,返回相关结果摘要。",
            parameters={
                "query": {"type": "string", "description": "要搜索的关键词"}
            },
        )

    def execute(self, query) -> str:
        return json.dumps(
            {"query": query, "results": [f"《{query}》百科词条:……(摘要)", f"关于「{query}」的 3 篇相关文章(摘要)……"]},
            ensure_ascii=False,
        )


def validate_args(schema: dict, args: dict) -> str | None:
    """按 schema 校验参数(本项目的简化 schema 中,每个字段都是必填)。

    schema 是契约而不只是提示词:渲染进 system prompt 是"告诉 LLM 怎么传",
    执行前校验是"不合法就不执行"。合法返回 None,非法返回给 LLM 看的错误说明。
    """
    for pname, spec in schema.items():
        if pname not in args:
            return f"缺少必填参数 '{pname}'({spec.get('description', '')})"
        value, expect = args[pname], spec.get("type")
        # 注意:Python 里 bool 是 int 的子类,必须显式排除,否则 num1=true 能混过 number 校验
        if expect == "string" and not isinstance(value, str):
            return f"参数 '{pname}' 应为 string,实际是 {type(value).__name__}"
        if expect == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"参数 '{pname}' 应为 number,实际是 {type(value).__name__}"
    return None


class ToolRegistry:
    def __init__(self):
        self.tools = {}   # name -> Tool 实例

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        return self

    def get(self, name) -> Tool | None:
        return self.tools.get(name)

    def execute(self, name, args: dict | None = None) -> str:
        """执行层永不抛异常,一切错误都变成字符串返回给 LLM。"""
        args = args or {}
        tool = self.get(name)
        if tool is None:
            return json.dumps(
                {"error": f"unknown tool '{name}'", "available": list(self.tools)},
                ensure_ascii=False,
            )
        try:
            error = validate_args(tool.parameters, args)   # 契约闸门:不合法就不执行
            if error:
                return json.dumps(
                    {"error": f"参数校验失败: {error}", "expect": tool.parameters},
                    ensure_ascii=False,
                )
            return str(tool.execute(**args))
        except TypeError as e:
            # 防御:多传了 schema 之外的参数名等(校验已拦必填/类型,这里兜底形状不匹配)
            return json.dumps(
                {"error": f"参数不匹配: {e}", "expect": tool.parameters},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    def schemas_text(self) -> str:
        """把所有工具渲染成文本,最终嵌进 system prompt。"""
        return "\n".join(t.schema_text() for t in self.tools.values())


def default_registry() -> ToolRegistry:
    r = ToolRegistry()
    for t in (Calculator(), Weather(), Search()):
        r.register(t)
    return r
