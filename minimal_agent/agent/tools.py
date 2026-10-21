class ToolRegistry:
    def __init__(self):
        self.tools = {}
    def register_tool(self,tool):
        self.tools[tool.name] = tool

    def get_tool(self, name):
        return self.tools.get(name)

    def execute_tool(self, name, **kwargs):
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        return self._execute(tool, **kwargs)




    
class Tools:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

class Calculator(Tools):
        def __init__(self):
            super().__init__(
                name="Calculator",
                description="这是一个简易的计算工具,可以进行加减乘除运算。",
                parameters={
                    "operation": {
                        "type": "string",
                        "description": "要进行的算术操作(add, subtract, multiply, divide)。"
                    },
                    "num1": {
                        "type": "number",
                        "description": "操作的第一个数字。"
                    },
                    "num2": {
                        "type": "number",
                        "description": "操作的第二个数字。"
                    }
                }
            )

class weather(Tools):
        def __init__(self):
            super().__init__(
                name="weather",
                description="这是一个天气查询工具,可以查询指定城市的天气情况。",
                parameters={
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市名称。"
                    }
                }
            )

class search(Tools):
        def __init__(self):
            super().__init__(
                name="search",
                description="这是一个搜索工具,可以在互联网上搜索指定的关键词。",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词。"
                    }
                }
            )