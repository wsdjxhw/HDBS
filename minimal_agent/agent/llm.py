"""LLM 客户端:OpenAI 兼容的 /chat/completions 接口。

通过环境变量配置,适配 DeepSeek / 智谱 / 通义 / OpenAI 等任意兼容端点:
    LLM_BASE_URL   默认 https://api.deepseek.com/v1
    LLM_API_KEY    必填(从 .env 读取,绝不进 git)
    LLM_MODEL      默认 deepseek-chat
"""
import os

import requests


class LLMClient:
    def __init__(self, base_url, api_key, model, temperature=0.3, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
        )

    def chat(self, messages: list[dict]) -> str:
        """输入 OpenAI 格式的 messages,返回模型输出的纯文本。"""
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
