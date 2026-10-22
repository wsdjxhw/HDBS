"""核心 Agent Runtime。

两层循环中的内层:一次用户输入 -> 循环(调 LLM -> 解析 -> 执行工具 -> 结果塞回 context)
-> 直到 LLM 给出最终答案 / 轮次上限 / 格式修复超限 / 异常兜底。

对应笔试题的四个 Step:
    Step one   接收用户输入          -> run() 开头 append(USER)
    Step two   判断直接回复还是调工具  -> parser 按 answer/action 字段判别(LLM 决策,代码执行)
    Step three 调用工具              -> registry.execute(),结果 append(TOOL)
    Step four  继续循环还是返回结果    -> ANSWER 则 return,否则 continue
"""
import json
import logging
import time

from agent.context import ASSISTANT, TOOL, USER, build_llm_messages
from agent.parser import ACTION, ANSWER, parse_llm_output
from agent.prompts import REPAIR_PROMPT, build_system_prompt

log = logging.getLogger("agent")


class Agent:
    def __init__(
        self,
        registry,
        llm,
        session,
        system_prompt: str | None = None,
        max_rounds: int = 8,          # 单次 run 的最大循环轮次
        max_repairs: int = 2,         # 格式修复的最大次数,超限兜底返回原文
        max_context_chars: int = 8000,
    ):
        self.registry = registry
        self.llm = llm
        self.session = session
        self.system_prompt = system_prompt or build_system_prompt(registry)
        self.max_rounds = max_rounds
        self.max_repairs = max_repairs
        self.max_context_chars = max_context_chars

    def run(self, user_input: str) -> str:
        self.session.append(USER, user_input)
        repairs = 0
        for round_no in range(1, self.max_rounds + 1):
            messages = build_llm_messages(self.system_prompt, self.session.messages, self.max_context_chars)
            try:
                raw = self.llm.chat(messages)
            except Exception as e:  # 基本异常处理:网络波动/服务错误,返回友好提示而不是崩掉
                log.error("[session=%s round=%d] LLM 调用失败: %s", self.session.id, round_no, e)
                return f"[系统] 调用模型失败,请稍后再试({type(e).__name__})"

            self.session.append(ASSISTANT, raw)
            pr = parse_llm_output(raw)
            log.info(
                "[session=%s round=%d] type=%s thought=%s",
                self.session.id, round_no, pr.type, pr.thought[:80],
            )

            if pr.type == ANSWER:
                return pr.answer

            if pr.type == ACTION:
                name = pr.action["name"]
                args = pr.action.get("args", {})
                t0 = time.time()
                result = self.registry.execute(name, args)   # 永不抛异常,错误也是字符串
                log.info(
                    "[session=%s round=%d] tool=%s args=%s -> %dms %s",
                    self.session.id, round_no, name,
                    json.dumps(args, ensure_ascii=False),
                    (time.time() - t0) * 1000, result[:120],
                )
                self.session.append(TOOL, result, name=name)  # 工具结果塞回 context,喂给下一轮
                continue

            # INVALID:格式坏了,把修正指令塞回 context 让 LLM 重出;有上限,超限兜底
            repairs += 1
            log.warning("[session=%s round=%d] 非法输出(第 %d 次),尝试修复", self.session.id, round_no, repairs)
            if repairs > self.max_repairs:
                return raw
            self.session.append(USER, REPAIR_PROMPT)

        log.warning("[session=%s] 达到最大轮次 %d", self.session.id, self.max_rounds)
        return f"[系统] 已达到单次任务最大轮次({self.max_rounds}),任务提前结束。"
