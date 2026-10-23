"""核心 Agent Runtime。

两层循环中的内层:一次用户输入 -> 循环(调 LLM -> 解析 -> 执行工具 -> 结果塞回 context)
-> 直到 LLM 给出最终答案 / 轮次上限 / 格式修复超限 / 异常兜底。

对应笔试题的四个 Step:
    Step one   接收用户输入          -> run() 开头 append(USER)
    Step two   判断直接回复还是调工具  -> parser 按 answer/action 字段判别(LLM 决策,代码执行)
    Step three 调用工具              -> registry.execute():先 schema 校验,后执行
    Step four  继续循环还是返回结果    -> ANSWER 则 return,否则 continue

runtime 对工具结果不做"搬运工",而是"第一读者"(评审教训):
    - 解析结果 JSON,识别 error 分支(单独计数、trace 单独标注,meta.ok 留痕)
    - 同名同参连续重复调用是 LLM 卡死的典型信号:第 3 次起拦截执行,
      返回合成纠正结果,把"别重复了,基于已有结果回答"喂回 context
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
        # 结果解读与卡死检测的状态(每次 run 重置)
        self._last_call_key: str | None = None
        self._repeat_streak = 0
        self.tool_errors = 0

    def run(self, user_input: str) -> str:
        self.session.append(USER, user_input)
        repairs = 0
        self._last_call_key, self._repeat_streak, self.tool_errors = None, 0, 0
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

                # 卡死检测:同名同参的连续调用(第 1、2 次可能是合法重试,第 3 次起拦截)
                call_key = name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)
                self._repeat_streak = self._repeat_streak + 1 if call_key == self._last_call_key else 0
                self._last_call_key = call_key
                if self._repeat_streak >= 2:
                    log.warning(
                        "[session=%s round=%d] 拦截重复调用 tool=%s(连续第 %d 次)",
                        self.session.id, round_no, name, self._repeat_streak + 1,
                    )
                    result = json.dumps(
                        {"error": "检测到连续重复调用同一工具与相同参数,已拦截。请基于已有工具结果,直接用 answer 回答用户。"},
                        ensure_ascii=False,
                    )
                else:
                    t0 = time.time()
                    result = self.registry.execute(name, args)   # 校验+执行,永不抛异常
                    log.info(
                        "[session=%s round=%d] tool=%s args=%s -> %dms",
                        self.session.id, round_no, name,
                        json.dumps(args, ensure_ascii=False),
                        (time.time() - t0) * 1000,
                    )

                # 结果解读:识别 error 分支,trace/计数/留痕区别于成功结果
                is_error = False
                try:
                    payload = json.loads(result)
                    is_error = isinstance(payload, dict) and "error" in payload
                except (json.JSONDecodeError, TypeError):
                    pass
                if is_error:
                    self.tool_errors += 1
                log.info(
                    "[session=%s round=%d] tool=%s outcome=%s",
                    self.session.id, round_no, name, "ERROR" if is_error else "OK",
                )
                self.session.append(TOOL, result, name=name, ok=not is_error)  # 结果塞回 context
                continue

            # INVALID:格式坏了,把修正指令塞回 context 让 LLM 重出;有上限,超限兜底
            repairs += 1
            log.warning("[session=%s round=%d] 非法输出(第 %d 次),尝试修复", self.session.id, round_no, repairs)
            if repairs > self.max_repairs:
                return raw
            self.session.append(USER, REPAIR_PROMPT)

        log.warning("[session=%s] 达到最大轮次 %d", self.session.id, self.max_rounds)
        return f"[系统] 已达到单次任务最大轮次({self.max_rounds}),任务提前结束。"
