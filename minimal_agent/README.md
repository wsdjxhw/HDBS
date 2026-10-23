# Minimal Agent

从零实现的最小可用 Agent Runtime——不依赖任何 agent 框架(langgraph / openhands / PI 等),
核心循环、工具注册、输出协议解析、session 与 context 管理均为手写,仅用 `requests` 调用 LLM 的 HTTP API。

## 与笔试要求对照

| 要求 | 实现位置 |
|---|---|
| 基本循环(接收输入→判断→调工具→继续/返回) | `agent/runtime.py` |
| 三个工具 calculator / search(mock) / weather(mock) | `agent/tools.py` |
| 工具注册机制(名称+描述+参数 Schema) | `agent/tools.py`:schema 渲染进 system prompt(给 LLM 看)+ 执行前 `validate_args` 校验必填/类型——schema 是契约,不只是提示词 |
| LLM 输出解析(思考过程/工具调用/最终答案) | `agent/parser.py`,自定义 JSON 协议 |
| session 管理(多窗口独立、可随时接着聊) | `agent/context.py`,每窗口一个 Session,JSON 落盘,重启可续 |
| context 管理(轮次限制、追问、超长压缩) | `agent/runtime.py`(max_rounds)+ `agent/context.py` 结构感知压缩:输出永远是合法 JSON,被丢弃轮次以摘要替代 |
| 基本异常处理 | LLM 网络错误友好提示;工具层"永不抛异常",错误作为字符串塞回 context |
| 工具调用 trace / 执行日志 | `logging` 输出每轮 thought / action / 耗时 / 结果摘要(outcome=OK/ERROR),落 `logs/agent.log` |
| 测试用例 | `tests/` 37 个用例,FakeLLM 跑完整循环,不依赖真实 API |

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入你的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL(任意 OpenAI 兼容端点)

python main.py --session window1    # 窗口1:查天气记待办
python main.py --session window2    # 窗口2:另开终端,写周报,与窗口1互不影响

python -m pytest                    # 跑测试
```

REPL 内命令:`/history` 查看当前 session 历史,`/exit` 退出。进程退出后用同名 session 再启动,历史从 `sessions/<name>.json` 恢复,可以接着聊。

## 系统设计

### 两层循环

```
外层:REPL(对话)          内层:agent loop(单次任务)
while True:                while round <= max_rounds:
    user_input = input()       messages = system + 压缩后的历史
    agent.run(input)           raw = llm.chat(messages)          # ① 调模型
                               pr  = parse_llm_output(raw)       # ② 解析:answer / action / invalid
                               if answer: return                 # ④ 终止
                               result = registry.execute(action) # ③ 调工具
                               session.append(TOOL, result)      #    结果塞回 context,继续循环
```

所有循环中的消息(user / assistant 原文 / tool 结果)都 append 进 Session——这就是"哪些信息要塞入 context"的答案:
**用户输入、LLM 每轮输出原文、工具执行结果,三者缺一不可**(assistant 原文保留保证了协议轮次的连续性)。

### 输出协议(自定义,非原生 function calling)

LLM 每轮只输出一个 JSON,二选一,用字段 presence 判别、不设标志位:

```json
{"thought": "...", "action": {"name": "weather", "args": {"city": "广州"}}}
{"thought": "...", "answer": "最终答案"}
```

选自定义协议而非原生 function calling 的原因:题目考察的正是解析逻辑本身;
同时 `thought` 天然沉淀为 trace 素材。代价是格式无保证,因此 parser 做了容错:
取第一个 `{` 到最后一个 `}` 之间内容再 `json.loads`(治代码块围栏/夹带废话);
非法输出把修复指令塞回 context 让 LLM 重出,上限 2 次,超限兜底返回原文。

### 工具注册机制

```
Tool(name, description, parameters) ──register──▶ ToolRegistry(字典)
                                                      │
                                        schemas_text() 渲染成文本
                                                      ▼
                                             system prompt(每次请求都带)
```

执行层"永不抛异常":工具不存在、参数校验失败、内部崩溃(如除零),一律返回错误 JSON 字符串,
作为工具结果塞回 context——**错误也是上下文**,LLM 下一轮自己修正(重试或换路)。

runtime 同时是工具结果的**第一读者**:每轮解析结果、识别 error 分支(单独计数与 trace 标注,
`meta.ok` 留痕),并对**同名同参的连续重复调用**做卡死拦截(第 3 次起不再执行,返回合成
纠正结果喂回 context,不烧轮次、不重复副作用)。

### Session 与 context 管理

- **隔离**:每个窗口一个 `Session` 对象,天然互不影响;每轮对话后落盘 `sessions/<id>.json`。
- **召回与放置**(memory 说明):
  - *召回时机*:每次调 LLM 前,把该 session 的全部历史(经压缩)拼进 messages;session 文件在 `--session` 启动时按名加载——**按 session 名精确召回,不做相似度检索**。对一个 CLI agent,精确召回是最合适的粒度;RAG 式语义召回适合海量历史,这里引入只增加复杂度。
  - *放置方式*:`system(含工具 schema 与协议) + 完整历史`,工具结果以 `[工具结果]` 前缀的 user 消息呈现(不依赖原生 tool 角色,任何兼容端点都能跑)。system prompt 每次请求都完整携带。
- **压缩**(**结构感知**,触发于发给 LLM 前超过 8000 字符):① 旧工具结果按 JSON 字段
  截断后**重新序列化**——进 context 的永远是合法 JSON,绝不产生残缺片段污染协议
  (LLM 的历史就是它的 few-shot 示例);② 旧 assistant 输出只保留 thought 与动作标记
  (历史的价值在"格式正确"这个示范,内容可丢);③ 仍超预算则丢弃最旧轮次,但**保住
  第一条 user**(承载会话目标),被丢弃的轮次以一行规则摘要替代——基础的分层记忆
  (摘要层+原文层)。session 内永远保留完整原文,压缩只发生在"发送视图"上。
- **轮次限制**:单次任务最多 8 轮,防失控;追问时历史仍在,支持纯对话追问与带工具结果追问。

## 已知限制

- 压缩的摘要层是规则式一行摘要,未用 LLM 生成语义摘要(题目说明复杂压缩不必实现);预算按字符数而非 token
- search / weather 为 mock;无并发与速率控制;单用户 CLI 形态
