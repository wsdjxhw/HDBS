"""交互入口。

复现笔试题的"窗口1/窗口2"场景:开两个终端,各用一个 session 名,即可同时聊、互不影响,
进程重启后凭同名 session 继续接着聊:

    python main.py --session window1
    python main.py --session window2

REPL 内命令:/exit 退出,/history 查看当前 session 历史。
"""
import argparse
import logging
import os

try:  # 有 .env 就加载(放了 LLM_API_KEY),没有也不影响
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent.context import SessionManager
from agent.llm import LLMClient
from agent.runtime import Agent
from agent.tools import default_registry


def main():
    parser = argparse.ArgumentParser(description="最小可用 Agent")
    parser.add_argument("--session", default="default", help="session 名,不同窗口用不同名字")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/agent.log", encoding="utf-8")],
    )

    sm = SessionManager()
    session = sm.load(args.session)
    agent = Agent(default_registry(), LLMClient.from_env(), session)

    print(f"[session={session.id}] 已就绪(已有历史 {len(session.messages)} 条)。/exit 退出,/history 查看历史。")
    while True:
        try:
            text = input(f"[{session.id}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/history":
            for m in session.messages:
                print(f"  {m.role:9s} {m.content[:80]}")
            continue

        answer = agent.run(text)
        print(f"agent> {answer}\n")
        sm.save(session)   # 每轮落盘,随时可接着聊

    print("bye.")


if __name__ == "__main__":
    main()
