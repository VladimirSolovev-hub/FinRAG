from __future__ import annotations
import argparse
import uuid
from pathlib import Path

from agentic.config import settings
from agentic.graph import invoke_agent
from agentic.tracing import configure_logging, write_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="FinRAG Agentic Financial Analytics")
    parser.add_argument("question", nargs="?", help="Вопрос агенту. Если не задан, включается интерактивный режим.")
    args = parser.parse_args()
    configure_logging()
    thread_id = str(uuid.uuid4())

    def ask(question: str) -> None:
        result = invoke_agent(question, thread_id=thread_id)
        print("\n" + result["answer"] + "\n")
        write_trace(settings.trace_path, {
            "thread_id": thread_id,
            "question": question,
            "answer": result["answer"],
            "tool_events": result["tool_events"],
            "evidence_reviews": result.get("evidence_reviews", []),
        })

    if args.question:
        ask(args.question)
        return
    print("FinRAG Agent. Для выхода введи /exit")
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if question:
            try:
                ask(question)
            except Exception as exc:
                print(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
