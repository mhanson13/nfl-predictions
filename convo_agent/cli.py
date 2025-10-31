from __future__ import annotations

import argparse
import sys
from typing import List

from .chat import ChatOrchestrator
from .schemas import Document


def format_documents(docs: List[Document], title: str) -> str:
    if not docs:
        return ""
    lines = [title]
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata.get("source", "context")
        score = doc.metadata.get("score")
        score_part = f" (score={score:.3f})" if isinstance(score, float) else ""
        snippet = doc.text.strip().replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:140] + "..."
        lines.append(f"  {idx}. {meta}{score_part}: {snippet}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Terminal chat for the NFL conversational agent.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of passages to retrieve per query.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context after each answer.")
    parser.add_argument("--reset", action="store_true", help="Start with a fresh conversation session.")
    args = parser.parse_args(argv)

    try:
        orchestrator = ChatOrchestrator()
    except RuntimeError as exc:
        print(f"Failed to initialise conversational agent: {exc}")
        return 1
    if args.reset:
        orchestrator.reset()

    print("NFL Conversational Agent. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        result = orchestrator.ask(query, top_k=args.top_k)

        print(f"\n{result.answer}\n")

        if args.show_context:
            structured = format_documents(result.structured_docs, "Structured context:")
            retrieved = format_documents(result.vector_docs, "Retrieved passages:")
            if structured:
                print(structured)
            if retrieved:
                print(retrieved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
