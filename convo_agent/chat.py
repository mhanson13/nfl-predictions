from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .llm import GPT4AllLLM, Message
from .retriever import RetrievalPipeline
from .schemas import Document


SYSTEM_PROMPT = (
    "You are a local analytics copilot for NFL modeling. "
    "Answer using the supplied context from predictions, calibration metrics, "
    "and volatility analysis. When providing statistics, prefer structured "
    "tables or bullet summaries. Always reference seasons, weeks, and teams "
    "explicitly when available. If information is missing, say so."
)


@dataclass
class ChatResponse:
    answer: str
    structured_docs: List[Document]
    vector_docs: List[Document]
    context: str = ""


class ChatOrchestrator:
    """Coordinate retrieval-augmented responses from the local LLM."""

    def __init__(
        self,
        retriever: Optional[RetrievalPipeline] = None,
        llm: Optional[GPT4AllLLM] = None,
        max_context_chars: int = 4000,
    ) -> None:
        self.retriever = retriever or RetrievalPipeline()
        self.llm = llm or GPT4AllLLM()
        self.max_context_chars = max_context_chars

        self._history: List[Message] = [Message(role="system", content=SYSTEM_PROMPT)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str, top_k: int = 6) -> ChatResponse:
        structured_docs, vector_docs = self.retriever.retrieve(query, top_k=top_k)
        context = self._build_context(structured_docs, vector_docs)

        context_message = Message(role="system", content=context)
        user_message = Message(role="user", content=query)

        conversation: List[Message] = self._history + [context_message, user_message]
        answer = self.llm.generate(conversation)

        # Maintain rolling history with the latest user/assistant turns, but drop
        # the ad-hoc context message to avoid uncontrolled growth.
        self._history.append(user_message)
        self._history.append(Message(role="assistant", content=answer))

        return ChatResponse(
            answer=answer,
            structured_docs=structured_docs,
            vector_docs=vector_docs,
            context=context,
        )

    def reset(self) -> None:
        self._history = [Message(role="system", content=SYSTEM_PROMPT)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, structured_docs: List[Document], vector_docs: List[Document]) -> str:
        lines: List[str] = ["# Retrieved Context"]

        if structured_docs:
            lines.append("## Structured summaries")
            lines.extend(self._format_documents(structured_docs, with_scores=False))

        if vector_docs:
            lines.append("## Retrieved passages")
            lines.extend(self._format_documents(vector_docs, with_scores=True))

        context = "\n".join(lines)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n[context truncated]"
        return context

    def _format_documents(self, docs: Iterable[Document], *, with_scores: bool) -> List[str]:
        formatted: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            meta = self._summarize_metadata(doc.metadata, with_scores=with_scores)
            text = doc.text.strip().replace("\n", " ")
            if len(text) > 600:
                text = text[:600] + "..."
            formatted.append(f"{idx}. {meta}\n{text}")
        return formatted

    @staticmethod
    def _summarize_metadata(metadata: Dict[str, object], *, with_scores: bool) -> str:
        parts: List[str] = []
        source = metadata.get("source")
        if source:
            parts.append(f"source={source}")

        season = metadata.get("season")
        if season:
            parts.append(f"season={season}")

        week = metadata.get("week")
        if week:
            parts.append(f"week={week}")

        if with_scores and "score" in metadata:
            parts.append(f"score={float(metadata['score']):.3f}")

        file = metadata.get("file")
        if file and not parts:
            parts.append(f"file={file}")

        if not parts:
            return "context"
        return ", ".join(parts)

