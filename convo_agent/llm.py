from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from gpt4all import GPT4All

from .utils import choose_gpt4all_device


def _env_model_override() -> Optional[str]:
    return os.getenv("CONVO_AGENT_LLM_MODEL") or os.getenv("GPT4ALL_MODEL_NAME")


MODEL_CANDIDATES = [
    _env_model_override(),
    "orca-mini-3b-gguf2-q4_0",
    "gpt4all-falcon-q4_0",
    "ggml-gpt4all-j-v1.3-groovy",
]


@dataclass
class Message:
    """Minimal chat message abstraction."""

    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class GPT4AllLLM:
    """Thread-safe wrapper around GPT4All inference."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

        gpt_device = choose_gpt4all_device()

        tried: List[str] = []
        last_error: Optional[Exception] = None

        candidates = [model_name] if model_name else MODEL_CANDIDATES
        for candidate in candidates:
            if not candidate:
                continue
            tried.append(candidate)
            try:
                model = GPT4All(
                    model_name=candidate,
                    model_path=model_path,
                    allow_download=True,
                    device=gpt_device,
                )
                self._verify_model(model)
                self._model = model
                self.model_name = candidate
                break
            except Exception as exc:  # pragma: no cover - depends on local install
                last_error = exc
        else:
            attempted = ", ".join(tried) or "<none>"
            raise RuntimeError(
                f"Failed to load a GPT4All model (tried: {attempted}). "
                "Delete any corrupted files under ~/.cache/gpt4all or set CONVO_AGENT_LLM_MODEL to a known-good model."
            ) from last_error

        # GPT4All isn't inherently thread-safe; guard generation with a lock.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, messages: Iterable[Message], **overrides) -> str:
        """Generate a completion given a sequence of chat messages."""

        params = {
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "repeat_penalty": overrides.get("repeat_penalty", self.repeat_penalty),
        }
        message_dicts = [msg.as_dict() for msg in messages]

        with self._lock:
            try:
                if hasattr(self._model, "chat_completion"):
                    response = self._model.chat_completion(
                        messages=message_dicts,
                        **params,
                    )
                    content: str = ""
                    if response and "choices" in response and response["choices"]:
                        content = response["choices"][0]["message"]["content"]
                    return content.strip()

                # Fallback for older GPT4All builds that expose `generate`.
                prompt = self._build_prompt(message_dicts)
                content = self._model.generate(
                    prompt=prompt,
                    temp=params["temperature"],
                    top_p=params["top_p"],
                    max_tokens=params["max_tokens"],
                    repeat_penalty=params["repeat_penalty"],
                )
                return content.strip()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "GPT4All failed to generate a response. "
                    "Delete cached models under ~/.cache/gpt4all or choose a different model via CONVO_AGENT_LLM_MODEL."
                ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_model(self, model: GPT4All) -> None:
        """Run a tiny warm-up query to ensure the backend is ready."""
        try:
            if hasattr(model, "chat_completion"):
                model.chat_completion(
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                    top_p=0.1,
                    repeat_penalty=1.0,
                )
            else:
                model.generate(
                    prompt="ping",
                    max_tokens=1,
                    temp=0.0,
                    top_p=0.1,
                    repeat_penalty=1.0,
                )
        except Exception as exc:
            raise RuntimeError("GPT4All warm-up failed") from exc

    @staticmethod
    def _build_prompt(messages: Iterable[Dict[str, str]]) -> str:
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)
