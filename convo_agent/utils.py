from __future__ import annotations

import warnings


def choose_torch_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda' when a CUDA-enabled torch build is available, else 'cpu'."""
    if not prefer_gpu:
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "cuda", None) and torch.backends.cuda.is_built():
            # CUDA build present but device not available (e.g., drivers missing)
            warnings.warn("CUDA build detected but no GPU runtime available; falling back to CPU.", RuntimeWarning)
    except Exception:  # pragma: no cover - defensive
        warnings.warn("Unable to determine CUDA availability; defaulting to CPU embeddings.", RuntimeWarning)
    return "cpu"


def choose_gpt4all_device() -> str:
    """Map torch device selection to GPT4All's expected device string."""
    device = choose_torch_device()
    return "gpu" if device == "cuda" else "cpu"

