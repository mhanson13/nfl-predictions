from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chat import ChatOrchestrator, ChatResponse
from .retriever import StructuredLookup
from .schemas import Document

app = FastAPI(
    title="NFL Predictions Conversational API",
    description="Query local model outputs, calibration artifacts, and volatility metrics via natural language.",
    version="0.1.0",
)

chat = ChatOrchestrator()
lookup = StructuredLookup()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DocumentPayload(BaseModel):
    text: str
    metadata: Dict[str, Any]

    @classmethod
    def from_document(cls, doc: Document) -> "DocumentPayload":
        return cls(text=doc.text, metadata=_to_python(doc.metadata))


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=12)


class QueryResponseModel(BaseModel):
    answer: str
    context: str
    structured: List[DocumentPayload]
    retrieved: List[DocumentPayload]


class SummaryResponse(BaseModel):
    history_rows: int
    history_seasons: List[int]
    metrics_rows: int
    volatility_rows: int
    latest_metric_timestamp: Optional[str]


class ExplanationResponse(BaseModel):
    game_id: str
    prediction: Dict[str, Any]
    volatility: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponseModel)
def run_query(request: QueryRequest) -> QueryResponseModel:
    result: ChatResponse = chat.ask(request.query, top_k=request.top_k)

    structured = [DocumentPayload.from_document(doc) for doc in result.structured_docs]
    retrieved = [DocumentPayload.from_document(doc) for doc in result.vector_docs]

    return QueryResponseModel(
        answer=result.answer,
        context=result.context,
        structured=structured,
        retrieved=retrieved,
    )


@app.get("/summary", response_model=SummaryResponse)
def dataset_summary() -> SummaryResponse:
    data = lookup.describe()
    return SummaryResponse(**data)


@app.get("/explain/{game_id}", response_model=ExplanationResponse)
def explain_game(game_id: str) -> ExplanationResponse:
    history = lookup._load_history()  # pylint: disable=protected-access
    if history.empty:
        raise HTTPException(status_code=404, detail="No prediction history available.")

    if "game_id" not in history.columns:
        raise HTTPException(status_code=404, detail="Prediction history missing game identifiers.")

    subset = history.loc[history["game_id"].astype(str) == game_id]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found in predictions.")

    latest = subset.sort_index().iloc[-1].to_dict()
    prediction = _to_python(latest)

    volatility = None
    volatility_df = lookup._load_volatility()  # pylint: disable=protected-access
    if not volatility_df.empty and "game_id" in volatility_df.columns:
        vol_row = volatility_df.loc[volatility_df["game_id"].astype(str) == game_id]
        if not vol_row.empty:
            volatility = _to_python(vol_row.iloc[-1].to_dict())

    return ExplanationResponse(game_id=game_id, prediction=prediction, volatility=volatility)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_python(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_python(item) for item in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
