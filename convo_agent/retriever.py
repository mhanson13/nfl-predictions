from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datetime import datetime

import numpy as np
import pandas as pd
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from .schemas import Document
from .utils import choose_torch_device

# ---------------------------------------------------------------------------
# Constants shared with the indexer
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_HISTORY_DIR = BASE_DIR / "predictions" / "history"
EVALUATION_CSV = BASE_DIR / "predictions" / "evaluation" / "overall_metrics.csv"
VOLATILITY_CSV = BASE_DIR / "analysis" / "volatility_classifier_dataset.csv"

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "nfl_agent_docs"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

# ---------------------------------------------------------------------------
# Helper structures
# ---------------------------------------------------------------------------

TEAM_CODES = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LV",
    "LAC",
    "LAR",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
}


@dataclass
class StructuredHit:
    """Structured lookup result that can be fed to the chat orchestrator."""

    text: str
    metadata: Dict[str, object]

    def to_document(self) -> Document:
        return Document(text=self.text, metadata=self.metadata)


# ---------------------------------------------------------------------------
# Structured lookups over tabular data
# ---------------------------------------------------------------------------


class StructuredLookup:
    """Provide dataframe-backed answers when the query targets explicit stats."""

    def __init__(self, base_dir: Path = BASE_DIR) -> None:
        self.base_dir = base_dir
        self._history_df: Optional[pd.DataFrame] = None
        self._metrics_df: Optional[pd.DataFrame] = None
        self._volatility_df: Optional[pd.DataFrame] = None

    # -- public API ---------------------------------------------------------

    def lookup(self, query: str) -> List[Document]:
        filters = self._detect_filters(query)
        hits: List[StructuredHit] = []

        if self._should_use_history(query, filters):
            hits.extend(self._summarize_history(filters))

        if self._should_use_metrics(query):
            hits.extend(self._summarize_metrics(filters))

        if self._should_use_volatility(query):
            hits.extend(self._summarize_volatility(filters))

        return [hit.to_document() for hit in hits]

    def describe(self) -> Dict[str, object]:
        """Return a lightweight snapshot of currently indexed tabular data."""
        history = self._load_history()
        metrics = self._load_metrics()
        volatility = self._load_volatility()

        seasons: List[int] = []
        if not history.empty and "season" in history.columns:
            seasons = sorted(
                int(s) for s in pd.to_numeric(history["season"], errors="coerce").dropna().astype(int).unique()
            )

        return {
            "history_rows": int(len(history)),
            "history_seasons": seasons,
            "metrics_rows": int(len(metrics)),
            "volatility_rows": int(len(volatility)),
            "latest_metric_timestamp": metrics["timestamp"].iloc[-1] if "timestamp" in metrics.columns and len(metrics) else None,
        }

    # -- detection helpers --------------------------------------------------

    def _detect_filters(self, query: str) -> Dict[str, object]:
        query_clean = query.lower()
        week_match = re.search(r"week[\s\-]*(\d{1,2})", query_clean)
        season_match = re.search(r"(20[0-4]\d)", query_clean)
        threshold_match = re.search(r"(?:threshold|cut(?:off)?)\s*(0\.\d+)", query_clean)
        date_match = re.search(r"(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](20\d{2})", query_clean)

        teams = {code for code in TEAM_CODES if code.lower() in query_clean}

        filters: Dict[str, object] = {}
        if week_match:
            filters["week"] = int(week_match.group(1))
        if season_match:
            filters["season"] = int(season_match.group(1))
        if teams:
            filters["teams"] = teams
        if threshold_match:
            filters["threshold"] = float(threshold_match.group(1))
        if date_match:
            month, day, year = map(int, date_match.groups())
            try:
                filters["date"] = datetime(year, month, day).date()
            except ValueError:
                pass

        return filters

    def _should_use_history(self, query: str, filters: Dict[str, object]) -> bool:
        if not PREDICTIONS_HISTORY_DIR.exists():
            return False
        keywords = ("prediction", "prob", "win", "margin", "bet", "confidence")
        return any(word in query.lower() for word in keywords) or "teams" in filters or "week" in filters

    def _should_use_metrics(self, query: str) -> bool:
        if not EVALUATION_CSV.exists():
            return False
        keywords = ("metric", "brier", "auc", "calibration", "evaluation")
        return any(word in query.lower() for word in keywords)

    def _should_use_volatility(self, query: str) -> bool:
        if not VOLATILITY_CSV.exists():
            return False
        keywords = ("volatility", "shrink", "ensemble", "high variance", "uncertain")
        return any(word in query.lower() for word in keywords)

    # -- dataframe loaders --------------------------------------------------

    def _load_history(self) -> pd.DataFrame:
        if self._history_df is not None:
            return self._history_df
        frames: List[pd.DataFrame] = []
        for path in sorted(PREDICTIONS_HISTORY_DIR.glob("*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            df["source_file"] = path.relative_to(self.base_dir).as_posix()
            frames.append(df)
        if frames:
            df_all = pd.concat(frames, ignore_index=True)
        else:
            df_all = pd.DataFrame()
        self._history_df = df_all
        return df_all

    def _load_metrics(self) -> pd.DataFrame:
        if self._metrics_df is not None:
            return self._metrics_df
        try:
            df = pd.read_csv(EVALUATION_CSV)
        except Exception:
            df = pd.DataFrame()
        self._metrics_df = df
        return df

    def _load_volatility(self) -> pd.DataFrame:
        if self._volatility_df is not None:
            return self._volatility_df
        try:
            df = pd.read_csv(VOLATILITY_CSV)
        except Exception:
            df = pd.DataFrame()
        self._volatility_df = df
        return df

    # -- summarizers --------------------------------------------------------

    def _summarize_history(self, filters: Dict[str, object]) -> List[StructuredHit]:
        df = self._load_history()
        if df.empty:
            return []

        mask = pd.Series(True, index=df.index)
        season = filters.get("season")
        week = filters.get("week")
        teams = filters.get("teams", set())
        target_date = filters.get("date")

        if season is not None and "season" in df.columns:
            mask &= df["season"] == season
        if week is not None and "week" in df.columns:
            week_numeric = pd.to_numeric(df["week"], errors="coerce")
            mask &= week_numeric == week
        if teams:
            team_mask = pd.Series(False, index=df.index)
            if "home_team" in df.columns:
                team_mask |= df["home_team"].astype(str).str.upper().isin(teams)
            if "away_team" in df.columns:
                team_mask |= df["away_team"].astype(str).str.upper().isin(teams)
            mask &= team_mask
        if target_date is not None:
            kickoff_col = None
            for candidate in ("kickoff", "game_date", "date"):
                if candidate in df.columns:
                    kickoff_col = candidate
                    break
            if kickoff_col:
                kickoff_ts = pd.to_datetime(df[kickoff_col], errors="coerce").dt.date
                mask &= kickoff_ts == target_date

        subset = df.loc[mask].copy()
        if subset.empty:
            return []

        numeric_cols = {
            "home_win_prob": "avg_home_win_prob",
            "pred_home_margin": "avg_home_margin",
            "volatility_prob": "avg_volatility_prob",
            "volatility_label": "volatility_rate",
        }
        aggregations: Dict[str, str] = {}
        for col, alias in numeric_cols.items():
            if col in subset.columns:
                if col == "volatility_label":
                    aggregations[col] = "mean"
                else:
                    aggregations[col] = "mean"

        summary_parts = []
        if aggregations:
            agg = subset.agg(aggregations)
            for col, alias in numeric_cols.items():
                if col in agg.index and not pd.isna(agg[col]):
                    value = float(agg[col])
                    if alias.endswith("prob") or "rate" in alias:
                        summary_parts.append(f"{alias}={value:.3f}")
                    else:
                        summary_parts.append(f"{alias}={value:.2f}")

        subset = subset.sort_values(by=["kickoff"] if "kickoff" in subset.columns else ["season", "week"])
        top_games = subset.head(5)
        lines = []
        for _, row in top_games.iterrows():
            teams = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
            prob = row.get("home_win_prob", "")
            margin = row.get("pred_home_margin", "")
            vol = row.get("volatility_prob", "")
            kickoff = row.get("kickoff", "")
            kickoff_str = f" | kickoff={kickoff}" if kickoff else ""
            lines.append(
                f"{row.get('game_id', '')}: {teams} | win_prob={prob} | margin={margin} | vol={vol}{kickoff_str}"
            )

        context = ", ".join(summary_parts) if summary_parts else "Summary statistics unavailable"
        text = (
            f"Structured prediction summary: games={len(subset)} | {context}. "
            f"Sample games:\n" + "\n".join(lines)
        )
        metadata = {
            "source": "structured_predictions",
            "filters": {k: sorted(v) if isinstance(v, set) else v for k, v in filters.items()},
            "rows": len(subset),
        }
        return [StructuredHit(text=text, metadata=metadata)]

    def _summarize_metrics(self, filters: Dict[str, object]) -> List[StructuredHit]:
        df = self._load_metrics()
        if df.empty:
            return []
        subset = df.copy()
        season = filters.get("season")
        if season is not None and "season" in subset.columns:
            subset = subset.loc[subset["season"] == season]
        if "timestamp" in subset.columns:
            subset = subset.sort_values("timestamp", ascending=False)
        subset = subset.head(5)
        rows = []
        for _, row in subset.iterrows():
            rows.append(
                f"{row.get('timestamp', '')}: stage={row.get('stage', '')}, "
                f"Brier={row.get('brier', '')}, AUC={row.get('auc', '')}, "
                f"Volatility threshold={row.get('volatility_threshold', '')}"
            )
        text = "Recent evaluation metrics:\n" + "\n".join(rows)
        metadata = {"source": "structured_metrics", "rows": len(subset)}
        return [StructuredHit(text=text, metadata=metadata)]

    def _summarize_volatility(self, filters: Dict[str, object]) -> List[StructuredHit]:
        df = self._load_volatility()
        if df.empty:
            return []

        threshold = filters.get("threshold", 0.6)
        season = filters.get("season")

        subset = df.copy()
        if season is not None and "season" in subset.columns:
            subset = subset.loc[subset["season"] == season]

        rows: List[str] = []
        if "volatility_prob" in subset.columns:
            vol_col = pd.to_numeric(subset["volatility_prob"], errors="coerce")
            high_vol = vol_col >= threshold
            rate = float(high_vol.mean()) if len(subset) else float("nan")
            count = int(high_vol.sum())
            preview = subset.loc[high_vol].head(5)
        else:
            rate = float("nan")
            count = 0
            preview = subset.head(5)

        for _, row in preview.iterrows():
            rows.append(
                f"{row.get('game_id', '')}: prob={row.get('volatility_prob', '')}, "
                f"label={row.get('volatility_label', '')}"
            )
        detail = "\n".join(rows) if rows else "No games above threshold in preview."

        rate_part = f"{rate:.3f}" if not np.isnan(rate) else "n/a"
        text = (
            f"Volatility summary (threshold={threshold:.2f}): "
            f"{count} games flagged, positive rate={rate_part}.\n"
            f"{detail}"
        )
        metadata = {
            "source": "structured_volatility",
            "threshold": threshold,
            "season": season,
            "flagged": count,
        }
        return [StructuredHit(text=text, metadata=metadata)]


# ---------------------------------------------------------------------------
# Milvus vector retrieval
# ---------------------------------------------------------------------------


class MilvusRetriever:
    """Wrapper around Milvus for dense retrieval over embedded artifacts."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        connections.connect("default", host=host, port=port)
        if not utility.has_collection(collection_name):
            raise RuntimeError(
                f"Milvus collection '{collection_name}' not found. "
                "Run `python -m convo_agent.data_indexer --reset` first."
            )

        self.collection = Collection(collection_name)
        self.collection.load()

        self.device = choose_torch_device()
        self.embedder = SentenceTransformer(embedding_model, device=self.device, trust_remote_code=True)

    def search(self, query: str, top_k: int = 6) -> List[Document]:
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["text", "metadata"],
        )

        documents: List[Document] = []
        if not results:
            return documents

        for hit in results[0]:
            payload = hit.entity
            metadata_raw = payload.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata_raw}
            metadata["score"] = float(hit.score)
            documents.append(Document(text=payload.get("text", ""), metadata=metadata))
        return documents


# ---------------------------------------------------------------------------
# Combined retrieval pipeline
# ---------------------------------------------------------------------------


class RetrievalPipeline:
    """Hybrid retriever that combines structured lookups with dense search."""

    def __init__(self) -> None:
        self.structured = StructuredLookup()
        self.vector = MilvusRetriever()

    def retrieve(self, query: str, top_k: int = 6) -> Tuple[List[Document], List[Document]]:
        structured_docs = self.structured.lookup(query)
        vector_docs = self.vector.search(query, top_k=top_k)
        return structured_docs, vector_docs
