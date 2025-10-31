from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .schemas import Document
from .utils import choose_torch_device

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_HISTORY_DIR = BASE_DIR / "predictions" / "history"
PREDICTIONS_FULL_PATH = BASE_DIR / "predictions" / "predictions_full.csv"
EVALUATION_CSV = BASE_DIR / "predictions" / "evaluation" / "overall_metrics.csv"
VOLATILITY_CSV = BASE_DIR / "analysis" / "volatility_classifier_dataset.csv"
README_PATH = BASE_DIR / "README.md"
CALIBRATOR_PATH = BASE_DIR / "models" / "isotonic_calibrator.pkl"
VOL_METRICS_PATH = BASE_DIR / "analysis" / "volatility_classifier_metrics.json"

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "nfl_agent_docs"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBED_BATCH_SIZE = 256


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _chunk_text(text: str, max_tokens: int = 800, overlap: int = 200) -> Iterator[str]:
    """Simple rule-based splitter that approximates tokens by characters."""
    if not text:
        return
    approx = max_tokens * 4  # rough char-to-token mapping
    ov_char = overlap * 4
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + approx)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == length:
            break
        start = max(0, end - ov_char)


def _load_predictions_history() -> Iterable[Document]:
    if not PREDICTIONS_HISTORY_DIR.exists():
        history_frames: list[pd.DataFrame] = []
    else:
        history_frames = []
        csv_files = sorted(PREDICTIONS_HISTORY_DIR.glob("*.csv"))
        for csv_path in tqdm(csv_files, desc="Predictions history", unit="file"):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:  # pragma: no cover - diagnostic
                print(f"[indexer] skipping {csv_path}: {exc}")
                continue
            if df.empty:
                continue
            df["source_file"] = str(csv_path.relative_to(BASE_DIR))
            history_frames.append(df)

    if PREDICTIONS_FULL_PATH.exists():
        try:
            df_upcoming = pd.read_csv(PREDICTIONS_FULL_PATH)
        except Exception as exc:
            print(f"[indexer] skipping {PREDICTIONS_FULL_PATH}: {exc}")
        else:
            if not df_upcoming.empty:
                df_upcoming["source_file"] = str(PREDICTIONS_FULL_PATH.relative_to(BASE_DIR))
                history_frames.append(df_upcoming)

    if not history_frames:
        return []

    documents: list[Document] = []
    combined = pd.concat(history_frames, ignore_index=True)
    combined = combined.replace({np.nan: None})
    for _, row in combined.iterrows():
        season_raw = row.get("season", "")
        season = int(season_raw) if str(season_raw).isdigit() else season_raw
        week = row.get("week", "")
        teams = f"{row.get('away_team', '')} @ {row.get('home_team', '')}"
        probability = row.get("home_win_prob", "")
        margin = row.get("pred_home_margin", "")
        volatility_prob = row.get("volatility_prob", "")
        kickoff = row.get("kickoff", "")
        source_file = row.get("source_file", "")
        stage = row.get("prediction_source", "")
        text_parts = [
            f"Prediction ({stage or 'history'}) season {season}, week {week}: {teams}.",
            f"Home win probability {probability}.",
            f"Predicted home margin {margin}.",
            f"Volatility probability {volatility_prob}.",
        ]
        if kickoff:
            text_parts.append(f"Kickoff {kickoff}.")
        text = " ".join(filter(None, text_parts))
        metadata = {
            "source": "predictions_history",
            "file": source_file,
            "season": season,
            "week": week,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "game_id": row.get("game_id", ""),
            "kickoff": kickoff,
            "stage": stage,
        }
        documents.append(Document(text=text, metadata=metadata))
    return documents


def _load_evaluation_metrics() -> Iterable[Document]:
    if not EVALUATION_CSV.exists():
        return []
    try:
        df = pd.read_csv(EVALUATION_CSV)
    except Exception as exc:
        print(f"[indexer] skipping evaluation metrics: {exc}")
        return []
    documents: list[Document] = []
    df = df.fillna("")
    for _, row in df.iterrows():
        text = (
            f"Evaluation snapshot ({row.get('timestamp', '')}, stage={row.get('stage', '')}): "
            f"Brier={row.get('brier', '')}, AUC={row.get('auc', '')}, "
            f"Samples={row.get('samples', '')}, Volatility threshold={row.get('volatility_threshold', '')}, "
            f"Strength={row.get('volatility_strength', '')}, Coverage={row.get('volatility_coverage', '')}."
        )
        metadata = {
            "source": "evaluation_metrics",
            "timestamp": row.get("timestamp", ""),
            "stage": row.get("stage", ""),
        }
        documents.append(Document(text=text, metadata=metadata))
    return documents


def _load_volatility_dataset(sample_limit: int | None = 3000) -> Iterable[Document]:
    if not VOLATILITY_CSV.exists():
        return []
    try:
        df = pd.read_csv(VOLATILITY_CSV)
    except Exception as exc:
        print(f"[indexer] skipping volatility dataset: {exc}")
        return []
    if sample_limit and len(df) > sample_limit:
        # stratified sampling by season to keep index smaller
        df = (
            df.groupby("season", group_keys=False)
            .apply(lambda group: group.sample(min(len(group), math.ceil(sample_limit / df["season"].nunique())), random_state=42))
            .reset_index(drop=True)
        )
    documents: list[Document] = []
    df = df.fillna("")
    for _, row in df.iterrows():
        season = row.get("season", "")
        week = row.get("week", "")
        text = (
            f"Volatility record season {season}, week {week}, game {row.get('game_id', '')}: "
            f"volatility_prob={row.get('volatility_prob', '')}, "
            f"label={row.get('volatility_label', '')}, "
            f"threshold={row.get('volatility_threshold', '')}, "
            f"abs_margin_error={row.get('abs_margin_error', '')}, "
            f"log_loss_per_game={row.get('log_loss_per_game', '')}."
        )
        metadata = {
            "source": "volatility_dataset",
            "season": season,
            "week": week,
            "game_id": row.get("game_id", ""),
        }
        documents.append(Document(text=text, metadata=metadata))
    return documents


def _load_readme_chunks() -> Iterable[Document]:
    if not README_PATH.exists():
        return []
    text = README_PATH.read_text(encoding="utf-8", errors="ignore")
    documents: list[Document] = []
    for idx, chunk in enumerate(_chunk_text(text, max_tokens=600, overlap=100)):
        metadata = {"source": "readme", "chunk": idx}
        documents.append(Document(text=chunk, metadata=metadata))
    return documents


def _load_calibrator_metadata() -> Iterable[Document]:
    if not CALIBRATOR_PATH.exists():
        return []
    try:
        import joblib

        calibrator = joblib.load(CALIBRATOR_PATH)
    except Exception as exc:
        print(f"[indexer] skipping calibrator metadata: {exc}")
        return []
    metadata = calibrator.get("volatility_metadata", {})
    info = {
        "method": calibrator.get("method"),
        "sample_count": calibrator.get("sample_count"),
        "baseline_brier": calibrator.get("baseline_brier"),
        "calibrated_brier": calibrator.get("calibrated_brier"),
        "baseline_auc": calibrator.get("baseline_auc"),
        "calibrated_auc": calibrator.get("calibrated_auc"),
        "volatility_metadata": metadata,
    }
    text = (
        "Isotonic calibrator metadata: "
        f"method={info['method']}, samples={info['sample_count']}, "
        f"baseline_brier={info['baseline_brier']}, calibrated_brier={info['calibrated_brier']}, "
        f"baseline_auc={info['baseline_auc']}, calibrated_auc={info['calibrated_auc']}, "
        f"volatility={metadata}."
    )
    return [Document(text=text, metadata={"source": "calibrator", **info})]


def _load_volatility_metrics() -> Iterable[Document]:
    if not VOL_METRICS_PATH.exists():
        return []
    try:
        metrics = json.loads(VOL_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[indexer] skipping volatility metrics: {exc}")
        return []
    text = (
        f"Volatility classifier metrics: {metrics.get('metrics')} with settings {metrics.get('settings')}."
    )
    return [Document(text=text, metadata={"source": "volatility_metrics", **metrics})]


def _connect_milvus() -> None:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)


def _prepare_collection(embed_dim: int, drop_existing: bool) -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        if drop_existing:
            utility.drop_collection(COLLECTION_NAME)
        else:
            collection = Collection(COLLECTION_NAME)
            return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
    ]
    schema = CollectionSchema(fields, description="NFL predictions conversational index")
    collection = Collection(COLLECTION_NAME, schema)
    # Create index for vector search
    index_params = {
        "metric_type": "IP",
        "index_type": "AUTOINDEX",
        "params": {},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def _insert_documents(collection: Collection, embedder: SentenceTransformer, documents: List[Document]) -> None:
    if not documents:
        return
    # load existing max id
    existing_count = collection.num_entities
    start_id = existing_count
    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for batch_start in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts_batch = [doc.text for doc in batch]
        embeddings_batch = embedder.encode(texts_batch, batch_size=EMBED_BATCH_SIZE, device="cuda", show_progress_bar=False)
        for idx, doc in enumerate(batch):
            ids.append(start_id)
            start_id += 1
            texts.append(doc.text)
            metadatas.append(json.dumps(doc.metadata))
            embeddings.append(embeddings_batch[idx].tolist())

    collection.insert([ids, texts, metadatas, embeddings])
    collection.flush()


def _collect_documents() -> List[Document]:
    docs: list[Document] = []
    docs.extend(_load_predictions_history())
    docs.extend(_load_evaluation_metrics())
    docs.extend(_load_volatility_dataset())
    docs.extend(_load_readme_chunks())
    docs.extend(_load_calibrator_metadata())
    docs.extend(_load_volatility_metrics())
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Milvus index of local NFL model artifacts.")
    parser.add_argument("--reset", action="store_true", help="Drop existing Milvus collection before indexing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of documents ingested.")
    args = parser.parse_args()

    print("[indexer] collecting documents...")
    documents = _collect_documents()
    if args.limit is not None:
        documents = documents[: args.limit]
    if not documents:
        print("[indexer] No documents found to index.")
        return
    print(f"[indexer] collected {len(documents)} documents.")

    print("[indexer] loading embedding model...")
    device = choose_torch_device()
    print(f"[indexer] using torch device: {device}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device, trust_remote_code=True)
    embed_dim = embedder.get_sentence_embedding_dimension()

    print("[indexer] connecting to Milvus...")
    _connect_milvus()
    collection = _prepare_collection(embed_dim, drop_existing=args.reset)
    collection.load()

    print("[indexer] inserting embeddings...")
    _insert_documents(collection, embedder, documents)

    print(f"[indexer] completed. Collection '{COLLECTION_NAME}' now holds {collection.num_entities} entities.")


if __name__ == "__main__":
    main()
