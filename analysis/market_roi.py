from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PREDICTIONS_PATH = Path("predictions/evaluation/merged_predictions_actuals.csv")
MATCHUP_FEATURES_PATH = Path("data/processed/matchup_features.parquet")


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0


def american_to_implied_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def standard_normal_cdf(value: pd.Series | np.ndarray | float) -> np.ndarray:
    scale = sqrt(2.0)
    arr = np.asarray(value, dtype=float)
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(arr / scale))


def load_predictions(start_season: int | None, end_season: int | None) -> pd.DataFrame:
    if not MERGED_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{MERGED_PREDICTIONS_PATH} not found. Run evaluation before market ROI analysis."
        )
    df = pd.read_csv(MERGED_PREDICTIONS_PATH)
    if start_season is not None:
        df = df[df["season"] >= start_season]
    if end_season is not None:
        df = df[df["season"] <= end_season]
    df = df.dropna(subset=["home_win_prob", "pred_home_margin", "home_margin"])
    df["home_margin"] = df["home_margin"].astype(float)
    return df


def load_market_lines(columns: Iterable[str]) -> pd.DataFrame:
    if not MATCHUP_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHUP_FEATURES_PATH} not found. Build features before market ROI analysis."
        )
    try:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH, columns=list(columns))
    except Exception:
        df = pd.read_parquet(MATCHUP_FEATURES_PATH, columns=list(columns), engine="fastparquet")
    return df


def prepare_dataset(pred_df: pd.DataFrame) -> pd.DataFrame:
    odds_columns = [
        "game_id",
        "home_moneyline",
        "away_moneyline",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
    ]
    odds_df = load_market_lines(odds_columns)
    if odds_df.duplicated("game_id").any():
        dupes = odds_df.loc[odds_df.duplicated("game_id", keep=False), "game_id"].unique()
        print(f"[market_roi] warning: dropping duplicate odds rows for game_id(s) {dupes[:5]!r}")
        odds_df = odds_df.drop_duplicates("game_id", keep="last")
    merged = pred_df.merge(odds_df, on="game_id", how="left", validate="one_to_one")
    merged = merged.dropna(subset=["home_moneyline", "away_moneyline", "spread_line"])

    merged["away_win_prob"] = 1.0 - merged["home_win_prob"]
    merged["home_decimal"] = merged["home_moneyline"].apply(american_to_decimal)
    merged["away_decimal"] = merged["away_moneyline"].apply(american_to_decimal)
    merged["home_implied_prob"] = merged["home_moneyline"].apply(american_to_implied_prob)
    merged["away_implied_prob"] = merged["away_moneyline"].apply(american_to_implied_prob)
    merged["home_tie"] = merged["home_margin"] == 0.0
    merged["home_actual_win"] = (merged["home_margin"] > 0.0).astype(int)
    merged["away_actual_win"] = (merged["home_margin"] < 0.0).astype(int)

    merged["home_spread_decimal"] = merged["home_spread_odds"].apply(american_to_decimal)
    merged["away_spread_decimal"] = merged["away_spread_odds"].apply(american_to_decimal)
    merged["spread_edge_points"] = merged["pred_home_margin"] + merged["spread_line"]
    merged["spread_result_points"] = merged["home_margin"] + merged["spread_line"]
    merged["home_cover"] = merged["spread_result_points"] > 0.0
    merged["away_cover"] = merged["spread_result_points"] < 0.0
    merged["spread_push"] = merged["spread_result_points"] == 0.0
    merged["margin_error"] = merged["pred_home_margin"] - merged["home_margin"]
    return merged


def summarize_moneyline(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for side in ("home", "away"):
        prob_col = "home_win_prob" if side == "home" else "away_win_prob"
        decimal_col = f"{side}_decimal"
        implied_col = f"{side}_implied_prob"
        actual_col = f"{side}_actual_win"
        tie_col = "home_tie"
        moneyline_col = f"{side}_moneyline"

        base = df.dropna(subset=[prob_col, decimal_col, implied_col, moneyline_col]).copy()
        base["expected_profit"] = base[prob_col] * (base[decimal_col] - 1.0) - (1.0 - base[prob_col])

        for threshold in thresholds:
            selection = base[base["expected_profit"] > threshold]
            n_bets = int(len(selection))
            if n_bets == 0:
                records.append(
                    {
                        "bet_type": "moneyline",
                        "bet_side": side,
                        "threshold": float(threshold),
                        "n_bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pushes": 0,
                        "hit_rate": np.nan,
                        "avg_model_prob": np.nan,
                        "avg_implied_prob": np.nan,
                        "avg_expected_profit": np.nan,
                        "avg_edge_prob": np.nan,
                        "avg_odds_decimal": np.nan,
                        "avg_odds_american": np.nan,
                        "total_profit": 0.0,
                        "roi": np.nan,
                    }
                )
                continue

            pushes = int(selection[tie_col].sum())
            wins = int(selection.loc[~selection[tie_col], actual_col].sum())
            losses = n_bets - wins - pushes
            profits = np.where(
                selection[tie_col],
                0.0,
                np.where(
                    selection[actual_col] == 1,
                    selection[decimal_col] - 1.0,
                    -1.0,
                ),
            )
            total_profit = float(np.sum(profits))
            denom = wins + losses
            hit_rate = float(wins / denom) if denom > 0 else np.nan
            roi = float(total_profit / n_bets)

            avg_model_prob = float(selection[prob_col].mean())
            avg_implied = float(selection[implied_col].mean())
            records.append(
                {
                    "bet_type": "moneyline",
                    "bet_side": side,
                    "threshold": float(threshold),
                    "n_bets": n_bets,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "hit_rate": hit_rate,
                    "avg_model_prob": avg_model_prob,
                    "avg_implied_prob": avg_implied,
                    "avg_expected_profit": float(selection["expected_profit"].mean()),
                    "avg_edge_prob": float(avg_model_prob - avg_implied),
                    "avg_odds_decimal": float(selection[decimal_col].mean()),
                    "avg_odds_american": float(selection[moneyline_col].mean()),
                    "total_profit": total_profit,
                    "roi": roi,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_spread(
    df: pd.DataFrame,
    thresholds: Iterable[float],
    sigma_margin_error: float,
) -> pd.DataFrame:
    valid = df.dropna(
        subset=[
            "spread_line",
            "pred_home_margin",
            "home_spread_odds",
            "away_spread_odds",
            "home_spread_decimal",
            "away_spread_decimal",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "bet_type",
                "bet_side",
                "threshold",
                "n_bets",
                "wins",
                "losses",
                "pushes",
                "hit_rate",
                "avg_cover_prob",
                "avg_expected_profit",
                "avg_edge_points",
                "avg_odds_decimal",
                "avg_odds_american",
                "total_profit",
                "roi",
            ]
        )

    sigma = max(sigma_margin_error, 1e-6)
    valid["home_cover_prob"] = standard_normal_cdf(valid["spread_edge_points"] / sigma)
    valid["away_cover_prob"] = standard_normal_cdf(-valid["spread_edge_points"] / sigma)
    valid["home_expected_profit"] = (
        valid["home_cover_prob"] * (valid["home_spread_decimal"] - 1.0)
        - (1.0 - valid["home_cover_prob"])
    )
    valid["away_expected_profit"] = (
        valid["away_cover_prob"] * (valid["away_spread_decimal"] - 1.0)
        - (1.0 - valid["away_cover_prob"])
    )

    records: list[dict[str, float | int | str]] = []
    for side, prob_col, decimal_col, odds_col, expected_col, cover_col in [
        ("home", "home_cover_prob", "home_spread_decimal", "home_spread_odds", "home_expected_profit", "home_cover"),
        ("away", "away_cover_prob", "away_spread_decimal", "away_spread_odds", "away_expected_profit", "away_cover"),
    ]:
        edge = valid["spread_edge_points"] if side == "home" else -valid["spread_edge_points"]
        base = valid.assign(edge_points=edge)

        for threshold in thresholds:
            selection = base[edge > threshold]
            n_bets = int(len(selection))
            if n_bets == 0:
                records.append(
                    {
                        "bet_type": "spread",
                        "bet_side": side,
                        "threshold": float(threshold),
                        "n_bets": 0,
                        "wins": 0,
                        "losses": 0,
                        "pushes": 0,
                        "hit_rate": np.nan,
                        "avg_cover_prob": np.nan,
                        "avg_expected_profit": np.nan,
                        "avg_edge_points": np.nan,
                        "avg_odds_decimal": np.nan,
                        "avg_odds_american": np.nan,
                        "total_profit": 0.0,
                        "roi": np.nan,
                    }
                )
                continue

            pushes = int(selection["spread_push"].sum())
            wins = int(selection.loc[~selection["spread_push"], cover_col].sum())
            losses = n_bets - wins - pushes
            profits = np.where(
                selection["spread_push"],
                0.0,
                np.where(
                    selection[cover_col],
                    selection[decimal_col] - 1.0,
                    -1.0,
                ),
            )
            total_profit = float(np.sum(profits))
            denom = wins + losses
            hit_rate = float(wins / denom) if denom > 0 else np.nan

            records.append(
                {
                    "bet_type": "spread",
                    "bet_side": side,
                    "threshold": float(threshold),
                    "n_bets": n_bets,
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "hit_rate": hit_rate,
                    "avg_cover_prob": float(selection[prob_col].mean()),
                    "avg_expected_profit": float(selection[expected_col].mean()),
                    "avg_edge_points": float(selection["edge_points"].mean()),
                    "avg_odds_decimal": float(selection[decimal_col].mean()),
                    "avg_odds_american": float(selection[odds_col].mean()),
                    "total_profit": total_profit,
                    "roi": float(total_profit / n_bets),
                }
            )
    return pd.DataFrame.from_records(records)


def run(args: argparse.Namespace) -> None:
    predictions = load_predictions(args.start_season, args.end_season)
    if args.debug:
        print(
            f"[market_roi] Loaded {len(predictions)} evaluated games "
            f"({predictions['season'].min()}-{predictions['season'].max()})"
        )
    dataset = prepare_dataset(predictions)
    if args.debug:
        missing = len(predictions) - len(dataset)
        print(f"[market_roi] Prepared dataset with {len(dataset)} games ({missing} missing odds filtered out)")

    sigma = float(np.sqrt(np.mean(np.square(dataset["margin_error"])))) if not dataset.empty else float("nan")
    if args.debug and not np.isnan(sigma):
        print(f"[market_roi] Estimated margin RMSE sigma={sigma:.3f}")

    moneyline_df = summarize_moneyline(dataset, args.moneyline_thresholds)
    spread_df = summarize_spread(dataset, args.spread_thresholds, sigma)

    moneyline_path = ANALYSIS_DIR / "market_roi_moneyline.csv"
    spread_path = ANALYSIS_DIR / "market_roi_spread.csv"
    summary_path = ANALYSIS_DIR / "market_roi_summary.json"

    moneyline_df.sort_values(["bet_side", "threshold"]).to_csv(moneyline_path, index=False)
    spread_df.sort_values(["bet_side", "threshold"]).to_csv(spread_path, index=False)

    summary: dict[str, dict[str, float | str]] = {
        "settings": {
            "start_season": args.start_season,
            "end_season": args.end_season,
            "moneyline_thresholds": list(map(float, args.moneyline_thresholds)),
            "spread_thresholds": list(map(float, args.spread_thresholds)),
            "margin_error_sigma": float(sigma),
        },
        "moneyline_best": {},
        "spread_best": {},
    }
    valid_ml = moneyline_df[(moneyline_df["n_bets"] > 0) & moneyline_df["roi"].notna()]
    if not valid_ml.empty:
        best_ml = valid_ml.loc[valid_ml["roi"].idxmax()]
        summary["moneyline_best"] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_ml.items()
        }
    valid_spread = spread_df[(spread_df["n_bets"] > 0) & spread_df["roi"].notna()]
    if not valid_spread.empty:
        best_spread = valid_spread.loc[valid_spread["roi"].idxmax()]
        summary["spread_best"] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_spread.items()
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Moneyline ROI summary ===")
    if moneyline_df.empty:
        print("No moneyline bets matched the criteria.")
    else:
        display_cols = [
            "bet_side",
            "threshold",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "avg_model_prob",
            "avg_implied_prob",
            "avg_expected_profit",
        ]
        print(moneyline_df[display_cols].sort_values(["bet_side", "threshold"]).to_string(index=False))

    print("\n=== Spread ROI summary ===")
    if spread_df.empty:
        print("No spread bets matched the criteria.")
    else:
        display_cols = [
            "bet_side",
            "threshold",
            "n_bets",
            "wins",
            "losses",
            "pushes",
            "roi",
            "avg_cover_prob",
            "avg_expected_profit",
            "avg_edge_points",
        ]
        print(spread_df[display_cols].sort_values(["bet_side", "threshold"]).to_string(index=False))

    print(f"\nMoneyline results written to {moneyline_path.resolve()}")
    print(f"Spread results written to {spread_path.resolve()}")
    print(f"Summary written to {summary_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate market ROI for model predictions.")
    parser.add_argument("--start-season", type=int, default=2016, help="Lower bound season for analysis (inclusive).")
    parser.add_argument("--end-season", type=int, default=None, help="Upper bound season for analysis (inclusive).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--moneyline-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05],
        help="Expected value thresholds (per unit) for moneyline bets.",
    )
    parser.add_argument(
        "--spread-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        help="Predicted margin edge thresholds (points) for spread bets.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
