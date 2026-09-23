from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.utils.io import PROC_DIR, RAW_DIR, read_df, write_df
from src.utils.teams import normalize_team_abbr
from src.utils.week_filter import filter_before_week


DEFAULT_SOURCE_PATH = RAW_DIR / "nfl_player_stats.parquet"
DEFAULT_SCHEDULE_PATH = RAW_DIR / "nfl_schedules.parquet"
DEFAULT_OUTPUT_PATH = PROC_DIR / "player_prop_labels.parquet"


@dataclass(frozen=True)
class MarketSpec:
    market: str
    target_col: str
    position_groups: tuple[str, ...]
    value_type: str


MARKETS: tuple[MarketSpec, ...] = (
    MarketSpec("qb_passing_yards", "passing_yards", ("QB",), "yards"),
    MarketSpec("rb_rushing_yards", "rushing_yards", ("RB",), "yards"),
    MarketSpec("wrte_receiving_yards", "receiving_yards", ("WR", "TE"), "yards"),
    MarketSpec("def_sacks", "def_sacks", ("DB", "DL", "LB"), "count"),
)

SOURCE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "position_group",
    "recent_team",
    "team",
    "opponent_team",
    "season_type",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "def_sacks",
]

OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "player_id",
    "player_name",
    "position",
    "position_group",
    "market",
    "actual_value",
    "actual_over_zero",
    "played_flag",
    "active_flag",
    "value_type",
    "source",
]


def _season_type_is_regular(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.eq(2) | text.isin({"REG", "REGULAR", "REGULAR_SEASON", "REGULAR SEASON"})


def _clean_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN", "NA"}:
        return None
    try:
        return normalize_team_abbr(text)
    except Exception:
        return text


def _load_player_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Player stats source not found: {path}")
    frame = read_df(path)
    columns = [col for col in SOURCE_COLUMNS if col in frame.columns]
    if "season" not in frame.columns or "week" not in frame.columns:
        raise ValueError(f"Player stats source missing season/week columns: {path}")
    return frame[columns].copy()


def _load_schedule(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    columns = ["season", "week", "game_id", "home_team", "away_team"]
    try:
        return read_df(path, columns=columns)
    except Exception:
        return read_df(path)[columns]


def _game_id_lookup(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=["season", "week", "team", "opponent", "game_id_lookup"])

    required = {"season", "week", "game_id", "home_team", "away_team"}
    if missing := required - set(schedule.columns):
        raise ValueError(f"Schedule data missing required columns: {sorted(missing)}")

    sched = schedule[list(required)].copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched = sched[sched["season"].notna() & sched["week"].notna() & sched["game_id"].notna()]
    sched["season"] = sched["season"].astype(int)
    sched["week"] = sched["week"].astype(int)
    sched["home_team"] = sched["home_team"].apply(_clean_team)
    sched["away_team"] = sched["away_team"].apply(_clean_team)

    home = sched.rename(columns={"home_team": "team", "away_team": "opponent", "game_id": "game_id_lookup"})
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent", "game_id": "game_id_lookup"})
    lookup = pd.concat(
        [
            home[["season", "week", "team", "opponent", "game_id_lookup"]],
            away[["season", "week", "team", "opponent", "game_id_lookup"]],
        ],
        ignore_index=True,
    )
    lookup = lookup.dropna(subset=["season", "week", "team", "opponent", "game_id_lookup"])
    return lookup.drop_duplicates(subset=["season", "week", "team", "opponent"], keep="last")


def _base_frame(stats: pd.DataFrame, seasons: Sequence[int] | None) -> pd.DataFrame:
    frame = stats.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame[frame["season"].notna() & frame["week"].notna()]
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)

    if seasons:
        season_set = {int(season) for season in seasons}
        frame = frame[frame["season"].isin(season_set)]

    if "season_type" in frame.columns:
        frame = frame[_season_type_is_regular(frame["season_type"])]

    if "recent_team" not in frame.columns and "team" in frame.columns:
        frame["recent_team"] = frame["team"]
    if "team" not in frame.columns and "recent_team" in frame.columns:
        frame["team"] = frame["recent_team"]
    if "opponent_team" not in frame.columns:
        frame["opponent_team"] = pd.NA

    if "player_name" not in frame.columns and "player_display_name" in frame.columns:
        frame["player_name"] = frame["player_display_name"]
    elif "player_display_name" in frame.columns:
        frame["player_name"] = frame["player_name"].where(frame["player_name"].notna(), frame["player_display_name"])

    for col in ["player_id", "player_name", "position", "position_group", "recent_team", "opponent_team"]:
        if col not in frame.columns:
            frame[col] = pd.NA

    frame["team"] = frame["recent_team"].apply(_clean_team)
    frame["opponent"] = frame["opponent_team"].apply(_clean_team)
    frame["position_group"] = frame["position_group"].astype("string").str.upper()
    frame["position"] = frame["position"].astype("string").str.upper()
    frame = frame[frame["team"].notna() & frame["opponent"].notna()]
    frame = frame[frame["team"] != frame["opponent"]]
    return frame


def build_player_prop_labels(
    stats: pd.DataFrame,
    *,
    seasons: Sequence[int] | None = None,
    markets: Sequence[str] | None = None,
    schedule: pd.DataFrame | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    base = _base_frame(stats, seasons)
    base = filter_before_week(base, exclude_from_season, exclude_from_week)
    lookup = _game_id_lookup(schedule if schedule is not None else pd.DataFrame())
    if not lookup.empty:
        base = base.merge(lookup, on=["season", "week", "team", "opponent"], how="left")
        if "game_id" not in base.columns:
            base["game_id"] = pd.NA
        game_id = base["game_id"].astype("string")
        missing_game_id = game_id.isna() | game_id.str.strip().isin(["", "None", "nan", "<NA"])
        base.loc[missing_game_id, "game_id"] = base.loc[missing_game_id, "game_id_lookup"]
        base = base.drop(columns=["game_id_lookup"], errors="ignore")
    requested = {market.lower() for market in markets} if markets else {spec.market for spec in MARKETS}
    records: list[pd.DataFrame] = []

    for spec in MARKETS:
        if spec.market not in requested:
            continue
        if spec.target_col not in base.columns:
            continue

        subset = base[base["position_group"].isin(spec.position_groups)].copy()
        if subset.empty:
            continue

        subset["actual_value"] = pd.to_numeric(subset[spec.target_col], errors="coerce")
        subset = subset[subset["actual_value"].notna()]
        if subset.empty:
            continue

        subset["market"] = spec.market
        subset["actual_over_zero"] = subset["actual_value"] > 0
        subset["played_flag"] = True
        subset["active_flag"] = True
        subset["value_type"] = spec.value_type
        subset["source"] = "nfl_player_stats"
        records.append(
            subset[
                [
                    "season",
                    "week",
                    "game_id",
                    "team",
                    "opponent",
                    "player_id",
                    "player_name",
                    "position",
                    "position_group",
                    "market",
                    "actual_value",
                    "actual_over_zero",
                    "played_flag",
                    "active_flag",
                    "value_type",
                    "source",
                ]
            ]
        )

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    labels = pd.concat(records, ignore_index=True)
    labels = labels.dropna(subset=["season", "week", "player_id", "market"])
    labels = labels.sort_values(["season", "week", "game_id", "team", "market", "player_id"], na_position="last")
    labels = labels.drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"], keep="last")
    return labels[OUTPUT_COLUMNS]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized player prop labels from completed player game stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Seasons to include.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="Source player stats parquet.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH, help="Schedule file used to fill game_id.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output label parquet.")
    parser.add_argument("--exclude-from-season", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument("--exclude-from-week", type=int, default=None, help="Exclude this season/week and later rows.")
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=[spec.market for spec in MARKETS],
        default=None,
        help="Optional subset of markets to build.",
    )
    parser.add_argument("--debug", action="store_true", help="Print row counts by market.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = _load_player_stats(args.source)
    schedule = _load_schedule(args.schedule)
    labels = build_player_prop_labels(
        stats,
        seasons=args.seasons,
        markets=args.markets,
        schedule=schedule,
        exclude_from_season=args.exclude_from_season,
        exclude_from_week=args.exclude_from_week,
    )
    if labels.empty:
        raise RuntimeError("No player prop labels were produced.")

    written = write_df(labels, args.output)
    if args.debug:
        counts = labels.groupby("market").size().sort_index()
        print("[player_props.labels] row counts by market:")
        for market, count in counts.items():
            print(f"  {market}: {count}")
    print(f"[player_props.labels] wrote {len(labels)} rows -> {written.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
