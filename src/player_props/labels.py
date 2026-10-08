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
DEFAULT_PBP_SOURCE_PATH = RAW_DIR / "nfl_pbp.parquet"
DEFAULT_SCHEDULE_PATH = RAW_DIR / "nfl_schedules.parquet"
DEFAULT_OUTPUT_PATH = PROC_DIR / "player_prop_labels.parquet"
DEFAULT_NOVELTY_TD_OUTPUT_PATH = PROC_DIR / "player_prop_novelty_td_labels.parquet"


@dataclass(frozen=True)
class MarketSpec:
    market: str
    target_col: str
    position_groups: tuple[str, ...]
    value_type: str


MARKETS: tuple[MarketSpec, ...] = (
    MarketSpec("qb_passing_yards", "passing_yards", ("QB",), "yards"),
    MarketSpec("qb_passing_tds", "passing_tds", ("QB",), "count"),
    MarketSpec("rb_rushing_yards", "rushing_yards", ("RB",), "yards"),
    MarketSpec("rb_rushing_tds", "rushing_tds", ("RB",), "count"),
    MarketSpec("wrte_receiving_yards", "receiving_yards", ("WR", "TE"), "yards"),
    MarketSpec("wrte_receiving_tds", "receiving_tds", ("WR", "TE"), "count"),
    MarketSpec("anytime_td", "anytime_tds", ("QB", "RB", "WR", "TE"), "count"),
    MarketSpec("def_sacks", "def_sacks", ("DB", "DL", "LB"), "count"),
)

ANYTIME_TD_COLUMNS = [
    "rushing_tds",
    "receiving_tds",
    "special_teams_tds",
    "def_tds",
    "fumble_recovery_tds",
    "pt_return_tds",
]

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
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "special_teams_tds",
    "def_tds",
    "fumble_recovery_tds",
    "pt_return_tds",
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

NOVELTY_TD_MARKETS = {
    "first_td": "first_td_flag",
    "last_td": "last_td_flag",
    "two_plus_tds": "two_plus_td_flag",
    "three_plus_tds": "three_plus_td_flag",
}


def _market_debug_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "market" not in frame.columns:
        return pd.DataFrame(columns=["market", "rows", "positives", "positive_rate", "avg_actual"])
    out = frame.copy()
    if "actual_over_zero" not in out.columns:
        out["actual_over_zero"] = False
    if "actual_value" not in out.columns:
        out["actual_value"] = pd.NA
    out["actual_value"] = pd.to_numeric(out["actual_value"], errors="coerce")
    summary = (
        out.groupby("market", as_index=False)
        .agg(
            rows=("market", "size"),
            positives=("actual_over_zero", "sum"),
            avg_actual=("actual_value", "mean"),
        )
        .sort_values("market")
    )
    summary["positive_rate"] = summary["positives"] / summary["rows"]
    return summary[["market", "rows", "positives", "positive_rate", "avg_actual"]]


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


def _load_pbp(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PBP source not found: {path}")
    frame = read_df(path)
    required = {"season", "week", "game_id", "td_player_id", "touchdown"}
    if missing := required - set(frame.columns):
        raise ValueError(f"PBP source missing required columns: {sorted(missing)}")
    return frame.copy()


def _add_derived_touchdown_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    available = [col for col in ANYTIME_TD_COLUMNS if col in out.columns]
    if not available:
        out["anytime_tds"] = pd.NA
        return out

    total = pd.Series(0.0, index=out.index)
    for col in available:
        total = total.add(pd.to_numeric(out[col], errors="coerce").fillna(0.0), fill_value=0.0)
    out["anytime_tds"] = total
    return out


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
    base = _add_derived_touchdown_columns(base)
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


def _prepare_touchdown_candidate_pool(labels: pd.DataFrame | None) -> pd.DataFrame:
    if labels is None or labels.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    required = {"season", "week", "game_id", "player_id", "market"}
    if missing := required - set(labels.columns):
        raise ValueError(f"Candidate labels missing required columns: {sorted(missing)}")
    out = labels.copy()
    out = out[out["market"].astype("string").str.lower().eq("anytime_td")].copy()
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna()]
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out.drop_duplicates(subset=["season", "week", "game_id", "player_id"], keep="last")


def _touchdown_events_from_pbp(pbp: pd.DataFrame, seasons: Sequence[int] | None = None) -> pd.DataFrame:
    out = pbp.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["touchdown"] = pd.to_numeric(out["touchdown"], errors="coerce")
    out = out[out["season"].notna() & out["week"].notna() & out["touchdown"].eq(1)]
    out = out.dropna(subset=["game_id", "td_player_id"])
    if seasons:
        season_set = {int(season) for season in seasons}
        out = out[out["season"].astype(int).isin(season_set)]
    if out.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "player_id",
                "player_name",
                "td_count",
                "first_td_flag",
                "last_td_flag",
                "two_plus_td_flag",
                "three_plus_td_flag",
            ]
        )
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["player_id"] = out["td_player_id"].astype("string").str.strip()
    out["player_name"] = out.get("td_player_name", pd.Series(pd.NA, index=out.index))
    out["_td_order"] = pd.to_numeric(out.get("game_seconds_remaining", pd.Series(pd.NA, index=out.index)), errors="coerce")
    out["_td_order"] = out["_td_order"].fillna(-1)
    out["_row_order"] = range(len(out))

    sort_cols = ["game_id", "_td_order", "_row_order"]
    first = (
        out.sort_values(sort_cols, ascending=[True, False, True], kind="stable")
        .drop_duplicates("game_id", keep="first")[["game_id", "player_id"]]
        .assign(first_td_flag=True)
    )
    last = (
        out.sort_values(sort_cols, ascending=[True, True, True], kind="stable")
        .drop_duplicates("game_id", keep="first")[["game_id", "player_id"]]
        .assign(last_td_flag=True)
    )
    counts = (
        out.groupby(["season", "week", "game_id", "player_id"], as_index=False)
        .agg(td_count=("touchdown", "sum"), player_name=("player_name", "last"))
        .merge(first, on=["game_id", "player_id"], how="left")
        .merge(last, on=["game_id", "player_id"], how="left")
    )
    counts["first_td_flag"] = counts["first_td_flag"].eq(True)
    counts["last_td_flag"] = counts["last_td_flag"].eq(True)
    counts["two_plus_td_flag"] = pd.to_numeric(counts["td_count"], errors="coerce").ge(2)
    counts["three_plus_td_flag"] = pd.to_numeric(counts["td_count"], errors="coerce").ge(3)
    return counts


def build_touchdown_novelty_labels(
    pbp: pd.DataFrame,
    *,
    candidate_labels: pd.DataFrame | None = None,
    seasons: Sequence[int] | None = None,
    exclude_from_season: int | None = None,
    exclude_from_week: int | None = None,
) -> pd.DataFrame:
    events = _touchdown_events_from_pbp(pbp, seasons=seasons)
    candidates = _prepare_touchdown_candidate_pool(candidate_labels)
    if candidates.empty:
        if events.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        candidates = events.rename(columns={"player_id": "player_id"}).copy()
        for col in ["team", "opponent", "position", "position_group"]:
            candidates[col] = pd.NA
        candidates["active_flag"] = True
    else:
        if seasons:
            season_set = {int(season) for season in seasons}
            candidates = candidates[candidates["season"].isin(season_set)].copy()
    candidates = filter_before_week(candidates, exclude_from_season, exclude_from_week)
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    event_cols = [
        "season",
        "week",
        "game_id",
        "player_id",
        "td_count",
        "first_td_flag",
        "last_td_flag",
        "two_plus_td_flag",
        "three_plus_td_flag",
    ]
    joined = candidates.merge(events[event_cols], on=["season", "week", "game_id", "player_id"], how="left")
    for col in ["td_count", *NOVELTY_TD_MARKETS.values()]:
        if col not in joined.columns:
            joined[col] = 0 if col == "td_count" else False
    joined["td_count"] = pd.to_numeric(joined["td_count"], errors="coerce").fillna(0)
    for col in NOVELTY_TD_MARKETS.values():
        joined[col] = joined[col].astype("boolean").fillna(False).astype(bool)

    records: list[pd.DataFrame] = []
    for market, flag_col in NOVELTY_TD_MARKETS.items():
        subset = joined.copy()
        subset["market"] = market
        subset["actual_value"] = subset[flag_col].astype(int)
        subset["actual_over_zero"] = subset[flag_col].astype(bool)
        subset["played_flag"] = True
        subset["active_flag"] = subset.get("active_flag", pd.Series(True, index=subset.index)).fillna(True).astype(bool)
        subset["value_type"] = "event"
        subset["source"] = "nfl_pbp"
        records.append(subset[OUTPUT_COLUMNS])

    labels = pd.concat(records, ignore_index=True, sort=False)
    labels = labels.dropna(subset=["season", "week", "game_id", "player_id", "market"])
    labels = labels.sort_values(["season", "week", "game_id", "market", "player_id"], na_position="last")
    labels = labels.drop_duplicates(subset=["season", "week", "game_id", "player_id", "market"], keep="last")
    return labels[OUTPUT_COLUMNS]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized player prop labels from completed player game stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Seasons to include.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="Source player stats parquet.")
    parser.add_argument("--pbp-source", type=Path, default=DEFAULT_PBP_SOURCE_PATH, help="Source PBP parquet.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH, help="Schedule file used to fill game_id.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output label parquet.")
    parser.add_argument(
        "--novelty-td-output",
        type=Path,
        default=DEFAULT_NOVELTY_TD_OUTPUT_PATH,
        help="Optional output for PBP-derived novelty TD labels.",
    )
    parser.add_argument(
        "--include-novelty-td-labels",
        action="store_true",
        help="Also build first/last/2+/3+ TD labels from PBP into a separate output.",
    )
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
    novelty_written = None
    novelty_labels = pd.DataFrame()
    if args.include_novelty_td_labels:
        pbp = _load_pbp(args.pbp_source)
        novelty_labels = build_touchdown_novelty_labels(
            pbp,
            candidate_labels=labels,
            seasons=args.seasons,
            exclude_from_season=args.exclude_from_season,
            exclude_from_week=args.exclude_from_week,
        )
        novelty_written = write_df(novelty_labels, args.novelty_td_output)
    if args.debug:
        print("[player_props.labels] row counts by market:")
        print(_market_debug_summary(labels).to_string(index=False))
        if args.include_novelty_td_labels:
            print("[player_props.labels] novelty TD row counts by market:")
            print(_market_debug_summary(novelty_labels).to_string(index=False))
    print(f"[player_props.labels] wrote {len(labels)} rows -> {written.resolve()}")
    if novelty_written is not None:
        print(f"[player_props.labels] wrote {len(novelty_labels)} novelty TD rows -> {novelty_written.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
