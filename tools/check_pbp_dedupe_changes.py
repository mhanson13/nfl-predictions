"""Post-merge check for the PBP / player-stats dedupe (PR #17).

Compares matchup features saved just before the merge (``--before``) with the
first build after it (``--after``) and prints ALERT lines for anything the
dedupe should not have caused:

1. Any value change in a row from a season before 2026.
2. A changed column that is not in the expected-changes list.
3. A non-negative count/sum column that rises. Dedupe only removes rows, so these
   can only fall or stay. Exceptions, reported but not alerted:
   * signed yardage sums (a removed negative row raises the sum);
   * rows of teams whose ``sr_games_sampled`` rose (the refetch has a game the
     old cache lacked, e.g. a Monday night game).
4. A model input (``_diff`` feature used by the win model) that changed but is
   not one of the 17 expected model features.

Usage:
    python -m tools.check_pbp_dedupe_changes --before <pre-merge matchup_features.parquet> \
        [--after data/processed/matchup_features.parquet]

Exit code 1 if there is any ALERT.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.models.feature_columns import make_feature_diffs, select_feature_columns

DEFAULT_EXPECTED = Path("docs/checks/pbp_dedupe_expected_feature_changes.csv")
FIRST_AFFECTED_SEASON = 2026

# Columns that are rates, means, shares or differences rather than counts/sums.
NOT_A_COUNT = re.compile(
    r"(rate|pct|_per_|per_|mean|share|epa|ypp|yards_per|cpoe|pacr|racr|wopr|_long|avg|_eff|ratio|"
    r"_diff$|delta|pct_change|_lg|distance|season_type)"
)
# Count/sum columns whose source values can be negative (yards lost, own-fumble
# return yards, negative air yards...). Removing a duplicated negative row raises them.
SIGNED = re.compile(r"yards")
SIDE = re.compile(r"_(home|away)$")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.drop_duplicates("game_id", keep="last").set_index("game_id")


def _changed(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return ~((a - b).abs().le(1e-9) | (a.isna() & b.isna()))


def check(before: pd.DataFrame, after: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    alerts: list[str] = []
    notes: list[str] = []
    ids = before.index.intersection(after.index)
    before, after = before.loc[ids], after.loc[ids]
    season = pd.to_numeric(after["season"], errors="coerce")
    old = season < FIRST_AFFECTED_SEASON
    cols = [
        c for c in before.columns
        if c in after.columns and pd.api.types.is_numeric_dtype(before[c]) and pd.api.types.is_numeric_dtype(after[c])
    ]
    expected_cols = set(expected["column"])
    expected_model = set(expected.loc[expected["model_feature"].astype(bool), "column"])

    changed_cols: dict[str, pd.Series] = {}
    for c in cols:
        m = _changed(before[c], after[c])
        if m.any():
            changed_cols[c] = m
    print(f"games compared: {len(ids)} ({int(old.sum())} before {FIRST_AFFECTED_SEASON}); changed columns: {len(changed_cols)}")

    # 1. Older seasons must not move.
    for c, m in changed_cols.items():
        n = int((m & old).sum())
        if n:
            alerts.append(f"ALERT pre-{FIRST_AFFECTED_SEASON} rows changed: {c} ({n} rows)")

    # 2. Only expected columns.
    for c in sorted(set(changed_cols) - expected_cols):
        alerts.append(f"ALERT unexpected column changed: {c} ({int(changed_cols[c].sum())} rows)")

    # 3. Counts/sums only fall.
    games_rose = {}
    for side in ("home", "away"):
        g = f"sr_games_sampled_{side}"
        if g in before.columns and g in after.columns:
            games_rose[side] = pd.to_numeric(after[g], errors="coerce") > pd.to_numeric(before[g], errors="coerce")
    for c in sorted(changed_cols):
        if NOT_A_COUNT.search(c):
            continue
        rose = pd.to_numeric(after[c], errors="coerce") > pd.to_numeric(before[c], errors="coerce") + 1e-9
        rose &= ~old
        if not rose.any():
            continue
        side_match = SIDE.search(c)
        explained_by_games = games_rose.get(side_match.group(1)) if side_match else None
        if c.startswith("sr_games_sampled"):
            teams = sorted(set(after.loc[rose, f"{side_match.group(1)}_team"])) if side_match else []
            notes.append(f"note: {c} rose in {int(rose.sum())} rows (refetch has games the old cache lacked; teams {teams})")
            continue
        if explained_by_games is not None:
            unexplained = rose & ~explained_by_games
            if int((rose & explained_by_games).sum()):
                notes.append(f"note: {c} rose in {int((rose & explained_by_games).sum())} rows of teams with new games")
            rose = unexplained
        if not rose.any():
            continue
        if SIGNED.search(c):
            notes.append(f"note: {c} rose in {int(rose.sum())} rows (signed yardage; removed negative rows)")
            continue
        alerts.append(f"ALERT count/sum rose after dedupe: {c} ({int(rose.sum())} rows)")

    # 4. Model inputs: only the 17.
    recent = ~old
    fb = make_feature_diffs(before.loc[recent].reset_index())
    fa = make_feature_diffs(after.loc[recent].reset_index())
    fb = fb.loc[:, ~fb.columns.duplicated()]
    fa = fa.loc[:, ~fa.columns.duplicated()]
    model = select_feature_columns(fa)
    moved = [c for c in model if c in fb.columns and _changed(fb[c], fa[c]).any()]
    print(f"model inputs: {len(model)}; changed on {FIRST_AFFECTED_SEASON}+ rows: {len(moved)}")
    for c in moved:
        tag = "expected" if c in expected_model else "ALERT"
        line = f"  {tag}: {c}"
        print(line)
        if c not in expected_model:
            alerts.append(f"ALERT unexpected model input changed: {c}")
    missing = sorted(expected_model - set(moved))
    if missing:
        notes.append(f"note: expected model features that did not change: {missing}")

    for n in notes:
        print(n)
    return alerts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", type=Path, required=True, help="matchup_features.parquet saved before the merge")
    ap.add_argument("--after", type=Path, default=Path("data/processed/matchup_features.parquet"))
    ap.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    args = ap.parse_args(argv)
    alerts = check(_load(args.before), _load(args.after), pd.read_csv(args.expected))
    for a in alerts:
        print(a)
    print("RESULT:", "ALERTS" if alerts else "OK (only expected dedupe changes)")
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
