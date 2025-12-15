# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}

REGULAR_SEASON_URL = "https://en.wikipedia.org/wiki/NFL_regular_season"
SEASON_HISTORY_URL = "https://en.wikipedia.org/wiki/List_of_NFL_seasons"

REFERENCE_DIR = Path("data/reference")
REGULAR_SEASON_PATH = REFERENCE_DIR / "nfl_regular_season_games.csv"
SEASON_HISTORY_PATH = REFERENCE_DIR / "nfl_seasons_history.csv"


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("\xa0", " ").strip()
    for ch in ("–", "—", "−", "�"):
        text = text.replace(ch, "-")
    return text


def fetch_regular_season_games() -> pd.DataFrame:
    html = _fetch_html(REGULAR_SEASON_URL)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError("Unable to parse regular season games table.")
    table = tables[0].copy()
    table.columns = ["season_range", "games_desc"]
    table["season_range"] = table["season_range"].map(_clean_text)
    table["games_desc"] = table["games_desc"].map(_clean_text)

    start_years: list[pd.Series] = []
    end_years: list[pd.Series] = []
    for rng in table["season_range"]:
        parts = [p.strip() for p in rng.split("-") if p.strip()]
        if not parts:
            start_years.append(pd.NA)
            end_years.append(pd.NA)
            continue
        start_part = parts[0]
        end_part = parts[-1] if len(parts) > 1 else parts[0]
        start_match = re.search(r"(\d{4})", start_part)
        end_match = re.search(r"(\d{4})", end_part)
        start_years.append(int(start_match.group(1)) if start_match else pd.NA)
        if re.search(r"present", end_part, flags=re.IGNORECASE):
            end_years.append(pd.NA)
        else:
            end_years.append(int(end_match.group(1)) if end_match else pd.NA)

    table["season_start"] = pd.Series(start_years, dtype="Int64")
    table["season_end"] = pd.Series(end_years, dtype="Int64")
    table["games_per_team"] = (
        table["games_desc"].str.extract(r"(\d+)\s*games", expand=False).astype("Int64")
    )
    table["schedule_weeks"] = (
        table["games_desc"].str.extract(r"(\d+)\s*weeks", expand=False).astype("Int64")
    )
    table["notes"] = table["games_desc"].str.extract(r"\(([^)]*)\)", expand=False).fillna("").str.strip()
    return table[["season_range", "season_start", "season_end", "games_per_team", "schedule_weeks", "notes"]]


@dataclass
class SeasonRecord:
    season: int
    teams: int | pd.NA
    champion: str | pd.NA


def _extract_year(text: str) -> int:
    match = re.search(r"(\d{4})", str(text))
    if not match:
        raise ValueError(f"Unable to find year in: {text!r}")
    return int(match.group(1))


def _extract_int(text: str | float | int) -> int | pd.NA:
    if pd.isna(text):
        return pd.NA
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else pd.NA


def _clean_champion(value: object) -> str | pd.NA:
    text = _clean_text(value)
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    return text if text else pd.NA


def fetch_season_history() -> pd.DataFrame:
    html = _fetch_html(SEASON_HISTORY_URL)
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise RuntimeError("Unable to parse season history tables.")

    records: list[SeasonRecord] = []

    # Table 0: 1920-1932
    if len(tables) > 0:
        df = tables[0].copy()
        df.columns = ["season", "teams", "champion", "ref"]
        for _, row in df.iterrows():
            records.append(
                SeasonRecord(
                    season=_extract_year(row["season"]),
                    teams=_extract_int(row["teams"]),
                    champion=_clean_champion(row["champion"]),
                )
            )

    # Table 1: 1933-1959
    if len(tables) > 1:
        df = tables[1].copy()
        df.columns = ["season", "teams", "east", "west", "year", "champion", "ref"]
        for _, row in df.iterrows():
            records.append(
                SeasonRecord(
                    season=_extract_year(row["season"]),
                    teams=_extract_int(row["teams"]),
                    champion=_clean_champion(row["champion"]),
                )
            )

    # Table 2: 1960-1969 (NFL/AFL split)
    if len(tables) > 2:
        df = tables[2].copy()
        df.columns = [
            "season",
            "teams",
            "nfl_east",
            "nfl_west",
            "nfl_year",
            "nfl_champion",
            "afl_season",
            "afl_teams",
            "afl_east",
            "afl_west",
            "afl_year",
            "afl_champion",
            "sb_game",
            "sb_champion",
            "ref",
        ]
        for _, row in df.iterrows():
            champion = _clean_champion(row["sb_champion"]) or _clean_champion(row["nfl_champion"])
            records.append(
                SeasonRecord(
                    season=_extract_year(row["season"]),
                    teams=_extract_int(row["teams"]),
                    champion=champion,
                )
            )

    # Table 3: 1970-present (post-merger)
    if len(tables) > 3:
        df = tables[3].copy()
        df.columns = [
            "season",
            "teams",
            "games",
            "afc_top_seed",
            "nfc_top_seed",
            "postseason",
            "afc_champ",
            "nfc_champ",
            "sb_game",
            "sb_champion",
            "ref",
        ]
        for _, row in df.iterrows():
            records.append(
                SeasonRecord(
                    season=_extract_year(row["season"]),
                    teams=_extract_int(row["teams"]),
                    champion=_clean_champion(row["sb_champion"]),
                )
            )

    if not records:
        raise RuntimeError("No season records were parsed.")

    df = pd.DataFrame(records).drop_duplicates(subset="season", keep="last").sort_values("season")
    df["teams"] = df["teams"].astype("Int64")
    return df.reset_index(drop=True)


def _write_if_changed(path: Path, df: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = df.copy()
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        norm = frame.copy()
        for col in norm.columns:
            norm[col] = norm[col].apply(lambda x: "" if pd.isna(x) else str(x))
        return norm
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) == list(serialized.columns) and _normalize(existing).equals(
            _normalize(serialized)
        ):
            print(f"[nfl_reference] No changes for {path}")
            return False
    serialized.to_csv(path, index=False)
    print(f"[nfl_reference] Updated {path} ({len(serialized)} rows)")
    return True


def update_references(*, skip_history: bool = False, skip_regular: bool = False) -> None:
    changed = False
    if not skip_regular:
        regular_df = fetch_regular_season_games()
        changed |= _write_if_changed(REGULAR_SEASON_PATH, regular_df)
    if not skip_history:
        history_df = fetch_season_history()
        changed |= _write_if_changed(SEASON_HISTORY_PATH, history_df)
    if not changed:
        print("[nfl_reference] Reference files already up-to-date.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh NFL regular season and season history reference CSVs.")
    parser.add_argument("--skip-history", action="store_true", help="Only refresh games-per-season metadata.")
    parser.add_argument("--skip-regular", action="store_true", help="Only refresh season history metadata.")
    parser.add_argument("--debug", action="store_true", help="Log progress details.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.debug:
        print("[nfl_reference] refreshing regular season reference files...")
    update_references(skip_history=args.skip_history, skip_regular=args.skip_regular)


if __name__ == "__main__":
    main()
