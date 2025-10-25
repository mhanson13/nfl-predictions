from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, nargs="+", required=True)
    ap.add_argument("--season_type", type=int, choices=[2, 3], default=2)
    ap.add_argument("--debug", action="store_true")
    ap.parse_args()
    print("[espn_team_defense] Removed: using PBP-derived defensive metrics instead.")


if __name__ == "__main__":
    main()
