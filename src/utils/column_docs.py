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

"""Common column descriptions for Streamlit display."""

COL_DOCS = {
    "season": "Season year",
    "week": "NFL week number",
    "matchup": "Format: Away team at Home team",
    "home_team": "Home team abbreviation",
    "away_team": "Away team abbreviation",
    "home_win_prob": "Model-predicted home win probability",
    "pred_home_margin": "Predicted home margin (positive favors home)",
    "pred_home_margin_lo": "Lower bound of predicted margin interval",
    "pred_home_margin_hi": "Upper bound of predicted margin interval",
    "news_count7_home": "News/injury keyword hits for the home team (last 7 days)",
    "news_count7_away": "News/injury keyword hits for the away team (last 7 days)",
    "inj_out_home": "Home players currently ruled out",
    "inj_out_away": "Away players currently ruled out",
    "weather_temp_kickoff_f": "Kickoff temperature in Fahrenheit",
    "weather_wind_speed_mph": "Wind speed at kickoff (mph)",
    "offense_team": "Team whose offense is highlighted",
    "defense_team": "Opponent defense facing that offense",
    "mismatch_score": "Composite offense-versus-defense mismatch score (higher is better for offense)",
    "offense_signal": "Aggregated offensive signal based on multiple metrics",
    "defense_vulnerability": "Aggregated defensive weakness signal (higher means more vulnerable)",
    "win_prob": "Win probability for the highlighted offense's team",
    "predicted_margin": "Predicted margin for the highlighted offense's team",
    "news_7d": "News/injury hits impacting the highlighted offense's team over 7 days",
    "injury_count": "Injury burden (out/doubtful/questionable players) for highlighted offense",
    "run_mismatch": "Composite rushing mismatch score",
    "rush_signal": "Aggregated rushing offense signal",
    "defensive_rush_vulnerability": "Rush defense vulnerability signal",
    "top_rush_metric": "Representative rushing metric (yards, attempts, etc.)",
    "rush_allowed_metric": "Representative rushing allowed metric",
    "receiving_mismatch": "Composite receiving mismatch score",
    "receiving_signal": "Aggregated passing/receiving offense signal",
    "def_pass_vulnerability": "Pass defense vulnerability signal",
    "top_receiving_metric": "Representative receiving metric (yards, targets, etc.)",
    "coverage_allowed_metric": "Representative coverage/receiving yards allowed metric",
    "favorite_team": "Model favourite in the matchup",
    "opponent": "Underdog opponent",
    "favorite_win_prob": "Favourite's win probability",
    "favorite_margin": "Favourite's predicted margin",
    "injury_pressure": "Injury burden on the favourite",
    "news_pressure": "News/injury keyword hits impacting the favourite",
    "depth_pressure": "Depth chart/snap stress indicator for the favourite",
    "weather_penalty": "Weather-based risk penalty",
    "risk_score": "Composite upset risk score",
    "home_news_hits": "News/injury hits impacting the home team",
    "away_news_hits": "News/injury hits impacting the away team",
    "home_injuries": "Injury burden for the home team",
    "away_injuries": "Injury burden for the away team",
    "combined_pressure": "Combined availability/news pressure on both teams",
    "weather_temp": "Kickoff temperature (F)",
    "weather_wind": "Wind speed (mph) or descriptive label",
    "weather_notes": "Additional weather details (joined from multiple columns)",
}
