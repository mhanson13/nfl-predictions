# Week 11 post-merge check (PR #17)

**Result: ALERT**

Generated 2026-11-20 14:54 UTC by auto_compare.sh. Run log: `C:\Code\nfl-predictions\logs\pipeline\2026\output_wk11_in_progress_20261120-0658_debug.txt`.

## 1. Post-merge check
- Same data, current code vs pre-merge code (`--before` = sandbox build): exit 1.
- Model inputs changed on 2026+ rows: 14 (expected set: 17); outside the expected 17: none.
- ALERT lines: 3
  - ALERT unexpected column changed: pstats_def_fumbles_pct_change_away (1 rows)
  - ALERT unexpected column changed: pstats_pt_downed_pct_change_home (1 rows)
  - ALERT unexpected column changed: pstats_pt_out_of_bounds_pct_change_away (2 rows)
- Pre-2026 changes vs Monday's copy (`matchup_features_pre_pr17.parquet`): 12
  - ALERT pre-2026 rows changed: qb_recovery_score_rolling3_home (7 rows)
  - ALERT pre-2026 rows changed: qb_recovery_score_rolling3_away (5 rows)
  - ALERT pre-2026 rows changed: qb_recovery_score_rolling3_diff (12 rows)
  - ALERT pre-2026 rows changed: news_count7_home (1 rows)
  - ALERT pre-2026 rows changed: news_count3_home (1 rows)
  - ALERT pre-2026 rows changed: news_rotowire7_home (1 rows)
  - ALERT pre-2026 rows changed: news_injury_kw7_home (1 rows)
  - ALERT pre-2026 rows changed: news_hours_before_game_min_home (1 rows)
  - ALERT pre-2026 rows changed: news_count7_diff (1 rows)
  - ALERT pre-2026 rows changed: news_count3_diff (1 rows)
  - ALERT pre-2026 rows changed: news_rotowire7_diff (1 rows)
  - ALERT pre-2026 rows changed: news_injury_kw7_diff (1 rows)

```
games compared: 2766 (2494 before 2026); changed columns: 1385
model inputs: 174; changed on 2026+ rows: 14
  expected: pass_epa_per_db_rolling3_diff
  expected: pass_epa_per_db_rolling5_diff
  expected: pass_epa_per_db_rolling3_diff_lg_diff
  expected: drive_epa_mean_rolling3_diff
  expected: drive_epa_first_drive_rolling3_diff
  expected: drive_epa_q4_rolling3_diff
  expected: drive_epa_red_zone_rolling3_diff
  expected: drive_epa_mean_rolling3_diff_lg_diff
  expected: off_adj_eff_rolling3_diff
  expected: def_adj_eff_rolling3_diff
  expected: rz_td_rate_for_rolling3_diff
  expected: rz_td_rate_against_rolling3_diff
  expected: sack_rate_allowed_rolling3_diff
  expected: sack_rate_generated_rolling3_diff
note: pstats_fumble_recovery_yards_own_away rose in 18 rows (signed yardage; removed negative rows)
note: pstats_fumble_recovery_yards_own_home rose in 12 rows (signed yardage; removed negative rows)
note: pstats_fumble_recovery_yards_own_prev_away rose in 9 rows (signed yardage; removed negative rows)
note: pstats_fumble_recovery_yards_own_prev_home rose in 17 rows (signed yardage; removed negative rows)
note: pstats_fumble_recovery_yards_own_rolling3_away rose in 26 rows (signed yardage; removed negative rows)
note: pstats_fumble_recovery_yards_own_rolling3_home rose in 39 rows (signed yardage; removed negative rows)
note: pstats_pt_return_yards_away rose in 1 rows (signed yardage; removed negative rows)
note: pstats_pt_return_yards_prev_home rose in 1 rows (signed yardage; removed negative rows)
note: pstats_pt_return_yards_rolling3_home rose in 1 rows (signed yardage; removed negative rows)
note: pstats_punt_return_yards_home rose in 4 rows (signed yardage; removed negative rows)
note: pstats_punt_return_yards_prev_away rose in 1 rows (signed yardage; removed negative rows)
note: pstats_punt_return_yards_prev_home rose in 3 rows (signed yardage; removed negative rows)
note: pstats_punt_return_yards_rolling3_away rose in 3 rows (signed yardage; removed negative rows)
note: pstats_punt_return_yards_rolling3_home rose in 3 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_away rose in 125 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_home rose in 127 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_prev_away rose in 116 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_prev_home rose in 109 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_rolling3_away rose in 133 rows (signed yardage; removed negative rows)
note: pstats_sack_yards_lost_rolling3_home rose in 127 rows (signed yardage; removed negative rows)
note: expected model features that did not change: ['drive_completion_rate_rolling3_diff', 'rz_trips_against_rolling3_diff', 'rz_trips_for_rolling3_diff']
ALERT unexpected column changed: pstats_def_fumbles_pct_change_away (1 rows)
ALERT unexpected column changed: pstats_pt_downed_pct_change_home (1 rows)
ALERT unexpected column changed: pstats_pt_out_of_bounds_pct_change_away (2 rows)
RESULT: ALERTS
```

## 2. Week 11 board and projections: current code vs pre-merge code, same data
- Any flips (side / recommendation / publish status / tier): YES, see below

```
qb: rows main 124 live 114 matched 114 | projection 0 line 0 prob_over 0 edge_ev 0 | flips: side 0 recommendation 0 status 0 tier 0
offense: rows main 956 live 891 matched 857 | projection 0 line 0 prob_over 0 edge_ev 0 | flips: side 0 recommendation 0 status 0 tier 0
defense: rows main 284 live 265 matched 240 | projection 1 line 0 prob_over 1 edge_ev 1 | flips: side 0 recommendation 0 status 0 tier 1
novelty: rows main 1591 live 1495 matched 1466 | projection 709 line 0 prob_over 709 edge_ev 538 | flips: side 0 recommendation 0 status 0 tier 0
public: rows main 1761 live 1761 matched 1761 | projection 529 line 0 prob_over 529 edge_ev 529 | flips: side 0 recommendation 0 status 0 tier 1
biggest public moves:
player_name_live         market  projection_main  projection_live  prob_over_main  prob_over_live
      James Cook   two_plus_tds            0.559            0.073           0.262           0.218
    Ja'Kobi Lane three_plus_tds            0.027            0.003           0.033           0.020
 Trevor Lawrence three_plus_tds            0.036            0.003           0.033           0.020
  Malachi Fields three_plus_tds            0.029            0.003           0.033           0.020
    Justice Hill three_plus_tds            0.030            0.003           0.033           0.020
  Ameer Abdullah three_plus_tds            0.045            0.003           0.033           0.020
      Joe Burrow three_plus_tds            0.073            0.003           0.033           0.020
   Dohnte Meyers three_plus_tds            0.033            0.003           0.033           0.020
projections qb: players 62/57, matched 57, changed 49, only main 5, only live 0
 player_name_live  projected_passing_yards_main  projected_passing_yards_live
     Kyler Murray                         530.2                         170.3
     Malik Willis                         438.0                         174.5
      Kyle McCord                         418.0                         209.0
Athan Kaliakmanis                         372.0                         186.0
    J.J. McCarthy                         366.0                         183.0
projections offense: players 260/240, matched 224, changed 220, only main 36, only live 16
player_name_live  projected_receiving_yards_main  projected_receiving_yards_live
    Carnell Tate                           156.3                            59.0
     Ryan Miller                           117.7                            30.7
   Caleb Douglas                           141.2                            61.2
        DJ Moore                           132.6                            53.6
 Omar Cooper Jr.                           153.0                            76.5
projections defense: players 260/240, matched 206, changed 189, only main 53, only live 33
player_name_live  projected_sacks_main  projected_sacks_live
       DJ Wonnum                   6.5                   1.5
     Ethan Burke                   6.0                   1.0
  Landon Jackson                   5.0                   1.0
    Tyree Wilson                   5.0                   1.0
  Tim Keenan III                   5.0                   1.0
```

## 3. MTB publish and deploy
- MTB commit `4194824` 2026-11-20T07:31:13-07:00: Publish NFL predictions 2026 week 11 (2026_wk11_Fri-manual_2026-11-20)
- deploy-gke.yml run **44754280494**: completed / success (created 2026-11-20T14:31:31Z) https://github.com/mhanson13/mattythebookie/actions/runs/44754280494
- Preflight saw: `deploy: completed success 2026-11-20T14:31:31Z`

## 4. No-pick ALERT
- Searched 13 files (Week 11 run logs + logs/dispatcher).
- None fired (expected: Week 11 is not graded yet).

## 5. Injury history manifest (PR #20 logging)
- `C:\Code\nfl-predictions\data\raw\history\nfl_injuries` exists; 2 version(s) saved
  - 2026-11-19T21:50:12.626789+00:00 | season 2026 | rows 3421 | injuries_2026_20261119T215012Z.parquet
  - 2026-11-20T14:03:12.588926+00:00 | season 2026 | rows 3459 | injuries_2026_20261120T140312Z.parquet

