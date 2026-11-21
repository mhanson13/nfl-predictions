# predictions_log

Optional prediction snapshots that weekly grading (`src.evaluation.live_results`) can grade from.

For each game, grading uses the first source that has it:

1. the picks-of-record ledger (`predictions/picks_of_record/`);
2. `predictions_log/<season>/week_NN_published_override.csv`: a hand-made snapshot of what MTB actually published, used when the archive CSV is known to be wrong (for example `2026/week_08_published_override.csv`);
3. the MTB season/week archive CSV;
4. `predictions_log/<season>/week_NN_predictions.csv`: a legacy locked-prediction file.

Nothing writes the legacy locked files any more. The `analysis.live_tracking` tool that wrote them was removed in the 2026-11 cleanup (see [`docs/archive/removed_code_2026-11.md`](../docs/archive/removed_code_2026-11.md)). The pick-of-record rule in [GOALS.md](../GOALS.md) decides which pick counts.
