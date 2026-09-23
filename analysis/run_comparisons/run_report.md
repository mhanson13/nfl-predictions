# Run Comparison Report
Generated: 2026-09-22 21:15:25 UTC

## Run Leaderboard
_Using 8 current-schema evaluation runs from 236 total metric rows._

 run_id    model_name                       created_at  n_samples      acc      auc    brier  logloss       mae      rmse
run_235 unknown_model 2026-09-22 21:15:21.746944+00:00     2251.0 0.650822 0.708779 0.217524 0.625350 10.215540 13.257166
run_232 unknown_model 2026-09-21 21:29:17.072883+00:00     2250.0 0.650222 0.707102 0.218334 0.627186 10.234063 13.273578
run_229 unknown_model 2026-09-21 21:08:46.965197+00:00     2250.0 0.648889 0.706475 0.218239 0.626864 10.221940 13.270975
run_226 unknown_model 2026-09-21 20:39:19.503695+00:00     2250.0 0.653778 0.707110 0.218094 0.626595 10.234992 13.274004
run_221 unknown_model 2026-09-21 19:43:27.265878+00:00     2250.0 0.651556 0.707310 0.218478 0.627529 10.235117 13.288476
run_218 unknown_model 2026-09-21 19:24:18.281882+00:00     2235.0 0.650112 0.707993 0.217389 0.624920 10.212232 13.253548
run_215 unknown_model 2026-09-21 18:55:16.719213+00:00     2235.0 0.651454 0.707055 0.219074 0.628962 10.265729 13.335960
run_212 unknown_model 2026-09-20 16:19:12.457238+00:00     2235.0 0.647427 0.708214 0.218857 0.628501 10.253754 13.320968

## Metric Leaders
### AUC leaders
 run_id    model_name                       created_at      auc
run_235 unknown_model 2026-09-22 21:15:21.746944+00:00 0.708779
run_212 unknown_model 2026-09-20 16:19:12.457238+00:00 0.708214
run_218 unknown_model 2026-09-21 19:24:18.281882+00:00 0.707993
run_221 unknown_model 2026-09-21 19:43:27.265878+00:00 0.707310
run_226 unknown_model 2026-09-21 20:39:19.503695+00:00 0.707110

### Brier leaders
 run_id    model_name                       created_at    brier
run_218 unknown_model 2026-09-21 19:24:18.281882+00:00 0.217389
run_235 unknown_model 2026-09-22 21:15:21.746944+00:00 0.217524
run_226 unknown_model 2026-09-21 20:39:19.503695+00:00 0.218094
run_229 unknown_model 2026-09-21 21:08:46.965197+00:00 0.218239
run_232 unknown_model 2026-09-21 21:29:17.072883+00:00 0.218334

### LogLoss leaders
 run_id    model_name                       created_at  logloss
run_218 unknown_model 2026-09-21 19:24:18.281882+00:00 0.624920
run_235 unknown_model 2026-09-22 21:15:21.746944+00:00 0.625350
run_226 unknown_model 2026-09-21 20:39:19.503695+00:00 0.626595
run_229 unknown_model 2026-09-21 21:08:46.965197+00:00 0.626864
run_232 unknown_model 2026-09-21 21:29:17.072883+00:00 0.627186

### Composite score (0.4*AUC - 0.3*Brier - 0.3*LogLoss)
 run_id    model_name                       created_at  combined_score
run_235 unknown_model 2026-09-22 21:15:21.746944+00:00        0.030649
run_218 unknown_model 2026-09-21 19:24:18.281882+00:00        0.030504
run_226 unknown_model 2026-09-21 20:39:19.503695+00:00        0.029437
run_232 unknown_model 2026-09-21 21:29:17.072883+00:00        0.029185
run_221 unknown_model 2026-09-21 19:43:27.265878+00:00        0.029122

## Highlights
Best run so far: **unknown_model (run_235)** on 2026-09-22. AUC=0.709, Brier=0.218, LogLoss=0.625, Accuracy=0.651, n=2251.
It edges the previous leader (unknown_model / run_212) by +0.001 AUC and +0.001 Brier.

## Metric Trends
![](./auc_over_time.png)
![](./brier_over_time.png)
![](./logloss_over_time.png)
