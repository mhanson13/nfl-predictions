"""Identical data must give identical probabilities in any process.

``make_feature_diffs`` once iterated a ``set`` of column names, so the ``_diff``
column order followed the per-process string hash seed (``PYTHONHASHSEED``), and
the model's column sampling (``colsample_bytree``) then trained on different
features in every run. Each check runs in a fresh interpreter with its own seed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import hashlib, numpy as np, pandas as pd
from src.models.feature_columns import make_feature_diffs, select_feature_columns
from src.models.win_model import build_win_classifier

rng = np.random.default_rng(7)
n = 400
cols = {"game_id": [f"g{i}" for i in range(n)]}
for k in range(40):
    cols[f"sched_stat{k:02d}_home"] = rng.normal(size=n)
    cols[f"sched_stat{k:02d}_away"] = rng.normal(size=n)
df = pd.DataFrame(cols)
signal = df["sched_stat03_home"] - df["sched_stat03_away"] + 0.5 * (df["sched_stat17_home"] - df["sched_stat17_away"])
y = (signal + rng.normal(scale=1.0, size=n) > 0).astype(int).to_numpy()

feats = make_feature_diffs(df)
names = select_feature_columns(feats)
X = feats[names].to_numpy(dtype=float)
model, kind = build_win_classifier({"model_type": "xgb", "xgb_n_estimators": 60, "xgb_colsample_bytree": 0.34,
                                    "xgb_subsample": 0.8, "xgb_max_depth": 3})
model.fit(X[:300], y[:300])
p = model.predict_proba(X[300:])[:, 1]
print(kind, len(names), hashlib.sha1(",".join(names).encode()).hexdigest()[:12],
      hashlib.sha1(np.round(p, 10).tobytes()).hexdigest()[:16])
"""


def _run(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_probabilities_identical_across_hash_seeds():
    runs = {seed: _run(seed) for seed in ("1", "2", "3")}
    assert len(set(runs.values())) == 1, runs
