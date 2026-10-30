"""Mekanisme feedback loop (label sintetis hanya untuk menguji alur, bukan evaluasi kinerja)."""
import numpy as np
import pandas as pd

from jaga_claim.config import resolve
from jaga_claim.feedback import analyze, load_feedback


def test_feedback_loop(tmp_path, cfg):
    S = pd.read_parquet(resolve("data/scored/scores_all.parquet")).sample(600, random_state=0)
    rng = np.random.default_rng(0)
    act = np.where(rng.random(len(S)) < 0.2 + 0.6 * S["risk_score"].to_numpy(), "confirm", "dismiss")
    fb = pd.DataFrame({"claim_id": S["claim_id"], "review_action": act, "reviewer_id": "R1",
                       "reviewed_at": "2026-10-24T03:00:00Z", "model_version": "hybrid-v1.0.0"})
    p = tmp_path / "fb.csv"; fb.to_csv(p, index=False)
    rep = analyze(load_feedback(str(p)), S, cfg)
    assert rep["n_labeled_claims"] == 600
    assert "suggested_weights" in rep["supervised_baseline"]
    assert abs(sum(rep["supervised_baseline"]["suggested_weights"].values()) - 1) < 1e-6
