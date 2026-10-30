"""Batch vs online: skor API untuk klaim nyata (test period) harus identik dengan skor batch."""
import numpy as np
import pandas as pd
import pytest

from jaga_claim.api.mapping import curated_row_to_request
from jaga_claim.api.schemas import ScoreRequest
from jaga_claim.api.service import ScoringService
from jaga_claim.config import resolve


@pytest.fixture(scope="module")
def svc():
    return ScoringService()


def test_batch_online_identical(svc, cfg):
    S = pd.read_parquet(resolve("data/scored/scores_all.parquet"), columns=["claim_id", "split", "risk_score", "risk_band"])
    te = S[S["split"] == "test"]
    rng = np.random.default_rng(0)
    pick = pd.concat([te[te["risk_band"] == "high"].sample(15, random_state=1),
                      te[te["risk_band"] == "medium"].sample(15, random_state=1),
                      te[te["risk_band"] == "low"].sample(20, random_state=1)])
    hs = svc.history.set_index("claim_id")
    for _, s in pick.iterrows():
        row = hs.loc[s["claim_id"]].copy(); row["claim_id"] = s["claim_id"]
        out = svc.score(ScoreRequest(**curated_row_to_request(row)))
        assert abs(out["risk_score"] - round(s["risk_score"], 4)) < 1e-4, (s["claim_id"], out["risk_score"], s["risk_score"])
        assert out["risk_band"] == s["risk_band"]
