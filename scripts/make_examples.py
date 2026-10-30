"""Membuat contoh request/response (fixture kontrak) dari data ter-pseudonim.

    PYTHONPATH=src python scripts/make_examples.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaga_claim.api.mapping import curated_row_to_request  # noqa: E402
from jaga_claim.config import REASON_CODES, resolve  # noqa: E402

EX = ROOT / "examples"
EX.mkdir(exist_ok=True)


def dump(name, obj):
    (EX / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print("->", EX / name)


hs = pd.read_parquet(resolve("data/features/history_store.parquet"))
rc_cols = [f"rc_{code}" for code in REASON_CODES]
S = pd.read_parquet(resolve("data/scored/scores_all.parquet"), columns=["claim_id", "split", "risk_band", "risk_score",
                                                                         *rc_cols])
hs = hs.merge(S, on="claim_id")
te = hs[hs["split"] == "test"]

# 1) Klaim nyata periode test dengan band low (tanpa flag) -> diubah menjadi peserta BARU agar bebas riwayat
low = te[(te["risk_band"] == "low") & ~te[rc_cols].any(axis=1) & (te["inacbg_code"] == "Q-5-44-0") &
         (te["sec_dx"] != "")].iloc[0].copy()
low["claim_id"], low["participant_key"] = "CLM_demo_low_001", "PST_demo_new_001"
dump("request_low_risk.json", curated_row_to_request(low))

# 2) Duplikat persis dari klaim yang sudah ada di history store (ber-flag POSSIBLE_DUPLICATE)
src = te[(te["risk_band"] == "low") & (te["service_level"] == "RITL")].iloc[0].copy()
src["claim_id"] = "CLM_zz_demo_dup_001"
dump("request_flagged_duplicate.json", curated_row_to_request(src))

# 3) Klaim nyata high-band (COST_PEER_OUTLIER) sebagaimana adanya
hi = te[(te["risk_band"] == "high") & te["rc_COST_PEER_OUTLIER"]].sort_values("risk_score").iloc[-1]
dump("request_high_real.json", curated_row_to_request(hi))

# 4) Contoh persis dari API Contract (nilai DEMO)
dump("request_contract_demo.json", {
    "claim_id": "hash-demo-FKL02-001", "participant_key": "hash-demo-PSTV01-001",
    "prediction_time": "2026-10-24T03:00:00Z",
    "episode": {"admission_date": "2026-10-10", "discharge_date": "2026-10-22", "provider_type": "DEMO_TYPE",
                "provider_region": "DEMO_REGION", "care_class": "3"},
    "diagnoses": {"primary_icd10": "A01.0", "secondary_icd10": []},
    "casemix": {"inacbg_code": "DEMO-CBG", "severity": "DEMO"},
    "procedures": {"procedure_type": "DEMO"},
    "financial_pre_submit": {"billed_cost": 15000000},
    "history_features": {}, "peer_features": {}})

# 5) Request invalid (primary_icd10 hilang + FKL48 dikirim)
bad = curated_row_to_request(low)
del bad["diagnoses"]["primary_icd10"]
bad["financial_pre_submit"]["FKL48"] = 123
dump("request_invalid.json", bad)
