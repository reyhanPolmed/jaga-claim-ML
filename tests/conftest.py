import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaga_claim.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_claim(**kw):
    """Klaim sintetis minimal dengan kolom curated (nilai default wajar, rawat jalan)."""
    base = dict(claim_id="CLM_T1", participant_key="PST_T1", admission_date=pd.Timestamp("2024-11-05"),
                discharge_date=pd.Timestamp("2024-11-05"), service_level="RJTL", provider_type="3", provider_region="31",
                provider_kind="1", provider_ownership="3", care_class="3", tariff_regional="1", service_unit="1",
                segment="2", discharge_status="1", admission_icd10="I10", admission_icd10_3="I10", primary_icd10="I10",
                primary_icd10_3="I10", inacbg_code="Q-5-44-0", cmg="17", case_type="5", specificity="44", severity="1",
                procedure_text=pd.NA, sec_dx="", sec_dx3="", cbg_tariff=200000.0, billed_cost=200000.0, sex="M",
                birth_date=pd.Timestamp("1970-01-01"))
    for p in ["sa", "sp", "rr", "si", "sd"]:
        base[f"{p}_code"] = pd.NA
        base[f"{p}_tariff"] = 0.0
    base.update(kw)
    return base


def frame(rows):
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c in ("admission_date", "discharge_date", "birth_date"):
            df[c] = pd.to_datetime(df[c])
        elif c in ("billed_cost", "cbg_tariff") or c.endswith("_tariff"):
            df[c] = pd.to_numeric(df[c])
        else:
            df[c] = df[c].astype("string")
    return df
