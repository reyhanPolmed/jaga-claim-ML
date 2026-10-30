"""Pemetaan baris curated (kolom DTA FKRTL) -> request JSON API v1.

Dipakai untuk: fixture uji, uji konsistensi batch vs online, dan sebagai acuan
pemetaan bagi Tim Website (lihat juga tabel pemetaan di dokumentasi PDF).
"""
from __future__ import annotations

import pandas as pd

REV_SEV = {"1": "0", "2": "I", "3": "II", "4": "III"}


def _s(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NA or v is pd.NaT:
        return None
    return str(v)


def curated_row_to_request(r: pd.Series, prediction_time: str | None = None) -> dict:
    dis = pd.Timestamp(r["discharge_date"])
    pt = prediction_time or (dis + pd.Timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    birth = r.get("birth_date")
    birth_date = pd.Timestamp(birth).strftime("%Y-%m-%d") if pd.notna(birth) else None
    sec = [x for x in str(r.get("sec_dx") or "").split("|") if x]
    special_codes = {p: _s(r.get(f"{p}_code")) for p in ["sa", "sp", "rr", "si", "sd"] if _s(r.get(f"{p}_code"))}
    special_tariffs = {p: float(r.get(f"{p}_tariff") or 0) for p in ["sa", "sp", "rr", "si", "sd"] if float(r.get(f"{p}_tariff") or 0) > 0}
    return {
        "claim_id": str(r["claim_id"]),
        "participant_key": str(r["participant_key"]),
        "prediction_time": pt,
        "episode": {
            "admission_date": pd.Timestamp(r["admission_date"]).strftime("%Y-%m-%d"),
            "discharge_date": dis.strftime("%Y-%m-%d"),
            "service_level": _s(r.get("service_level")),
            "provider_type": _s(r.get("provider_type")),
            "provider_region": _s(r.get("provider_region")),
            "care_class": _s(r.get("care_class")),
            "tariff_regional": _s(r.get("tariff_regional")),
            "provider_kind": _s(r.get("provider_kind")),
            "provider_ownership": _s(r.get("provider_ownership")),
            "service_unit": _s(r.get("service_unit")),
            "segment": _s(r.get("segment")),
            "discharge_status": _s(r.get("discharge_status")),
            "patient_sex": _s(r.get("sex")),
            "patient_birth_date": birth_date,
        },
        "diagnoses": {"primary_icd10": _s(r.get("primary_icd10")), "admission_icd10": _s(r.get("admission_icd10")),
                      "secondary_icd10": sec},
        "casemix": {"inacbg_code": _s(r.get("inacbg_code")), "severity": REV_SEV.get(_s(r.get("severity")))},
        "procedures": {"procedure_type": _s(r.get("procedure_text")), "special_codes": special_codes},
        "financial_pre_submit": {"billed_cost": float(r["billed_cost"]),
                                 "cbg_tariff": float(r["cbg_tariff"]) if _s(r.get("cbg_tariff")) else None,
                                 "special_tariffs": special_tariffs},
        "history_features": {},
        "peer_features": {},
    }
