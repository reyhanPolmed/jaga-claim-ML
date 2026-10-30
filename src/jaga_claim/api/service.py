"""ScoringService: logika scoring online (dipakai FastAPI & test), tanpa ketergantungan HTTP.

Alur per request:
  request JSON -> baris klaim kanonik (pemetaan field API v1 -> kolom curated)
  -> ambil riwayat peserta dari history store (hanya episode dengan admisi <= prediction_time)
  -> build_features (fungsi yang SAMA dengan batch) -> HybridRiskEngine.score_frame -> response kontrak.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, resolve
from ..engine import HybridRiskEngine
from ..engines.rules import INPATIENT_CASE_TYPES, SEV_FROM_ROMAN, parse_cbg
from ..features import build_features
from .schemas import ScoreRequest

log = logging.getLogger("jaga-claim.service")


class InsufficientData(ValueError):
    pass


def _icd(x):
    if x is None:
        return None
    s = str(x).upper().replace(".", "").strip()
    return s or None


def _sev(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in SEV_FROM_ROMAN:
        return SEV_FROM_ROMAN[s]
    return s if s in {"1", "2", "3", "4"} else None


class ScoringService:
    def __init__(self, cfg: dict | None = None, artifacts_dir: str | None = None, feature_dir: str | None = None,
                 curated_dir: str | None = None):
        self.cfg = cfg or load_config()
        a = resolve(artifacts_dir or self.cfg["paths"]["artifacts_dir"])
        active = (a / "ACTIVE").read_text().strip() if (a / "ACTIVE").exists() else self.cfg["versions"]["model_version"]
        t = time.time()
        self.engine = HybridRiskEngine.load(a / active)
        fdir = resolve(feature_dir or self.cfg["paths"]["feature_dir"])
        cdir = resolve(curated_dir or self.cfg["paths"]["curated_dir"])
        hs = pd.read_parquet(fdir / "history_store.parquet")
        for c in hs.columns:   # kolom string Arrow -> object: pengambilan baris per peserta jauh lebih cepat
            if pd.api.types.is_string_dtype(hs[c]) and not pd.api.types.is_datetime64_any_dtype(hs[c]):
                hs[c] = hs[c].astype(object).where(hs[c].notna(), None)
        self.engine.anomaly.model.set_params(n_jobs=1)   # 1 baris/request: hindari overhead thread
        self.history = hs
        self.hist_index = hs.groupby("participant_key", sort=False).indices
        pr = pd.read_parquet(cdir / "participants_ref.parquet")
        self.participants = pr.set_index("participant_key")
        self.loaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.load_seconds = round(time.time() - t, 1)
        log.info("ScoringService siap: %s, %d episode riwayat, %d peserta", active, len(hs), len(pr))

    # ------------------------------------------------------------------
    def to_row(self, r: ScoreRequest) -> dict:
        ep, dx, cm, pr, fin = r.episode, r.diagnoses, r.casemix, r.procedures, r.financial_pre_submit
        if ep.discharge_date is None:
            raise InsufficientData("episode.discharge_date wajib untuk scoring pra-submit (LOS & duplikasi)")
        code = str(cm.inacbg_code).strip().upper()
        pc = parse_cbg(pd.Series([code])).iloc[0]
        sev = _sev(cm.severity) or (pc["severity_from_code"] if pd.notna(pc["severity_from_code"]) else None)
        ct = pc["case_type_from_code"] if pd.notna(pc["case_type_from_code"]) else None
        level = ep.service_level or ("RITL" if ct in INPATIENT_CASE_TYPES else "RJTL")
        sex, birth = ep.patient_sex, ep.patient_birth_date
        if r.participant_key in self.participants.index:
            p = self.participants.loc[r.participant_key]
            sex = sex or (p["sex"] if pd.notna(p["sex"]) else None)
            birth = birth or (p["birth_date"] if pd.notna(p["birth_date"]) else None)
        sec_full = sorted({c for c in (_icd(x) for x in dx.secondary_icd10) if c})
        sec3 = sorted({c[:3] for c in sec_full})
        prim = _icd(dx.primary_icd10)
        adm = _icd(dx.admission_icd10)
        row = {
            "claim_id": r.claim_id, "participant_key": r.participant_key,
            "admission_date": pd.Timestamp(ep.admission_date), "discharge_date": pd.Timestamp(ep.discharge_date),
            "service_level": level, "provider_type": ep.provider_type, "provider_region": ep.provider_region,
            "provider_kind": ep.provider_kind, "provider_ownership": ep.provider_ownership, "care_class": ep.care_class,
            "tariff_regional": ep.tariff_regional, "service_unit": ep.service_unit, "segment": ep.segment,
            "discharge_status": ep.discharge_status,
            "admission_icd10": adm, "admission_icd10_3": adm[:3] if adm else None,
            "primary_icd10": prim, "primary_icd10_3": prim[:3] if prim else None,
            "inacbg_code": code, "cmg": pc["cmg_from_code"], "case_type": ct,
            "specificity": pc["specificity_from_code"], "severity": sev,
            "procedure_text": (pr.procedure_type or None), "sec_dx": "|".join(sec_full), "sec_dx3": "|".join(sec3),
            "cbg_tariff": fin.cbg_tariff, "billed_cost": fin.billed_cost,
            "sex": sex, "birth_date": pd.Timestamp(birth) if birth is not None else pd.NaT,
        }
        for p in ["sa", "sp", "rr", "si", "sd"]:
            c = pr.special_codes.get(p) or pr.special_codes.get(f"{p}_code")
            row[f"{p}_code"] = str(c).upper() if c not in (None, "", "NONE") else None
            row[f"{p}_tariff"] = float(fin.special_tariffs.get(p, fin.special_tariffs.get(f"{p}_tariff", 0)) or 0)
        return row

    def history_for(self, participant_key: str, claim_id: str, upto: pd.Timestamp) -> pd.DataFrame:
        idx = self.hist_index.get(participant_key)
        if idx is None:
            return self.history.iloc[0:0]
        h = self.history.iloc[idx]
        return h[(h["admission_date"] <= upto) & (h["claim_id"] != claim_id)]

    def score(self, req: ScoreRequest) -> dict:
        row = self.to_row(req)
        pt = pd.Timestamp(req.prediction_time).tz_convert("UTC").tz_localize(None)
        upto = min(row["admission_date"], pt.normalize())
        hist = self.history_for(req.participant_key, req.claim_id, upto)
        cur = pd.DataFrame([row])
        df = pd.concat([hist, cur], ignore_index=True) if len(hist) else cur
        for c in ["admission_date", "discharge_date", "birth_date"]:
            df[c] = pd.to_datetime(df[c])
        str_cols = [c for c in df.columns if c not in ("admission_date", "discharge_date", "birth_date", "billed_cost", "cbg_tariff")
                    and not c.endswith("_tariff")]
        for c in str_cols:
            df[c] = df[c].astype("string")
        for c in ["billed_cost", "cbg_tariff"] + [f"{p}_tariff" for p in ["sa", "sp", "rr", "si", "sd"]]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        F = build_features(df, self.cfg)
        F = F[F["claim_id"] == req.claim_id].reset_index(drop=True)
        O = self.engine.score_frame(F, prediction_time=pd.Series([req.prediction_time]))
        return self.engine.to_response(F.iloc[0], O.iloc[0])

    def components(self, req: ScoreRequest) -> dict:
        """Skor komponen untuk debugging/audit (tidak bagian dari kontrak)."""
        row = self.to_row(req)
        hist = self.history_for(req.participant_key, req.claim_id, row["admission_date"])
        df = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        F = build_features(df, self.cfg)
        F = F[F["claim_id"] == req.claim_id].reset_index(drop=True)
        O = self.engine.score_frame(F, prediction_time=pd.Series([req.prediction_time])).iloc[0]
        return {k: (None if pd.isna(O[k]) else float(O[k])) for k in ["R", "P", "D", "A", "S_weighted", "risk_score", "anomaly_percentile"]}
