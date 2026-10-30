"""Layer RAW -> CURATED.

* Membaca file DTA (immutable) dan memvalidasi schema.
* Pseudonimisasi PSTV01 / FKL02 / FKP02.
* Agregasi Diagnosis Sekunder per FKL02 SEBELUM join (hlm. 6).
* Join kepesertaan 1:1 (hanya atribut statis: jenis kelamin & tanggal lahir;
  PSTV17/PSTV18 adalah status akhir periode -> berpotensi leakage, tidak dipakai).
* FKL48 (biaya verifikasi BPJS) DIKELUARKAN dari tabel curated.
* Menghasilkan quality report (quality gates hlm. 6).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, pseudonymize, resolve

FKRTL_REQUIRED = [
    "PSTV01", "FKP02", "FKL02", "FKL03", "FKL04", "FKL05_NEW", "FKL06", "FKL07", "FKL08",
    "FKL09", "FKL10", "FKL11", "FKL12", "FKL13", "FKL14", "FKL15A", "FKL16", "FKL17A",
    "FKL18", "FKL19", "FKL20", "FKL21", "FKL22", "FKL23", "FKL30", "FKL31", "FKL32",
    "FKL33", "FKL34", "FKL35", "FKL37", "FKL38", "FKL40", "FKL41", "FKL43", "FKL44",
    "FKL46", "FKL47", "PSTV15",
]
FORBIDDEN_PRE_SUBMIT = ["FKL48"]
DXSEK_REQUIRED = ["FKL02", "FKL24", "FKL24A"]
MEMBER_REQUIRED = ["PSTV01", "PSTV03", "PSTV05"]

RENAME = {
    "FKL03": "admission_date", "FKL04": "discharge_date",
    "FKL05_NEW": "provider_region", "FKL06": "provider_city", "FKL07": "provider_ownership",
    "FKL08": "provider_kind", "FKL09": "provider_type", "FKL10": "service_level_code",
    "FKL11": "service_unit", "FKL12": "segment", "FKL13": "care_class", "FKL14": "discharge_status",
    "FKL15A": "admission_icd10_3", "FKL16": "admission_icd10", "FKL17A": "primary_icd10_3",
    "FKL18": "primary_icd10", "FKL19": "inacbg_code", "FKL20": "cmg", "FKL21": "case_type",
    "FKL22": "specificity", "FKL23": "severity", "FKL30": "procedure_text", "FKL31": "tariff_regional",
    "FKL32": "cbg_tariff", "FKL33": "sa_code", "FKL34": "sa_tariff", "FKL35": "sp_code",
    "FKL37": "sp_tariff", "FKL38": "rr_code", "FKL40": "rr_tariff", "FKL41": "si_code",
    "FKL43": "si_tariff", "FKL44": "sd_code", "FKL46": "sd_tariff", "FKL47": "billed_cost",
    "PSTV15": "sample_weight",
}
CAT_INT_COLS = ["provider_region", "provider_city", "provider_ownership", "provider_kind",
                "provider_type", "service_unit", "segment", "care_class", "discharge_status",
                "cmg", "case_type", "specificity", "severity", "tariff_regional"]


def norm_icd(s: pd.Series) -> pd.Series:
    """Kode ICD/INA-CBG tetap string (leading zero aman), uppercase, tanpa titik/spasi."""
    s = s.astype("string").str.upper().str.replace(".", "", regex=False).str.strip()
    return s.replace({"": pd.NA})


def _read_dta(path: Path, usecols=None) -> pd.DataFrame:
    """Read raw Stata data without turning value labels into categoricals.

    ``pyreadstat`` is faster when installed, but the standard pandas reader is
    a compatible local fallback.  Keeping ``convert_categoricals=False`` is
    important here: the numeric BPJS source codes must remain numeric until
    this module standardises them explicitly.
    """
    try:
        import pyreadstat
    except ImportError:
        return pd.read_stata(path, columns=usecols, convert_categoricals=False)
    df, _ = pyreadstat.read_dta(str(path), usecols=usecols, apply_value_formats=False)
    return df


def schema_check(df: pd.DataFrame, required: list[str], name: str) -> dict:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[schema] {name}: kolom wajib hilang {missing}")
    return {"table": name, "rows": int(len(df)), "columns": int(df.shape[1]), "missing_required": missing}


def aggregate_secondary_dx(dx: pd.DataFrame) -> pd.DataFrame:
    """Agregat wajib per FKL02 (hlm. 8)."""
    dx = dx.copy()
    dx["FKL24"] = norm_icd(dx["FKL24"])
    dx["FKL24A"] = norm_icd(dx["FKL24A"])
    dx = dx.dropna(subset=["FKL24"]).drop_duplicates(["FKL02", "FKL24"])
    dx = dx.sort_values(["FKL02", "FKL24"])
    full = dx.groupby("FKL02", sort=False)["FKL24"].agg("|".join)
    d3 = dx.dropna(subset=["FKL24A"]).drop_duplicates(["FKL02", "FKL24A"]).sort_values(["FKL02", "FKL24A"])
    three = d3.groupby("FKL02", sort=False)["FKL24A"].agg("|".join)
    out = pd.DataFrame({"sec_dx": full}).join(three.rename("sec_dx3")).reset_index()
    out["sec_dx3"] = out["sec_dx3"].fillna("")
    return out


def build_curated(cfg: dict | None = None, raw_dir: str | None = None, out_dir: str | None = None,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or load_config()
    raw = resolve(raw_dir or cfg["paths"]["raw_dir"])
    out = resolve(out_dir or cfg["paths"]["curated_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {"tables": []}

    # ---------- FKRTL ----------
    log("[ingest] membaca FKRTL ...")
    fk = _read_dta(raw / cfg["raw_files"]["fkrtl"])
    report["tables"].append(schema_check(fk, FKRTL_REQUIRED, "fkrtl"))
    fk = fk[[c for c in FKRTL_REQUIRED]]                 # FKL48 & deskripsi teks dibuang
    report["forbidden_columns_dropped"] = FORBIDDEN_PRE_SUBMIT
    n_dup = int(fk["FKL02"].duplicated().sum())
    report["fkl02_duplicates_before_dedup"] = n_dup
    fk = fk.drop_duplicates("FKL02")

    # ---------- Diagnosis sekunder ----------
    log("[ingest] membaca & mengagregasi diagnosis sekunder ...")
    dx = _read_dta(raw / cfg["raw_files"]["dx_secondary"], usecols=DXSEK_REQUIRED)
    report["tables"].append(schema_check(dx, DXSEK_REQUIRED, "dx_secondary"))
    dx_agg = aggregate_secondary_dx(dx)
    report["dx_secondary_orphans"] = int((~dx["FKL02"].isin(fk["FKL02"])).sum())

    # ---------- Kepesertaan ----------
    log("[ingest] membaca kepesertaan ...")
    mb = _read_dta(raw / cfg["raw_files"]["membership"], usecols=MEMBER_REQUIRED)
    report["tables"].append(schema_check(mb, MEMBER_REQUIRED, "membership"))
    mb = mb.drop_duplicates("PSTV01")

    # ---------- Join bertahap + rekonsiliasi row count ----------
    n0 = len(fk)
    cur = fk.merge(dx_agg, on="FKL02", how="left", validate="one_to_one")
    cur = cur.merge(mb, on="PSTV01", how="left", validate="many_to_one")
    report["join_reconciliation"] = {"fkrtl_rows": n0, "after_join": int(len(cur)), "ok": n0 == len(cur)}
    assert n0 == len(cur), "Join menggandakan baris!"

    # ---------- Standardisasi ----------
    cur = cur.rename(columns=RENAME).rename(columns={"PSTV05": "sex_code", "PSTV03": "birth_date"})
    cur["admission_date"] = pd.to_datetime(cur["admission_date"])
    cur["discharge_date"] = pd.to_datetime(cur["discharge_date"])
    cur["birth_date"] = pd.to_datetime(cur["birth_date"], errors="coerce")
    cur["service_level"] = cur["service_level_code"].map({1: "RJTL", 2: "RITL"}).astype("string")
    cur["sex"] = cur["sex_code"].map({1: "M", 2: "F"}).astype("string")
    for c in ["admission_icd10", "admission_icd10_3", "primary_icd10", "primary_icd10_3", "inacbg_code"]:
        cur[c] = norm_icd(cur[c])
    for c in CAT_INT_COLS:
        cur[c] = cur[c].astype("Int64").astype("string")
    for c in ["sa_code", "sp_code", "rr_code", "si_code", "sd_code"]:
        s = cur[c].astype("string").str.strip().str.upper()
        cur[c] = s.mask(s.isin(["", "NONE"]) | s.isna(), pd.NA)
    cur["procedure_text"] = cur["procedure_text"].astype("string").str.strip().replace({"": pd.NA})
    for c in ["sec_dx", "sec_dx3"]:
        cur[c] = cur[c].astype("string").fillna("")

    # ---------- Pseudonimisasi (identifier mentah tidak keluar zona raw) ----------
    log("[ingest] pseudonimisasi identifier ...")
    cur["claim_id"] = pseudonymize(cur["FKL02"].tolist(), cfg, "CLM")
    uniq = cur["PSTV01"].astype("Int64").astype("string").unique()
    pmap = dict(zip(uniq, pseudonymize(list(uniq), cfg, "PST")))
    cur["participant_key"] = cur["PSTV01"].astype("Int64").astype("string").map(pmap)
    cur["referral_key"] = pseudonymize(cur["FKP02"].tolist(), cfg, "REF")
    cur = cur.drop(columns=["PSTV01", "FKL02", "FKP02", "service_level_code", "sex_code"])

    # ---------- Quality gates ----------
    q = {}
    q["claim_id_unique"] = bool(cur["claim_id"].is_unique)
    q["discharge_before_admission"] = int((cur["discharge_date"] < cur["admission_date"]).sum())
    q["missing_rate"] = {c: round(float(v), 6) for c, v in cur.isna().mean().items() if v > 0}
    q["unknown_care_class_9"] = int((cur["care_class"] == "9").sum())
    q["participant_without_membership"] = int(cur["sex"].isna().sum())
    q["claims_with_secondary_dx"] = int((cur["sec_dx"] != "").sum())
    q["period"] = [str(cur["admission_date"].min().date()), str(cur["discharge_date"].max().date())]
    report["quality"] = q
    report["rows_curated"] = int(len(cur))
    report["elapsed_sec"] = round(time.time() - t0, 1)

    cur = cur.sort_values(["participant_key", "admission_date", "claim_id"]).reset_index(drop=True)
    cur.to_parquet(out / "claims_curated.parquet", index=False)

    # Referensi peserta (zona ML terbatas) untuk scoring online.
    part = cur.groupby("participant_key", as_index=False).agg(sex=("sex", "first"), birth_date=("birth_date", "first"))
    part.to_parquet(out / "participants_ref.parquet", index=False)
    with open(out / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"[ingest] selesai: {len(cur):,} klaim curated ({report['elapsed_sec']} dtk)")
    return cur, part, report
