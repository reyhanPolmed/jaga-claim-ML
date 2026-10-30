"""Engine 1 — Rule-Based Validation (deterministik, terversi).

Setiap aturan: id, reason_code, bobot (kontribusi ke rule_score), deskripsi, status
persetujuan. Status default "DRAFT" -> wajib disetujui reviewer casemix sebelum
dipakai untuk keputusan operasional (API Contract bag. 4 & 6).

rule_score R = 1 - prod(1 - w_i) untuk semua aturan yang terpicu (noisy-OR),
sehingga R ada di [0, 1] dan bertambah monoton dengan jumlah/berat temuan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- ICD-10 sex edits (disederhanakan ke 3 digit; sumber: WHO ICD-10 sex-specific categories) ----
MALE_ONLY_3 = {"C60", "C61", "C62", "C63", "D29", "D40", "E29", "Q53", "Q54", "Q55", "R86"} | {f"N{i}" for i in range(40, 52)}
FEMALE_ONLY_3 = ({"A34", "D06", "D25", "D26", "D27", "D28", "D39", "E28", "F53", "Q50", "Q51", "Q52", "R87",
                  "Z32", "Z33", "Z34", "Z35", "Z36", "Z37", "Z39"}
                 | {f"C{i}" for i in range(51, 59)} | {f"N{i}" for i in range(70, 99)}
                 | {f"O{i:02d}" for i in range(0, 100)})
CMG_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F", "7": "G", "8": "H", "9": "I", "10": "J",
              "11": "K", "12": "L", "13": "M", "14": "N", "15": "O", "16": "P", "17": "Q", "18": "S", "19": "T",
              "20": "U", "21": "V", "22": "W", "23": "Z"}
LETTER_CMG = {v: k for k, v in CMG_LETTER.items()}
SEV_FROM_ROMAN = {"0": "1", "I": "2", "II": "3", "III": "4"}
INPATIENT_CASE_TYPES = {"1", "4", "6", "8"}
PROCEDURE_CASE_TYPES = {"1", "2", "3"}


def parse_cbg(code: pd.Series) -> pd.DataFrame:
    """INA-CBG 'A-4-14-I' -> cmg, case_type, specificity, severity (kode FKL20-FKL23)."""
    parts = code.astype("string").str.split("-", expand=True).reindex(columns=range(4))
    out = pd.DataFrame(index=code.index)
    out["cmg_from_code"] = parts[0].map(LETTER_CMG).astype("string")
    out["case_type_from_code"] = parts[1].astype("string")
    out["specificity_from_code"] = parts[2].astype("string").str.lstrip("0").replace({"": "0"})
    out["severity_from_code"] = parts[3].map(SEV_FROM_ROMAN).astype("string")
    return out


@dataclass(frozen=True)
class Rule:
    rule_id: str
    reason_code: str
    weight: float
    description: str
    status: str = "DRAFT-perlu-persetujuan-casemix"


RULES: list[Rule] = [
    Rule("DQ01_MISSING_CRITICAL", "DATA_QUALITY", 0.70, "Field wajib kosong/invalid: diagnosis primer, INA-CBG, tanggal datang, atau biaya tagih <= 0."),
    Rule("DQ02_MISSING_CONTEXT", "DATA_QUALITY", 0.10, "Field konteks kosong/unknown: tarif regional, kelas rawat, atau tipe provider."),
    Rule("DQ03_DATE_INVALID", "DATA_QUALITY", 0.70, "Tanggal pulang lebih awal dari tanggal datang, atau tanggal datang setelah prediction_time."),
    Rule("DQ04_AGE_INVALID", "DATA_QUALITY", 0.50, "Usia saat layanan < 0 atau > 110 tahun (tanggal lahir tidak konsisten)."),
    Rule("DQ05_SERVICE_AFTER_DEATH", "DATA_QUALITY", 0.80, "Ada episode sebelumnya dengan status pulang Meninggal, tetapi terdapat layanan sesudahnya."),
    Rule("DQ06_TARIFF_RECON", "DATA_QUALITY", 0.60, "Biaya tagih melebihi tarif grup INA-CBG + seluruh tarif komponen special."),
    Rule("DQ07_SPECIAL_CODE_NO_TARIFF", "DATA_QUALITY", 0.25, "Kode komponen special terisi tetapi tarifnya nol."),
    Rule("CM01_SEX_DX", "CODING_MISMATCH", 0.70, "Diagnosis (primer/sekunder) spesifik jenis kelamin tidak sesuai jenis kelamin peserta."),
    Rule("CM02_AGE_DX", "CODING_MISMATCH", 0.50, "Diagnosis tidak lazim untuk usia: bab O pada usia <10/>55 th atau bab P primer pada usia >1 th."),
    Rule("CM03_SEX_CBG", "CODING_MISMATCH", 0.70, "Main group INA-CBG O (persalinan)/W (reproduksi wanita) pada laki-laki, atau V (reproduksi pria) pada perempuan."),
    Rule("CM04_PROC_CBG_NO_PROC", "CODING_MISMATCH", 0.40, "Tipe kasus INA-CBG berprosedur (digit-2 = 1/2/3) tetapi tidak ada prosedur tercatat."),
    Rule("CM05_HIGH_SEV_SHORT_LOS", "CODING_MISMATCH", 0.50, "Keparahan III rawat inap dengan LOS <= 1 hari dan pulang sehat."),
    Rule("CM06_SETTING_MISMATCH", "CODING_MISMATCH", 0.60, "Tingkat pelayanan (RJTL/RITL) tidak konsisten dengan tipe kasus/keparahan INA-CBG."),
    Rule("CM07_CBG_COMPONENT_MISMATCH", "CODING_MISMATCH", 0.50, "Komponen casemix/severity yang dikirim tidak sesuai dengan kode INA-CBG."),
]
RULE_BY_ID = {r.rule_id: r for r in RULES}
CRITICAL_RULES = {r.rule_id for r in RULES if r.weight >= 0.70}


def _codes_in(series: pd.Series, codeset: set) -> np.ndarray:
    return series.astype("string").fillna("").map(lambda s: any(x in codeset for x in s.split("|") if x)).to_numpy(dtype=bool)


def evaluate_rules(F: pd.DataFrame, prediction_time: pd.Series | None = None) -> pd.DataFrame:
    """Kembalikan DataFrame boolean (satu kolom per rule) + rule_score."""
    n = len(F)
    out = pd.DataFrame(index=F.index)
    s = lambda c: F[c].astype("string")
    billed = F["billed_cost"].astype(float)
    sex = s("sex")
    age = F["age_years"].astype(float)
    los = F["los_days"].astype(float)
    sev = s("severity")
    ct = s("case_type")
    lvl = s("service_level")

    out["DQ01_MISSING_CRITICAL"] = (F["primary_icd10"].isna() | F["inacbg_code"].isna() | F["admission_date"].isna()
                                   | billed.isna() | (billed <= 0)).to_numpy()
    out["DQ02_MISSING_CONTEXT"] = (F["tariff_regional"].isna() | F["care_class"].isna() | (s("care_class") == "9").fillna(False)
                                  | F["provider_type"].isna() | (s("provider_type") == "99").fillna(False)).to_numpy()
    bad_date = (pd.to_datetime(F["discharge_date"]) < pd.to_datetime(F["admission_date"])).fillna(False)
    if prediction_time is not None:
        pt = pd.to_datetime(prediction_time, utc=True).dt.tz_localize(None).dt.normalize()
        bad_date = bad_date | (pd.to_datetime(F["admission_date"]) > pt).fillna(False)
    out["DQ03_DATE_INVALID"] = bad_date.to_numpy()
    out["DQ04_AGE_INVALID"] = ((age < 0) | (age > 110)).fillna(False).to_numpy()
    out["DQ05_SERVICE_AFTER_DEATH"] = (F["service_after_death"] > 0).to_numpy()
    exp = F["expected_tariff"].astype(float)
    out["DQ06_TARIFF_RECON"] = ((exp > 0) & (billed > exp * 1.01 + 1)).fillna(False).to_numpy()
    sc = np.zeros(n, dtype=bool)
    for p in ["sa", "sp", "rr", "si", "sd"]:
        if f"{p}_code" in F and f"{p}_tariff" in F:
            sc |= (F[f"{p}_code"].notna() & (pd.to_numeric(F[f"{p}_tariff"], errors="coerce").fillna(0) <= 0)).to_numpy()
    out["DQ07_SPECIAL_CODE_NO_TARIFF"] = sc

    allp = F["primary_icd10_3"].astype("string").fillna("") + "|" + F["sec_dx3"].astype("string").fillna("")
    male_dx = _codes_in(allp, MALE_ONLY_3)
    fem_dx = _codes_in(allp, FEMALE_ONLY_3)
    out["CM01_SEX_DX"] = (((sex == "F").fillna(False).to_numpy() & male_dx) | ((sex == "M").fillna(False).to_numpy() & fem_dx))
    p3 = F["primary_icd10_3"].astype("string").fillna("")
    o_chapter = p3.str.startswith("O").to_numpy()
    p_chapter = p3.str.startswith("P").to_numpy()
    out["CM02_AGE_DX"] = ((o_chapter & ((age < 10) | (age > 55)).fillna(False).to_numpy())
                          | (p_chapter & (age > 1).fillna(False).to_numpy()))
    cmg = F["inacbg_code"].astype("string").str[0]
    out["CM03_SEX_CBG"] = (((cmg.isin(["O", "W"])) & (sex == "M")) | ((cmg == "V") & (sex == "F"))).fillna(False).to_numpy()
    out["CM04_PROC_CBG_NO_PROC"] = (ct.isin(PROCEDURE_CASE_TYPES).fillna(False) & F["procedure_text"].isna()).to_numpy()
    out["CM05_HIGH_SEV_SHORT_LOS"] = ((lvl == "RITL") & (sev == "4") & (los <= 1) & (s("discharge_status") == "1")).fillna(False).to_numpy()
    inpt_ct = ct.isin(INPATIENT_CASE_TYPES)
    out["CM06_SETTING_MISMATCH"] = (((lvl == "RJTL") & (inpt_ct | (sev.fillna("1") != "1")))
                                   | ((lvl == "RITL") & (~inpt_ct | (sev == "1")))).fillna(False).to_numpy()
    pc = parse_cbg(F["inacbg_code"])
    mis = np.zeros(n, dtype=bool)
    for a, b in [("cmg", "cmg_from_code"), ("case_type", "case_type_from_code"), ("severity", "severity_from_code")]:
        x, y = F[a].astype("string"), pc[b]
        mis |= (x.notna() & y.notna() & (x != y)).fillna(False).to_numpy()
    out["CM07_CBG_COMPONENT_MISMATCH"] = mis

    W = np.array([RULE_BY_ID[c].weight for c in out.columns])
    M = out.to_numpy(dtype=float)
    out["rule_score"] = 1.0 - np.prod(1.0 - M * W, axis=1)
    out["rule_critical"] = out[[c for c in out.columns if c in CRITICAL_RULES]].any(axis=1)
    return out


def rule_evidence(row: pd.Series, F_row: pd.Series, rule_version: str) -> list[dict]:
    ev = []
    for r in RULES:
        if not bool(row.get(r.rule_id, False)):
            continue
        val = {
            "DQ06_TARIFF_RECON": lambda: f"tagih Rp{F_row['billed_cost']:,.0f} vs tarif CBG+special Rp{F_row['expected_tariff']:,.0f}",
            "DQ04_AGE_INVALID": lambda: f"usia {F_row['age_years']:.1f} th",
            "CM01_SEX_DX": lambda: f"jenis kelamin {F_row['sex']}; dx {F_row['primary_icd10_3']} | sekunder {F_row['sec_dx3'] or '-'}",
            "CM02_AGE_DX": lambda: f"dx primer {F_row['primary_icd10_3']} pada usia {F_row['age_years']:.1f} th",
            "CM03_SEX_CBG": lambda: f"INA-CBG {F_row['inacbg_code']} pada jenis kelamin {F_row['sex']}",
            "CM04_PROC_CBG_NO_PROC": lambda: f"INA-CBG {F_row['inacbg_code']} (tipe kasus {F_row['case_type']}), prosedur kosong",
            "CM05_HIGH_SEV_SHORT_LOS": lambda: f"INA-CBG {F_row['inacbg_code']} keparahan III, LOS {F_row['los_days']:.0f} hari",
            "CM06_SETTING_MISMATCH": lambda: f"pelayanan {F_row['service_level']} vs INA-CBG {F_row['inacbg_code']}",
            "CM07_CBG_COMPONENT_MISMATCH": lambda: f"INA-CBG {F_row['inacbg_code']} vs severity dikirim {F_row['severity']}",
            "DQ01_MISSING_CRITICAL": lambda: "kosong/invalid: " + ", ".join(
                k for k, bad in [("primary_icd10", pd.isna(F_row["primary_icd10"])), ("inacbg_code", pd.isna(F_row["inacbg_code"])),
                                 ("admission_date", pd.isna(F_row["admission_date"])),
                                 ("billed_cost", pd.isna(F_row["billed_cost"]) or F_row["billed_cost"] <= 0)] if bad),
            "DQ02_MISSING_CONTEXT": lambda: "kosong/unknown: " + ", ".join(
                k for k, bad in [("tariff_regional", pd.isna(F_row["tariff_regional"])),
                                 ("care_class", pd.isna(F_row["care_class"]) or str(F_row["care_class"]) == "9"),
                                 ("provider_type", pd.isna(F_row["provider_type"]) or str(F_row["provider_type"]) == "99")] if bad),
            "DQ03_DATE_INVALID": lambda: f"datang {pd.Timestamp(F_row['admission_date']).date()}, pulang {pd.Timestamp(F_row['discharge_date']).date()}",
            "DQ07_SPECIAL_CODE_NO_TARIFF": lambda: ", ".join(f"{p}_code={F_row[p + '_code']}" for p in ["sa", "sp", "rr", "si", "sd"]
                                                             if pd.notna(F_row.get(p + "_code")) and not (F_row.get(p + "_tariff") or 0) > 0),
            "DQ05_SERVICE_AFTER_DEATH": lambda: f"layanan {F_row['prior_death_day_gap']:.0f} hari setelah episode berstatus Meninggal",
        }.get(r.rule_id, lambda: "lihat field terkait")()
        ev.append({"feature": f"rule:{r.rule_id}", "value": val,
                   "reference": f"{r.description} [{rule_version}; status {r.status}]"})
    return ev
