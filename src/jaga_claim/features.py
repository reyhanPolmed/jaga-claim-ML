"""Feature engineering point-in-time (Dokumentasi hlm. 9).

Satu fungsi `build_features` dipakai oleh batch (seluruh data) DAN oleh API
(riwayat peserta + 1 klaim baru). Karena semua fitur historis hanya melihat
episode SEBELUMNYA (urutan: participant_key, admission_date, claim_id),
hasil batch dan online identik (diuji di tests/test_consistency.py).

Definisi "prior" untuk episode i: episode j milik peserta yang sama dengan
(admission_date_j < admission_date_i) atau (tanggal sama dan claim_id_j < claim_id_i).
Asumsi pra-submit: SEP terbit saat admisi, sehingga episode yang admisinya lebih
awal sudah diketahui sistem pada saat klaim i dinilai (setelah pasien pulang).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-01")
SPECIAL = ["sa", "sp", "rr", "si", "sd"]
BIG = 1_000_000  # > jumlah hari sejak epoch


def _days(s: pd.Series) -> np.ndarray:
    return ((s.values.astype("datetime64[D]") - EPOCH).astype("int64"))


def _split_set(s: str) -> frozenset:
    return frozenset(x for x in s.split("|") if x) if isinstance(s, str) else frozenset()


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    adm = pd.to_datetime(df["admission_date"])
    dis = pd.to_datetime(df["discharge_date"])
    f["los_days"] = (dis - adm).dt.days.astype("float")
    f["is_inpatient"] = (df["service_level"] == "RITL").astype("float")
    f["age_years"] = ((adm - pd.to_datetime(df["birth_date"])).dt.days / 365.25).astype("float")
    sp_sum = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for p in SPECIAL:
        t = pd.to_numeric(df.get(f"{p}_tariff"), errors="coerce").fillna(0).to_numpy(dtype=float)
        c = df.get(f"{p}_code")
        sp_sum += t
        cnt += (c.notna().to_numpy() if c is not None else 0)
    billed = pd.to_numeric(df["billed_cost"], errors="coerce").astype(float)
    cbg_t = pd.to_numeric(df.get("cbg_tariff"), errors="coerce").astype(float)
    f["billed_cost"] = billed
    f["log_billed"] = np.log1p(billed.clip(lower=0))
    f["special_tariff_sum"] = sp_sum
    f["special_component_count"] = cnt
    f["special_share"] = np.where(billed > 0, sp_sum / billed, 0.0)
    expected = cbg_t + sp_sum
    f["expected_tariff"] = expected
    f["tariff_recon_ratio"] = np.where(expected > 0, billed / expected, np.nan)
    proc = df["procedure_text"].astype("string")
    f["procedure_count"] = proc.fillna("").map(lambda s: len([x for x in s.split(";") if x.strip()])).astype(float)
    sec = df["sec_dx"].fillna("").astype(str)
    sec3 = df["sec_dx3"].fillna("").astype(str)
    f["sec_dx_count"] = sec.map(lambda s: len([x for x in s.split("|") if x])).astype(float)
    f["sec_dx_unique3"] = sec3.map(lambda s: len([x for x in s.split("|") if x])).astype(float)
    f["sec_dx_chapter_count"] = sec3.map(lambda s: len({x[0] for x in s.split("|") if x})).astype(float)
    f["has_secondary_dx"] = (f["sec_dx_count"] > 0).astype(float)
    f["sec_dx_set_hash"] = sec3.map(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])
    f["admit_primary_same"] = (df["admission_icd10_3"].astype("string") == df["primary_icd10_3"].astype("string")).fillna(False).astype(float)
    f["severity_num"] = pd.to_numeric(df["severity"], errors="coerce").astype(float)
    return f


def _window_count(order: np.ndarray, gkey: np.ndarray, day: np.ndarray, window: int) -> np.ndarray:
    """Jumlah episode prior dalam grup yang sama dengan admisi >= day-window."""
    g = gkey[order].astype(np.int64)
    d = day[order]
    k = g * BIG + d
    pos = np.arange(len(k))
    lower = np.searchsorted(k, g * BIG + (d - window), side="left")
    res = np.empty(len(k), dtype=float)
    res[order] = pos - lower
    return res


def _prev_in_group(order: np.ndarray, gkey: np.ndarray, values: np.ndarray) -> np.ndarray:
    g = gkey[order]
    v = values[order].astype(float)
    prev = np.r_[np.nan, v[:-1]]
    same = np.r_[False, g[1:] == g[:-1]]
    prev[~same] = np.nan
    out = np.empty(len(v))
    out[order] = prev
    return out


def history_features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Fitur temporal/historis peserta. `df` WAJIB terurut (participant_key, admission_date, claim_id)."""
    n = len(df)
    h = pd.DataFrame(index=df.index)
    pid = pd.factorize(df["participant_key"], sort=False)[0].astype(np.int64)
    day = _days(pd.to_datetime(df["admission_date"]))
    dday = _days(pd.to_datetime(df["discharge_date"]))
    base_order = np.arange(n)

    h["prior_visits_30d"] = _window_count(base_order, pid, day, 30)
    h["prior_visits_90d"] = _window_count(base_order, pid, day, 90)
    h["prior_episodes_total"] = _window_count(base_order, pid, day, 10 ** 5)

    prev_day = _prev_in_group(base_order, pid, day.astype(float))
    h["days_since_last_visit"] = day - prev_day

    def grouped(col: str, window: int, name: str, gap_name: str | None = None):
        code = pd.factorize(df[col].astype("string").fillna("<NA>"))[0].astype(np.int64)
        g2 = pid * 200_000 + code
        order = np.lexsort((np.arange(n), day, g2))   # stabil terhadap urutan claim_id
        h[name] = _window_count(order, g2, day, window)
        if gap_name:
            h[gap_name] = day - _prev_in_group(order, g2, day.astype(float))

    grouped("inacbg_code", 30, "same_cbg_visits_30d")
    grouped("primary_icd10_3", 90, "same_dx3_visits_90d", "days_since_last_same_dx3")

    # Rawat inap ulang (RITL) dengan dx3 sama -> jarak dari pulang sebelumnya
    inp = (df["service_level"] == "RITL").to_numpy()
    h["readmit_gap_days"] = np.nan
    if inp.any():
        sub_idx = np.where(inp)[0]
        code = pd.factorize(df["primary_icd10_3"].astype("string").fillna("<NA>"))[0].astype(np.int64)
        g2 = (pid * 200_000 + code)[sub_idx]
        o = np.lexsort((np.arange(len(sub_idx)), day[sub_idx], g2))
        prev_dis = _prev_in_group(o, g2, dday[sub_idx].astype(float))
        gap = day[sub_idx] - prev_dis
        h.iloc[sub_idx, h.columns.get_loc("readmit_gap_days")] = gap

    # Overlap: admisi saat ini jatuh di dalam rawat inap PRIOR yang belum pulang
    ritl_dis = np.where(inp, dday, -1).astype(float)
    prev = _prev_in_group(base_order, pid, ritl_dis)
    prior_max = pd.Series(prev).groupby(pid).cummax().to_numpy(dtype=float, copy=True)
    h["overlap_prior_inpatient"] = ((~np.isnan(prior_max)) & (day < prior_max)).astype(float)

    # Layanan setelah episode berstatus pulang "Meninggal" (FKL14=3)
    death_day = np.where((df["discharge_status"].astype("string") == "3").fillna(False).to_numpy(), dday, np.inf).astype(float)
    prev_d = _prev_in_group(base_order, pid, death_day)
    prev_d = np.where(np.isnan(prev_d), np.inf, prev_d)
    pdd = pd.Series(prev_d).groupby(pid).cummin().to_numpy(dtype=float, copy=True)
    pdd[np.isinf(pdd)] = np.nan
    h["service_after_death"] = ((~np.isnan(pdd)) & (day > pdd)).astype(float)
    h["prior_death_day_gap"] = np.where(np.isnan(pdd), np.nan, day - pdd)
    return h


def duplicate_features(df: pd.DataFrame, base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Engine 3a: exact & fuzzy duplicate terhadap K episode prior peserta yang sama."""
    K = cfg["duplicate"]["lookback_episodes"]
    W = cfg["duplicate"]["fuzzy_window_days"]
    w = cfg["duplicate"]["weights"]
    n = len(df)
    pid = df["participant_key"].to_numpy()
    day = _days(pd.to_datetime(df["admission_date"]))
    dday = _days(pd.to_datetime(df["discharge_date"]))
    cbg = df["inacbg_code"].astype("string").fillna("").to_numpy()
    pdx = df["primary_icd10"].astype("string").fillna("").to_numpy()
    pdx3 = df["primary_icd10_3"].astype("string").fillna("").to_numpy()
    billed = base["billed_cost"].to_numpy(dtype=float)
    proc = df["procedure_text"].astype("string").fillna("").to_numpy()
    unit = df["service_unit"].astype("string").fillna("").to_numpy()
    sec_sets = [_split_set(s) for s in df["sec_dx3"].fillna("").astype(str)]
    cid = df["claim_id"].to_numpy()

    best = np.zeros(n)
    best_sim = np.zeros(n)
    best_gap = np.full(n, np.nan)
    best_ref = np.full(n, None, dtype=object)
    exact = np.zeros(n)
    exact_ref = np.full(n, None, dtype=object)
    same_unit = np.full(n, np.nan)
    decay = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.25}   # gap hari -> peluruhan (kunjungan rutin harian tidak dianggap duplikat)

    for k in range(1, K + 1):
        if k >= n:
            break
        i = np.arange(k, n)
        j = i - k
        m = pid[i] == pid[j]
        if not m.any():
            continue
        i, j = i[m], j[m]
        gap = day[i] - day[j]
        ex = (gap == 0) & (dday[i] == dday[j]) & (cbg[i] == cbg[j]) & (pdx[i] == pdx[j]) & (billed[i] == billed[j]) & (cbg[i] != "")
        newex = ex & (exact[i] == 0)
        exact[i[newex]] = 1.0
        exact_ref[i[newex]] = cid[j[newex]]
        same_unit[i[newex]] = (unit[i[newex]] == unit[j[newex]]).astype(float)

        win = gap <= W
        if not win.any():
            continue
        i, j, gap = i[win], j[win], gap[win]
        s_dx = np.where(pdx[i] == pdx[j], 1.0, np.where(pdx3[i] == pdx3[j], 0.5, 0.0))
        s_cbg = (cbg[i] == cbg[j]).astype(float)
        jac = np.array([
            1.0 if (not sec_sets[a] and not sec_sets[b]) else
            len(sec_sets[a] & sec_sets[b]) / max(1, len(sec_sets[a] | sec_sets[b]))
            for a, b in zip(i, j)
        ])
        mx = np.maximum(np.maximum(billed[i], billed[j]), 1.0)
        s_cost = 1.0 - np.abs(billed[i] - billed[j]) / mx
        s_proc = (proc[i] == proc[j]).astype(float)
        sim = (w["same_primary_dx"] * s_dx + w["same_cbg"] * s_cbg + w["sec_dx_jaccard"] * jac
               + w["cost_close"] * s_cost + w["same_procedure"] * s_proc)
        score = sim * np.vectorize(decay.get)(gap.astype(int))
        upd = score > best[i]
        best[i[upd]] = score[upd]
        best_sim[i[upd]] = sim[upd]
        best_gap[i[upd]] = gap[upd]
        best_ref[i[upd]] = cid[j[upd]]

    out = pd.DataFrame(index=df.index)
    out["dup_exact"] = exact
    out["dup_exact_ref"] = exact_ref
    out["dup_exact_same_unit"] = same_unit
    out["dup_fuzzy_score"] = best
    out["dup_fuzzy_sim"] = best_sim
    out["dup_fuzzy_gap_days"] = best_gap
    out["dup_fuzzy_ref"] = best_ref
    return out


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Gabungan fitur base + historis + duplikasi. Input diurutkan di sini."""
    df = df.sort_values(["participant_key", "admission_date", "claim_id"], kind="mergesort").reset_index(drop=True)
    b = base_features(df)
    h = history_features(df, b)
    d = duplicate_features(df, b, cfg)
    keep = ["claim_id", "participant_key", "admission_date", "discharge_date", "service_level",
            "provider_type", "provider_region", "provider_kind", "provider_ownership", "care_class",
            "tariff_regional", "service_unit", "segment", "discharge_status", "sex",
            "admission_icd10_3", "primary_icd10", "primary_icd10_3", "inacbg_code", "cmg", "case_type",
            "severity", "procedure_text", "sec_dx", "sec_dx3", "cbg_tariff",
            "sa_code", "sa_tariff", "sp_code", "sp_tariff", "rr_code", "rr_tariff", "si_code", "si_tariff", "sd_code", "sd_tariff"]
    keep = [c for c in keep if c in df.columns]
    return pd.concat([df[keep], b, h, d], axis=1)
