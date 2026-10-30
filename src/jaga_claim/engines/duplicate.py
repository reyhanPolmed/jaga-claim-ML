"""Engine 3 — Duplicate & Temporal Detection (skor dari fitur yang sudah dihitung).

Komponen (diambil maksimum, 0..1):
* exact duplicate (tanggal datang & pulang, INA-CBG, dx primer, biaya identik) -> 1.0
* fuzzy similarity x peluruhan jarak hari (window 3 hari), bila >= ambang flag
* overlap: admisi di dalam rawat inap prior yang belum pulang -> 0.8
* readmisi rawat inap dx3 sama <= 7 hari -> 0.6
* frekuensi CBG sama 30 hari > p99.5 peer CBG -> 0.4 + kelebihan relatif
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def duplicate_temporal_scores(F: pd.DataFrame, P: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["temporal"]
    thr = cfg["duplicate"]["fuzzy_flag_threshold"]
    out = pd.DataFrame(index=F.index)
    exact = F["dup_exact"].to_numpy() > 0
    fuzzy = F["dup_fuzzy_score"].to_numpy(dtype=float)
    fuzzy_c = np.where(fuzzy >= thr, fuzzy, 0.0)   # hanya komponen yang ber-flag yang menyumbang skor
    # overlap/readmisi yang sebenarnya adalah episode duplikat itu sendiri tidak dihitung dua kali
    overlap = (F["overlap_prior_inpatient"].to_numpy() > 0) & ~exact
    gap = F["readmit_gap_days"]
    readmit = ((gap >= 0) & (gap <= t["readmit_days"])).fillna(False).to_numpy() & ~exact
    freq = P["flag_freq_peer"].to_numpy(dtype=bool)
    freq_c = np.where(freq, np.clip(0.4 + P["freq_excess"].fillna(0).to_numpy(), 0, 1), 0.0)
    D = np.maximum.reduce([exact * 1.0, fuzzy_c, overlap * t["overlap_score"], readmit * t["readmit_score"], freq_c])
    out["duplicate_score"] = D
    out["flag_possible_duplicate"] = exact | (fuzzy >= thr)
    out["flag_temporal_repeat"] = overlap | readmit | freq
    out["flag_exact_duplicate"] = exact
    out["flag_overlap"] = overlap
    out["flag_readmit"] = readmit
    out["flag_freq"] = freq
    return out
