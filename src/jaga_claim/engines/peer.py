"""Engine 2 — Peer-Group Statistical Model (median + MAD, robust deviation).

Statistik peer di-FIT hanya dari periode TRAIN (mencegah leakage, hlm. 12).
Cohort berjenjang (paling spesifik -> umum); dipakai level pertama yang memiliki
n >= min_cohort. Skala robust = max(1.4826 * MAD, floor) karena tarif INA-CBG
cenderung flat (MAD sering 0).

Hanya ekor ATAS yang dinilai (biaya/LOS/frekuensi lebih tinggi dari peer), karena
itulah yang relevan untuk prioritas review. LOS sangat pendek dengan keparahan
tinggi ditangani rule CM05.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    k = df[cols[0]].astype("string").fillna("<NA>")
    for c in cols[1:]:
        k = k + "§" + df[c].astype("string").fillna("<NA>")
    return k.astype(object)


def _mad(x: pd.Series) -> float:
    m = x.median()
    return float((x - m).abs().median())


def fit_peer_table(train: pd.DataFrame, value: str, levels: list[list[str]], min_n: int,
                   quantiles=(0.5, 0.95, 0.99, 0.995)) -> list[dict]:
    tables = []
    for lv, cols in enumerate(levels):
        k = _key(train, cols)
        g = train.groupby(k)[value]
        t = pd.DataFrame({"n": g.size(), "median": g.median()})
        dev = (train[value] - k.map(t["median"]).astype(float)).abs()
        t["mad"] = dev.groupby(k).median()
        for q in quantiles:
            t[f"q{q}"] = g.quantile(q)
        t = t[t["n"] >= min_n]
        tables.append({"level": lv, "cols": cols, "table": t})
    return tables


def lookup(df: pd.DataFrame, tables: list[dict]) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index, columns=["peer_level", "peer_cohort", "n", "median", "mad", "q0.95", "q0.99", "q0.995"], dtype=object)
    res["peer_level"] = np.nan
    todo = pd.Series(True, index=df.index)
    for t in tables:
        if not todo.any():
            break
        k = _key(df.loc[todo], t["cols"])
        hit = k.isin(t["table"].index)
        idx = hit[hit].index
        if len(idx):
            m = t["table"].loc[k.loc[idx].values]
            for c in ["n", "median", "mad", "q0.95", "q0.99", "q0.995"]:
                if c in m:
                    res.loc[idx, c] = m[c].values
            res.loc[idx, "peer_level"] = t["level"]
            res.loc[idx, "peer_cohort"] = "+".join(t["cols"])
            todo.loc[idx] = False
    for c in ["n", "median", "mad", "q0.95", "q0.99", "q0.995", "peer_level"]:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


class PeerModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg["peer"]
        self.tcfg = cfg["temporal"]
        self.cost_tables = self.los_tables = self.freq_tables = None

    def fit(self, train: pd.DataFrame) -> "PeerModel":
        c = self.cfg
        self.cost_tables = fit_peer_table(train, "billed_cost", c["cost_levels"], c["min_cohort"])
        inp = train[train["service_level"] == "RITL"]
        self.los_tables = fit_peer_table(inp, "los_days", c["los_levels"], c["min_cohort"])
        self.freq_tables = fit_peer_table(train, "same_cbg_visits_30d", c["freq_levels"], c["min_cohort"])
        return self

    def _score(self, z, gap, min_gap):
        """Skor halus 0..1: 1 - exp(-(z - z_start)/tau), di-gate oleh selisih absolut
        (deviasi relatif besar pada nominal kecil tidak boleh mendominasi prioritas)."""
        c = self.cfg
        zz = np.clip(np.asarray(z, dtype=float) - c["z_start"], 0, None)
        gate = np.clip(np.asarray(gap, dtype=float) / min_gap, 0, 1)
        return np.nan_to_num((1 - np.exp(-zz / c["z_tau"])) * gate)

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        out = pd.DataFrame(index=F.index)
        # ---- biaya ----
        lc = lookup(F, self.cost_tables)
        med = lc["median"]
        scale = np.maximum(c["mad_const"] * lc["mad"], c["cost_scale_floor_frac"] * med)
        z = (F["billed_cost"] - med) / scale
        out["cost_peer_median"] = med
        out["cost_peer_scale"] = scale
        out["cost_peer_n"] = lc["n"]
        out["cost_peer_p95"] = lc["q0.95"]
        out["cost_peer_cohort"] = lc["peer_cohort"]
        out["cost_robust_z"] = z
        out["cost_peer_score"] = self._score(z.fillna(0), (F["billed_cost"] - med).fillna(0), c["min_abs_cost_gap"])
        out["flag_cost_peer"] = ((z >= c["flag_z"]) & ((F["billed_cost"] - med) >= c["min_abs_cost_gap"])).fillna(False)
        # ---- LOS (rawat inap saja) ----
        inp = F["service_level"] == "RITL"
        ll = lookup(F, self.los_tables)
        lmed = ll["median"]
        lscale = np.maximum(c["mad_const"] * ll["mad"], c["los_scale_floor_days"])
        lz = ((F["los_days"] - lmed) / lscale).where(inp)
        out["los_peer_median"] = lmed.where(inp)
        out["los_peer_p95"] = ll["q0.95"].where(inp)
        out["los_peer_n"] = ll["n"].where(inp)
        out["los_peer_cohort"] = ll["peer_cohort"].where(inp)
        out["los_robust_z"] = lz
        out["los_peer_score"] = self._score(lz.fillna(0), (F["los_days"] - lmed).where(inp).fillna(0), c["min_abs_los_gap"])
        out["flag_los_peer"] = ((lz >= c["flag_z"]) & ((F["los_days"] - lmed) >= c["min_abs_los_gap"])).fillna(False)
        out["peer_score"] = np.maximum(out["cost_peer_score"], out["los_peer_score"])
        # ---- frekuensi kunjungan CBG sama 30 hari (dipakai Engine 3 / temporal) ----
        lf = lookup(F, self.freq_tables)
        qcol = f"q{self.tcfg['freq_peer_quantile']}"
        thr = lf[qcol] if qcol in lf else lf["q0.995"]
        out["freq_peer_threshold"] = thr
        out["freq_peer_median"] = lf["median"]
        v = F["same_cbg_visits_30d"]
        out["freq_excess"] = ((v - thr) / np.maximum(thr, 1)).clip(lower=0)
        out["flag_freq_peer"] = ((v > thr) & (v >= self.tcfg["freq_min_visits"])).fillna(False)
        out["cost_peer_available"] = med.notna()
        return out
