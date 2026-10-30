"""Engine 4 — Unsupervised Anomaly Detection (Isolation Forest).

* Fit HANYA pada sampel periode train; preprocessing (imputasi, clipping, tabel
  kelangkaan dx-CBG) juga di-fit pada train.
* Output: anomaly_score (ECDF terhadap skor train, 0..1) -> BUKAN probabilitas fraud.
* Kontribusi ke fusi hanya dari ekor atas (default 5% teratas).
* Explainability: fitur dengan deviasi robust terbesar terhadap median train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg["anomaly"]
        self.features = list(self.cfg["features"])
        self.model = None
        self.medians = self.iqr = None
        self.clip_lo = self.clip_hi = None
        self.ecdf_q = None
        self.rarity = None
        self.rarity_default = None

    # ---- kelangkaan pasangan (INA-CBG, dx primer 3 digit) ----
    def _fit_rarity(self, train: pd.DataFrame):
        pair = train.groupby(["inacbg_code", "primary_icd10_3"], observed=True).size()
        cbg = train.groupby("inacbg_code", observed=True).size()
        r = -np.log((pair + 1) / (pair.index.get_level_values(0).map(cbg).to_numpy() + 2))
        self.rarity = r.rename("rar")
        self.rarity_default = float(np.log(len(train)))  # pasangan tidak pernah terlihat

    def _rarity(self, F: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([F["inacbg_code"].astype(object), F["primary_icd10_3"].astype(object)])
        v = self.rarity.reindex(idx).to_numpy(dtype=float)
        return np.where(np.isnan(v), self.rarity_default, v)

    def matrix(self, F: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=F.index)
        src = pd.concat([F, P[["cost_robust_z", "los_robust_z"]]], axis=1)
        for f in self.features:
            if f == "log_dx_cbg_rarity":
                X[f] = self._rarity(F)
            elif f == "severity":
                X[f] = pd.to_numeric(F["severity"], errors="coerce")
            else:
                X[f] = pd.to_numeric(src[f], errors="coerce")
        X["cost_robust_z"] = X["cost_robust_z"].fillna(0)
        X["los_robust_z"] = X["los_robust_z"].fillna(0)
        X["days_since_last_visit"] = X["days_since_last_visit"].fillna(365).clip(upper=365)
        X["tariff_recon_ratio"] = X["tariff_recon_ratio"].fillna(1.0)
        return X.astype(float)

    def fit(self, F_train: pd.DataFrame, P_train: pd.DataFrame) -> "AnomalyModel":
        self._fit_rarity(F_train)
        X = self.matrix(F_train, P_train)
        self.medians = X.median()
        X = X.fillna(self.medians)
        self.clip_lo = X.quantile(0.001)
        self.clip_hi = X.quantile(0.999)
        q75, q25 = X.quantile(0.75), X.quantile(0.25)
        self.iqr = (q75 - q25).where((q75 - q25) > 0, X.std().replace(0, 1.0))
        Xc = X.clip(self.clip_lo, self.clip_hi, axis=1)
        n = min(len(Xc), self.cfg["train_sample"])
        Xs = Xc.sample(n=n, random_state=self.cfg["random_state"])
        self.model = IsolationForest(n_estimators=self.cfg["n_estimators"], max_samples=self.cfg["max_samples"],
                                     random_state=self.cfg["random_state"], n_jobs=-1).fit(Xs.to_numpy())
        raw = -self.model.score_samples(Xs.to_numpy())
        self.ecdf_q = np.quantile(raw, np.linspace(0, 1, 2001))
        return self

    def transform(self, F: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
        X = self.matrix(F, P).fillna(self.medians)
        Xc = X.clip(self.clip_lo, self.clip_hi, axis=1)
        raw = -self.model.score_samples(Xc.to_numpy())
        pct = np.interp(raw, self.ecdf_q, np.linspace(0, 1, 2001))
        start = self.cfg["contribution_start_quantile"]
        out = pd.DataFrame(index=F.index)
        out["anomaly_raw"] = raw
        out["anomaly_percentile"] = pct
        out["anomaly_score"] = np.clip((pct - start) / (1 - start), 0, 1)
        dev = ((X - self.medians) / self.iqr).abs()
        top = np.argsort(-dev.to_numpy(), axis=1)[:, :3]
        cols = np.array(X.columns)
        out["anomaly_top_features"] = ["|".join(cols[r]) for r in top]
        self._last_X = X
        return out
