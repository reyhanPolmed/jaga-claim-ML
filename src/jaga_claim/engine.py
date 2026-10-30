"""Hybrid Risk Engine: 4 engine + score fusion + reason codes & evidence.

    S = w_r*R + w_p*P + w_d*D + w_a*A      (semua komponen sudah ternormalisasi 0..1)
    risk_score = max(S, floor kebijakan)    (floor hanya utk temuan deterministik berat)

risk_score adalah skor PRIORITAS review, bukan probabilitas fraud.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import REASON_CODES
from .engines.anomaly import AnomalyModel
from .engines.duplicate import duplicate_temporal_scores
from .engines.peer import PeerModel
from .engines.rules import RULE_BY_ID, RULES, evaluate_rules, rule_evidence

ANOMALY_LABEL = {
    "log_billed": "biaya tagih (log)", "cost_robust_z": "deviasi biaya vs peer", "los_days": "lama rawat",
    "los_robust_z": "deviasi LOS vs peer", "severity": "tingkat keparahan", "sec_dx_count": "jumlah dx sekunder",
    "sec_dx_chapter_count": "jumlah bab dx sekunder", "special_component_count": "jumlah komponen special",
    "special_share": "porsi tarif special", "procedure_count": "jumlah prosedur", "prior_visits_30d": "kunjungan 30 hari",
    "same_cbg_visits_30d": "kunjungan CBG sama 30 hari", "days_since_last_visit": "jarak dari kunjungan terakhir",
    "age_years": "usia", "tariff_recon_ratio": "rasio tagih/tarif", "is_inpatient": "rawat inap",
    "log_dx_cbg_rarity": "kelangkaan kombinasi dx-CBG", "admit_primary_same": "dx masuk = dx primer",
}


class HybridRiskEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.versions = cfg["versions"]
        self.peer = PeerModel(cfg)
        self.anomaly = AnomalyModel(cfg)
        self.thresholds: dict = {}
        self.meta: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, F_train: pd.DataFrame, F_valid: pd.DataFrame) -> "HybridRiskEngine":
        self.peer.fit(F_train)
        P_tr = self.peer.transform(F_train)
        self.anomaly.fit(F_train, P_tr)
        sv = self.score_frame(F_valid, apply_bands=False)["risk_score"]
        b = self.cfg["fusion"]["bands"]
        self.thresholds = {
            "high": float(np.quantile(sv, 1 - b["high_top_fraction"])),
            "medium": float(np.quantile(sv, 1 - b["medium_top_fraction"])),
            "basis": f"kuantil skor periode validasi (n={len(sv):,}); high=top {b['high_top_fraction']:.0%}, medium=top {b['medium_top_fraction']:.0%}",
        }
        # pastikan floor kebijakan minimal medium
        self.thresholds["medium"] = min(self.thresholds["medium"], self.cfg["fusion"]["floors"]["critical_rule"])
        self.meta = {"fitted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "n_train": int(len(F_train)), "n_valid": int(len(F_valid)),
                     "train_period": [str(F_train["admission_date"].min().date()), str(F_train["admission_date"].max().date())],
                     "valid_period": [str(F_valid["admission_date"].min().date()), str(F_valid["admission_date"].max().date())]}
        return self

    # ---------------------------------------------------------------- score
    def score_frame(self, F: pd.DataFrame, prediction_time: pd.Series | None = None, apply_bands: bool = True,
                    disable: tuple = ()) -> pd.DataFrame:
        w = self.cfg["fusion"]["weights"]
        fl = self.cfg["fusion"]["floors"]
        Rr = evaluate_rules(F, prediction_time)
        P = self.peer.transform(F)
        D = duplicate_temporal_scores(F, P, self.cfg)
        A = self.anomaly.transform(F, P)
        comp = {"rule": Rr["rule_score"].to_numpy(), "peer": P["peer_score"].to_numpy(),
                "duplicate": D["duplicate_score"].to_numpy(), "anomaly": A["anomaly_score"].to_numpy()}
        for k in disable:
            comp[k] = np.zeros(len(F))
        S = sum(w[k] * comp[k] for k in comp)
        floor = np.zeros(len(F))
        if "duplicate" not in disable:
            floor = np.maximum(floor, D["flag_exact_duplicate"].to_numpy() * fl["exact_duplicate"])
        if "rule" not in disable:
            floor = np.maximum(floor, Rr["rule_critical"].to_numpy() * fl["critical_rule"])
        out = pd.concat([Rr, P, D, A], axis=1)
        out["R"], out["P"], out["D"], out["A"] = comp["rule"], comp["peer"], comp["duplicate"], comp["anomaly"]
        out["S_weighted"] = S
        out["risk_score"] = np.clip(np.maximum(S, floor), 0, 1)
        # reason codes (flag boolean per kode)
        rc = pd.DataFrame(index=F.index)
        rc["COST_PEER_OUTLIER"] = P["flag_cost_peer"].to_numpy(dtype=bool) & ("peer" not in disable)
        rc["LOS_PEER_OUTLIER"] = P["flag_los_peer"].to_numpy(dtype=bool) & ("peer" not in disable)
        rc["TEMPORAL_REPEAT"] = D["flag_temporal_repeat"].to_numpy(dtype=bool) & ("duplicate" not in disable)
        rc["POSSIBLE_DUPLICATE"] = D["flag_possible_duplicate"].to_numpy(dtype=bool) & ("duplicate" not in disable)
        dq = [r.rule_id for r in RULES if r.reason_code == "DATA_QUALITY"]
        cm = [r.rule_id for r in RULES if r.reason_code == "CODING_MISMATCH"]
        rc["CODING_MISMATCH"] = Rr[cm].any(axis=1).to_numpy() & ("rule" not in disable)
        rc["DATA_QUALITY"] = Rr[dq].any(axis=1).to_numpy() & ("rule" not in disable)
        for c in REASON_CODES:
            out[f"rc_{c}"] = rc[c].to_numpy()
        if apply_bands and self.thresholds:
            out["risk_band"] = np.where(out["risk_score"] >= self.thresholds["high"], "high",
                                        np.where(out["risk_score"] >= self.thresholds["medium"], "medium", "low"))
        out.index = F.index
        return out

    # ------------------------------------------------------- explain (1 row)
    def explain(self, F_row: pd.Series, O_row: pd.Series) -> tuple[list[str], list[dict]]:
        rv = self.versions["rule_version"]
        codes = [c for c in REASON_CODES if bool(O_row.get(f"rc_{c}", False))]
        ev: list[dict] = []
        rp = lambda x: f"Rp{x:,.0f}".replace(",", ".")
        if "COST_PEER_OUTLIER" in codes:
            ev.append({"feature": "billed_cost", "value": rp(F_row["billed_cost"]),
                       "reference": (f"Median peer {rp(O_row['cost_peer_median'])}; skala robust {rp(O_row['cost_peer_scale'])}; "
                                     f"robust z = {O_row['cost_robust_z']:.1f}; p95 peer {rp(O_row['cost_peer_p95'])}; "
                                     f"cohort {O_row['cost_peer_cohort']} (n={int(O_row['cost_peer_n'])}); statistik periode train")})
        if "LOS_PEER_OUTLIER" in codes:
            ev.append({"feature": "los_days", "value": f"{F_row['los_days']:.0f} hari",
                       "reference": (f"Median peer {O_row['los_peer_median']:.0f} hari; p95 {O_row['los_peer_p95']:.0f} hari; "
                                     f"robust z = {O_row['los_robust_z']:.1f}; cohort {O_row['los_peer_cohort']} (n={int(O_row['los_peer_n'])})")})
        if "TEMPORAL_REPEAT" in codes:
            if O_row["flag_overlap"]:
                ev.append({"feature": "overlap_prior_inpatient", "value": "tanggal datang berada di dalam periode rawat inap sebelumnya",
                           "reference": "Episode rawat inap prior milik peserta yang sama belum pulang pada tanggal datang klaim ini"})
            if O_row["flag_readmit"]:
                ev.append({"feature": "readmit_gap_days", "value": f"{F_row['readmit_gap_days']:.0f} hari",
                           "reference": f"Rawat inap ulang dengan dx primer 3-digit sama ({F_row['primary_icd10_3']}); ambang <= {self.cfg['temporal']['readmit_days']} hari"})
            if O_row["flag_freq"]:
                ev.append({"feature": "same_cbg_visits_30d", "value": f"{F_row['same_cbg_visits_30d']:.0f} kunjungan sebelumnya",
                           "reference": f"INA-CBG sama dalam 30 hari; p99.5 peer = {O_row['freq_peer_threshold']:.0f}, median peer = {O_row['freq_peer_median']:.0f}"})
        if "POSSIBLE_DUPLICATE" in codes:
            if O_row["flag_exact_duplicate"]:
                ev.append({"feature": "dup_exact_ref", "value": str(F_row["dup_exact_ref"]),
                           "reference": "Episode pembanding (ter-pseudonim) identik: tgl datang & pulang, INA-CBG, dx primer, biaya tagih"
                                        + ("" if F_row.get("dup_exact_same_unit") == 1 else "; poli berbeda")})
            else:
                ev.append({"feature": "dup_fuzzy_ref", "value": str(F_row["dup_fuzzy_ref"]),
                           "reference": f"Similarity {F_row['dup_fuzzy_sim']:.2f}, selisih {F_row['dup_fuzzy_gap_days']:.0f} hari (dx, INA-CBG, dx sekunder, biaya, prosedur)"})
        if "CODING_MISMATCH" in codes or "DATA_QUALITY" in codes:
            ev.extend(rule_evidence(O_row, F_row, rv))
        if O_row["anomaly_score"] > 0:
            tops = [ANOMALY_LABEL.get(x, x) for x in str(O_row["anomaly_top_features"]).split("|")]
            ev.append({"feature": "anomaly_score", "value": f"persentil {100 * O_row['anomaly_percentile']:.1f}",
                       "reference": "Isolation Forest (kombinasi fitur tidak lazim, bukan probabilitas fraud); pendorong utama: " + ", ".join(tops)})
        if not bool(O_row.get("cost_peer_available", True)):
            ev.append({"feature": "peer_features", "value": "tidak tersedia",
                       "reference": f"Cohort INA-CBG {F_row['inacbg_code']} < {self.cfg['peer']['min_cohort']} klaim pada data train; skor peer = 0"})
        return codes, ev

    def to_response(self, F_row: pd.Series, O_row: pd.Series, scored_at: str | None = None) -> dict:
        codes, ev = self.explain(F_row, O_row)
        return {
            "claim_id": str(F_row["claim_id"]),
            "risk_score": round(float(O_row["risk_score"]), 4),
            "risk_band": str(O_row["risk_band"]),
            "reason_codes": codes,
            "evidence": ev,
            "model_version": self.versions["model_version"],
            "rule_version": self.versions["rule_version"],
            "scored_at": scored_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # -------------------------------------------------------------- persist
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "engine.joblib", compress=3)
        card = {"versions": self.versions, "thresholds": self.thresholds, "meta": self.meta,
                "weights": self.cfg["fusion"]["weights"], "floors": self.cfg["fusion"]["floors"],
                "rules": [{"rule_id": r.rule_id, "reason_code": r.reason_code, "weight": r.weight,
                           "description": r.description, "status": r.status} for r in RULES],
                "anomaly_features": self.anomaly.features,
                "intended_use": "Prioritisasi review klaim FKRTL pra-submit. Bukan keputusan fraud/penolakan klaim."}
        with open(path / "model_card.json", "w") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> "HybridRiskEngine":
        return joblib.load(Path(path) / "engine.joblib")
