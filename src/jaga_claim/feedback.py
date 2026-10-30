"""Feedback loop: Hasil Review -> Label Store -> Validasi & Model Improvement (E4).

Input: ekspor keputusan reviewer dari website (CSV) dengan kolom minimal
    claim_id, review_action (confirm|dismiss|request_correction), reviewer_id, reviewed_at, model_version
    [opsional] label_code (0 no issue, 1 administrative correction, 2 clinical/coding review,
               3 inefficiency confirmed, 4 suspected fraud escalation)

Keluaran (reports/feedback_report.json):
  * precision per band & precision@K terhadap label reviewer, confirm-rate per reason code,
  * agreement antar reviewer (Cohen's kappa) bila ada klaim yang dinilai >1 reviewer,
  * bila label cukup: Logistic Regression atas komponen [R, P, D, A] dengan GroupKFold per peserta
    -> USULAN bobot fusi baru (configs/candidate_weights.yaml). Tidak pernah dipromosikan otomatis:
    wajib locked-test report + persetujuan ML lead & risk owner (Dokumentasi hlm. 16).

    python -m jaga_claim.feedback path/to/feedback.csv [--scores data/scored/scores_all.parquet]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from .config import REASON_CODES, load_config, resolve

ACTIONS = {"confirm": 1, "request_correction": 1, "dismiss": 0}
MIN_LABELS = 200


def rounded_fusion_weights(weights: np.ndarray, decimals: int = 3) -> np.ndarray:
    """Round non-negative fusion weights while preserving an exact total of 1.

    Rounding each component independently can produce totals such as 1.001,
    which is invalid when the result is copied back into the fusion config.
    Allocate the remaining units by largest fractional remainder instead.
    """
    scale = 10 ** decimals
    base = np.floor(weights * scale).astype(int)
    remaining = int(scale - base.sum())
    if remaining:
        fractional = weights * scale - base
        for i in np.argsort(-fractional, kind="stable")[:remaining]:
            base[i] += 1
    return base.astype(float) / scale


def load_feedback(path: str) -> pd.DataFrame:
    fb = pd.read_csv(path, dtype={"claim_id": str, "reviewer_id": str})
    need = {"claim_id", "review_action", "reviewer_id", "reviewed_at"}
    miss = need - set(fb.columns)
    if miss:
        raise ValueError(f"kolom feedback hilang: {miss}")
    fb = fb[fb["review_action"].isin(ACTIONS)].copy()
    fb["y"] = fb["review_action"].map(ACTIONS)
    if "label_code" in fb:
        fb.loc[fb["label_code"].notna(), "y"] = (fb.loc[fb["label_code"].notna(), "label_code"] >= 1).astype(int)
    return fb


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def analyze(fb: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> dict:
    rep: dict = {"n_feedback_rows": int(len(fb))}
    multi = fb.groupby("claim_id").filter(lambda g: g["reviewer_id"].nunique() >= 2)
    if len(multi):
        pairs = multi.sort_values("reviewed_at").groupby("claim_id")["y"].agg(["first", "last"])
        rep["inter_reviewer_kappa"] = cohen_kappa(pairs["first"].to_numpy(), pairs["last"].to_numpy())
        rep["n_double_reviewed"] = int(len(pairs))
    lab = fb.sort_values("reviewed_at").groupby("claim_id").tail(1)       # label terakhir per klaim
    d = lab.merge(scores, on="claim_id", how="inner")
    rep["n_labeled_claims"] = int(len(d))
    rep["positive_rate"] = float(d["y"].mean()) if len(d) else None
    if not len(d):
        return rep
    rep["precision_by_band"] = d.groupby("risk_band")["y"].agg(["size", "mean"]).round(4).to_dict("index")
    d = d.sort_values("risk_score", ascending=False)
    rep["precision_at_k"] = {k: float(d["y"].head(k).mean()) for k in (10, 50, 100, 500) if len(d) >= k}
    rep["confirm_rate_by_reason_code"] = {c: float(d.loc[d[f"rc_{c}"], "y"].mean()) for c in REASON_CODES
                                          if f"rc_{c}" in d and d[f"rc_{c}"].sum() >= 5}
    if len(d) >= MIN_LABELS and d["y"].nunique() == 2 and "participant_key" in d:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import GroupKFold
        X = d[["R", "P", "D", "A"]].to_numpy()
        y = d["y"].to_numpy()
        oof = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=5).split(X, y, d["participant_key"]):
            m = LogisticRegression(class_weight="balanced").fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        m = LogisticRegression(class_weight="balanced").fit(X, y)
        w = np.clip(m.coef_[0], 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(4, 0.25)
        w = rounded_fusion_weights(w)
        rep["supervised_baseline"] = {"pr_auc_current_score": float(average_precision_score(y, d["risk_score"])),
                                      "pr_auc_logreg_oof": float(average_precision_score(y, oof)),
                                      "suggested_weights": dict(zip(["rule", "peer", "duplicate", "anomaly"], w.tolist())),
                                      "status": "KANDIDAT - perlu locked-test & persetujuan sebelum dipromosikan"}
        with open(resolve("configs") / "candidate_weights.yaml", "w") as f:
            yaml.safe_dump({"fusion": {"weights": rep["supervised_baseline"]["suggested_weights"]},
                            "status": rep["supervised_baseline"]["status"]}, f)
    else:
        rep["supervised_baseline"] = f"belum dijalankan: butuh >= {MIN_LABELS} klaim berlabel dengan dua kelas"
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("feedback_csv")
    ap.add_argument("--scores", default="data/scored/scores_all.parquet")
    a = ap.parse_args(argv)
    cfg = load_config()
    rep = analyze(load_feedback(a.feedback_csv), pd.read_parquet(resolve(a.scores)), cfg)
    out = resolve(cfg["paths"]["reports_dir"]) / "feedback_report.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
