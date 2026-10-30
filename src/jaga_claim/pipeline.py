"""Orkestrasi pipeline end-to-end.

    python -m jaga_claim.pipeline all        # ingest -> features -> train -> score -> evaluate
    python -m jaga_claim.pipeline ingest
    python -m jaga_claim.pipeline features
    python -m jaga_claim.pipeline train      # fit train, threshold di valid, simpan artefak
    python -m jaga_claim.pipeline score      # skor seluruh klaim + antrean review
    python -m jaga_claim.pipeline evaluate   # laporan evaluasi (reports/)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import REASON_CODES, load_config, resolve
from .engine import HybridRiskEngine
from .features import build_features
from .ingestion import build_curated

HISTORY_COLS = ["claim_id", "participant_key", "admission_date", "discharge_date", "service_level", "provider_type",
                "provider_region", "provider_kind", "provider_ownership", "care_class", "tariff_regional", "service_unit",
                "segment", "discharge_status", "admission_icd10", "admission_icd10_3", "primary_icd10", "primary_icd10_3",
                "inacbg_code", "cmg", "case_type", "specificity", "severity", "procedure_text", "sec_dx", "sec_dx3",
                "cbg_tariff", "sa_code", "sa_tariff", "sp_code", "sp_tariff", "rr_code", "rr_tariff", "si_code",
                "si_tariff", "sd_code", "sd_tariff", "billed_cost", "sex", "birth_date"]


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def split_of(F: pd.DataFrame, cfg: dict) -> pd.Series:
    a = pd.to_datetime(F["admission_date"])
    te, ve = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["valid_end"])
    return pd.Series(np.where(a <= te, "train", np.where(a <= ve, "valid", "test")), index=F.index)


def step_ingest(cfg):
    build_curated(cfg, log=log)


def step_features(cfg):
    cur = pd.read_parquet(resolve(cfg["paths"]["curated_dir"]) / "claims_curated.parquet")
    log(f"[features] membangun feature mart untuk {len(cur):,} klaim ...")
    t = time.time()
    F = build_features(cur, cfg)
    F["split"] = split_of(F, cfg)
    out = resolve(cfg["paths"]["feature_dir"])
    out.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out / "feature_mart.parquet", index=False)
    # history store utk scoring online (zona ML, ter-pseudonim)
    cur[[c for c in HISTORY_COLS if c in cur.columns]].to_parquet(out / "history_store.parquet", index=False)
    log(f"[features] selesai {F.shape} dalam {time.time() - t:.0f} dtk; split: {F['split'].value_counts().to_dict()}")


def load_features(cfg) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg["paths"]["feature_dir"]) / "feature_mart.parquet")


def step_train(cfg, F=None):
    F = load_features(cfg) if F is None else F
    tr, va = F[F["split"] == "train"], F[F["split"] == "valid"]
    log(f"[train] train={len(tr):,} valid={len(va):,} (test terkunci tidak dipakai)")
    eng = HybridRiskEngine(cfg).fit(tr, va)
    path = resolve(cfg["paths"]["artifacts_dir"]) / cfg["versions"]["model_version"]
    eng.save(path)
    (resolve(cfg["paths"]["artifacts_dir"]) / "ACTIVE").write_text(cfg["versions"]["model_version"])
    log(f"[train] artefak -> {path}; thresholds {eng.thresholds}")
    return eng


def step_score(cfg, F=None, eng=None, topk_export=500):
    F = load_features(cfg) if F is None else F
    eng = eng or HybridRiskEngine.load(resolve(cfg["paths"]["artifacts_dir"]) / cfg["versions"]["model_version"])
    log("[score] menskor seluruh klaim ...")
    O = eng.score_frame(F)
    keep = ["R", "P", "D", "A", "S_weighted", "risk_score", "risk_band", "anomaly_percentile", "rule_critical"] + \
           [f"rc_{c}" for c in REASON_CODES] + [r for r in O.columns if r[:2] in ("DQ", "CM") and r[2:4].isdigit()]
    S = pd.concat([F[["claim_id", "participant_key", "admission_date", "split", "service_level", "provider_type",
                      "provider_region", "care_class", "inacbg_code", "billed_cost"]], O[keep]], axis=1)
    out = resolve("data/scored")
    out.mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / "scores_all.parquet", index=False)
    # Antrean review contoh: top-K periode test, lengkap dengan reason/evidence (format kontrak API)
    test_idx = S.index[S["split"] == "test"]
    top = S.loc[test_idx].sort_values("risk_score", ascending=False).head(topk_export).index
    queue = [eng.to_response(F.loc[i], O.loc[i], scored_at="batch") for i in top]
    with open(out / "review_queue_test_top500.json", "w") as f:
        json.dump(queue, f, indent=1, ensure_ascii=False, default=str)
    log(f"[score] selesai; band: {S['risk_band'].value_counts().to_dict()}")
    return S, O


def main(argv=None):
    ap = argparse.ArgumentParser(description="JAGA-CLAIM pipeline")
    ap.add_argument("step", choices=["ingest", "features", "train", "score", "evaluate", "all"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.step in ("ingest", "all"):
        step_ingest(cfg)
    if a.step in ("features", "all"):
        step_features(cfg)
    F = eng = None
    if a.step in ("train", "all"):
        F = load_features(cfg)
        eng = step_train(cfg, F)
    if a.step in ("score", "all"):
        step_score(cfg, F, eng)
    if a.step in ("evaluate", "all"):
        from .evaluation import run_evaluation
        run_evaluation(cfg)


if __name__ == "__main__":
    main()
