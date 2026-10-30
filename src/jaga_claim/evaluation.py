"""Evaluasi tanpa label (Dokumentasi hlm. 10-13).

Karena dataset tidak memiliki label fraud/pending terverifikasi, evaluasi MVP terdiri dari:
 1. Profil operasional: distribusi band, flag rate per reason code per bulan.
 2. Stabilitas: PSI skor antar periode; stabilitas ranking antar seed (Spearman, Jaccard top-K).
 3. Subgroup: high-band rate per region / kelas / tipe provider / tingkat layanan.
 4. Uji skenario sintetis TERPISAH pada test set terkunci (injeksi pola yang diketahui) ->
    Precision@K, Recall@K, Lift@K untuk benchmark rules-only, peer-only, duplicate-only,
    anomaly-only, hybrid, dan ablation (hybrid tanpa satu engine).
    Hasil ini membuktikan kemampuan deteksi pola yang diketahui, BUKAN akurasi fraud.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import REASON_CODES, load_config, resolve
from .engine import HybridRiskEngine
from .features import build_features
from .pipeline import HISTORY_COLS, load_features, log

ENGINES = ["rule", "peer", "duplicate", "anomaly"]


# ------------------------------------------------------------------ helpers
def psi(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    pa = np.histogram(a, bins)[0] / len(a) + 1e-6
    pb = np.histogram(b, bins)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def recombine(O: pd.DataFrame, cfg: dict, disable=()) -> np.ndarray:
    w, fl = cfg["fusion"]["weights"], cfg["fusion"]["floors"]
    comp = {"rule": O["R"], "peer": O["P"], "duplicate": O["D"], "anomaly": O["A"]}
    S = sum(w[k] * comp[k].to_numpy() for k in comp if k not in disable)
    floor = np.zeros(len(O))
    if "duplicate" not in disable:
        floor = np.maximum(floor, O["flag_exact_duplicate"].to_numpy() * fl["exact_duplicate"])
    if "rule" not in disable:
        floor = np.maximum(floor, O["rule_critical"].to_numpy() * fl["critical_rule"])
    return np.maximum(S, floor)


def topk_metrics(score: np.ndarray, is_pos: np.ndarray, case_id: np.ndarray, k: int, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    tieb = rng.random(len(score))
    order = np.lexsort((tieb, -score))
    top = np.zeros(len(score), dtype=bool)
    top[order[:k]] = True
    prec = is_pos[top].mean()
    cases = pd.Series(top[is_pos]).groupby(case_id[is_pos]).any()
    rec = cases.mean()
    base = is_pos.mean()
    return {"precision": float(prec), "recall_case": float(rec), "lift": float(prec / base) if base else None}


# ------------------------------------------------------- synthetic scenarios
def make_scenarios(cur: pd.DataFrame, clean_ids: set, train_cbg_median: pd.Series, n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    test = cur[cur["claim_id"].isin(clean_ids)]
    injected, removed = [], set()

    def pick(mask, k):
        idx = test.index[mask.reindex(test.index, fill_value=False)]
        idx = [i for i in idx if test.at[i, "claim_id"] not in removed]
        return test.loc[rng.choice(idx, size=min(k, len(idx)), replace=False)]

    def add(rows, scen, replace_original=True, suffix="SYN"):
        for j, (_, r) in enumerate(rows.iterrows()):
            r = r.copy()
            orig = r["claim_id"]
            r["claim_id"] = f"{orig}_{suffix}{j}"
            r["_scenario"], r["_case"] = scen, f"{scen}:{orig}"
            injected.append(r)
            if replace_original:
                removed.add(orig)
    # a. Inflasi biaya via komponen special fiktif
    s = pick(test["billed_cost"] > 0, n).copy()
    extra = np.maximum(3 * s["billed_cost"], 2_000_000).round()
    s["sp_code"], s["sp_tariff"], s["billed_cost"] = "SP-SYN", extra, s["billed_cost"] + extra
    s["procedure_text"] = s["procedure_text"].fillna("9999 - Synthetic procedure")
    add(s, "A_COST_INFLATION")
    # b. Inflasi LOS rawat inap
    s = pick(test["service_level"] == "RITL", n).copy()
    s["discharge_date"] = s["discharge_date"] + pd.Timedelta(days=12)
    add(s, "B_LOS_INFLATION")
    # c. Duplikat persis (klaim ganda)
    s = pick(test["billed_cost"] > 0, n).copy()
    for j, (_, r) in enumerate(s.iterrows()):
        r = r.copy(); orig = r["claim_id"]
        r["claim_id"] = f"{orig}_DUP"; r["_scenario"], r["_case"] = "C_EXACT_DUPLICATE", f"C:{orig}"
        injected.append(r)
    # d. Upcoding keparahan pada rawat inap singkat
    m = (test["service_level"] == "RITL") & ((pd.to_datetime(test["discharge_date"]) - pd.to_datetime(test["admission_date"])).dt.days <= 1) \
        & test["inacbg_code"].astype(str).str.match(r".+-(I|II)$")
    s = pick(m, n).copy()
    newc = s["inacbg_code"].astype(str).str.replace(r"-(I|II)$", "-III", regex=True)
    s["inacbg_code"], s["severity"] = newc, "4"
    s["discharge_status"] = "1"
    med = newc.map(train_cbg_median)
    s["cbg_tariff"] = med.fillna(s["cbg_tariff"] * 2.0).round()
    s["billed_cost"] = s["cbg_tariff"]
    add(s, "D_SEVERITY_UPCODING")
    # e. Diagnosis tidak sesuai jenis kelamin
    s = pick(test["sex"] == "M", n).copy()
    s["primary_icd10"], s["primary_icd10_3"] = "O800", "O80"
    add(s, "E_SEX_DX_MISMATCH")
    # f. Kunjungan berulang (burst 10 kunjungan CBG sama dalam 10 hari)
    s = pick(test["service_level"] == "RJTL", max(1, n // 5))
    for _, r in s.iterrows():
        for d in range(1, 11):
            q = r.copy(); orig = r["claim_id"]
            q["claim_id"] = f"{orig}_REP{d:02d}"
            q["admission_date"] = r["admission_date"] + pd.Timedelta(days=d)
            q["discharge_date"] = q["admission_date"]
            q["_scenario"], q["_case"] = "F_REPEAT_BURST", f"F:{orig}"
            injected.append(q)
    # g. Klaim rawat jalan saat pasien masih dirawat inap
    m = (test["service_level"] == "RITL") & ((pd.to_datetime(test["discharge_date"]) - pd.to_datetime(test["admission_date"])).dt.days >= 3)
    s = pick(m, n)
    for _, r in s.iterrows():
        q = r.copy(); orig = r["claim_id"]
        q["claim_id"] = f"{orig}_OVL"
        q["admission_date"] = r["admission_date"] + pd.Timedelta(days=1)
        q["discharge_date"] = q["admission_date"]
        q.update({"service_level": "RJTL", "inacbg_code": "Q-5-44-0", "cmg": "17", "case_type": "5", "severity": "1",
                  "primary_icd10": "Z098", "primary_icd10_3": "Z09", "cbg_tariff": 200000, "billed_cost": 200000,
                  "procedure_text": pd.NA, "sec_dx": "", "sec_dx3": ""})
        q["_scenario"], q["_case"] = "G_OVERLAP_INPATIENT", f"G:{orig}"
        injected.append(q)
    # h. Anomali multivariat (tanpa satu aturan tunggal yang jelas)
    s = pick(test["service_level"] == "RITL", n).copy()
    chapters = ["E11", "I10", "J18", "K29", "N39", "M54", "R50", "L03"]
    s["sec_dx3"] = "|".join(chapters); s["sec_dx"] = "|".join(chapters)
    s["procedure_text"] = "8901 - A;8903 - B;9059 - C;8744 - D;9904 - E"
    s["si_code"], s["si_tariff"] = "SI-SYN", (0.25 * s["billed_cost"]).round()
    s["sd_code"], s["sd_tariff"] = "SD-SYN", (0.25 * s["billed_cost"]).round()
    s["billed_cost"] = s["billed_cost"] + s["si_tariff"] + s["sd_tariff"]
    add(s, "H_MULTIVARIATE")
    inj = pd.DataFrame(injected)
    return inj, removed


EXPECTED_CODE = {"A_COST_INFLATION": "COST_PEER_OUTLIER", "B_LOS_INFLATION": "LOS_PEER_OUTLIER",
                 "C_EXACT_DUPLICATE": "POSSIBLE_DUPLICATE", "D_SEVERITY_UPCODING": "CODING_MISMATCH",
                 "E_SEX_DX_MISMATCH": "CODING_MISMATCH", "F_REPEAT_BURST": "TEMPORAL_REPEAT",
                 "G_OVERLAP_INPATIENT": "TEMPORAL_REPEAT", "H_MULTIVARIATE": None}


def synthetic_eval(cfg, eng, F, S, n_per=250):
    cur = pd.read_parquet(resolve(cfg["paths"]["curated_dir"]) / "claims_curated.parquet")
    rc_cols = [f"rc_{c}" for c in REASON_CODES]
    clean = S[(S["split"] == "test") & (S["risk_band"] == "low") & ~S[rc_cols].any(axis=1)]["claim_id"]
    tr = F[F["split"] == "train"]
    cbg_med = tr.groupby("inacbg_code")["billed_cost"].median()
    inj, removed = make_scenarios(cur, set(clean), cbg_med, n_per)
    log(f"[eval] injeksi {len(inj):,} baris sintetis ({inj['_scenario'].nunique()} skenario)")
    parts = set(inj["participant_key"])
    base = cur[cur["participant_key"].isin(parts) & ~cur["claim_id"].isin(removed)]
    allrows = pd.concat([base, inj.drop(columns=["_scenario", "_case"])], ignore_index=True)
    Fp = build_features(allrows, cfg)
    Fp = Fp[(pd.to_datetime(Fp["admission_date"]) > pd.Timestamp(cfg["split"]["valid_end"]))]
    # pool test = test asli (peserta tak terdampak) + peserta terdampak (dihitung ulang)
    Ft = F[(F["split"] == "test") & ~F["participant_key"].isin(parts)]
    pool = pd.concat([Ft.drop(columns=["split"]), Fp], ignore_index=True)
    O = eng.score_frame(pool)
    lab = pool["claim_id"].map(inj.set_index("claim_id")["_case"])
    is_pos = lab.notna().to_numpy()
    case_id = lab.fillna("").to_numpy()
    scen = pool["claim_id"].map(inj.set_index("claim_id")["_scenario"])
    N = len(pool)
    Ks = {"top_0.5%": int(0.005 * N), "top_1%": int(0.01 * N), "top_5%": int(0.05 * N)}
    variants = {"rules_only": O["R"].to_numpy(), "peer_only": O["P"].to_numpy(),
                "duplicate_temporal_only": O["D"].to_numpy(), "anomaly_only": O["A"].to_numpy(),
                "hybrid": O["risk_score"].to_numpy()}
    for e in ENGINES:
        variants[f"hybrid_minus_{e}"] = recombine(O, cfg, disable=(e,))
    bench = {v: {k: topk_metrics(sc, is_pos, case_id, kk) for k, kk in Ks.items()} for v, sc in variants.items()}
    # recall per skenario @ top 5% (hybrid vs engine tunggal)
    per_s = {}
    for sname in sorted(scen.dropna().unique()):
        m = (scen == sname).to_numpy()
        row = {}
        for v in ["rules_only", "peer_only", "duplicate_temporal_only", "anomaly_only", "hybrid"]:
            sc = variants[v]
            order = np.lexsort((np.random.default_rng(0).random(N), -sc))
            top = np.zeros(N, dtype=bool); top[order[:Ks["top_5%"]]] = True
            row[v] = float(pd.Series(top[m]).groupby(case_id[m]).any().mean())
        exp = EXPECTED_CODE[sname]
        if exp:
            row["expected_reason_code"] = exp
            row["reason_code_hit_rate"] = float(pd.Series(O.loc[m, f"rc_{exp}"].to_numpy()).groupby(case_id[m]).any().mean())
        row["band_high_or_medium"] = float(pd.Series((O.loc[m, "risk_band"] != "low").to_numpy()).groupby(case_id[m]).any().mean())
        row["n_cases"] = int(pd.Series(case_id[m]).nunique())
        per_s[sname] = row
    return {"pool_size": N, "n_injected_rows": int(is_pos.sum()), "n_cases": int(pd.Series(case_id[is_pos]).nunique()),
            "K": Ks, "benchmark": bench, "per_scenario": per_s}


def stability_eval(cfg, F, seeds=(1, 2, 3, 4, 5)):
    """Latih ulang engine dengan seed berbeda; ukur kestabilan ranking di test."""
    tr, va, te = F[F["split"] == "train"], F[F["split"] == "valid"], F[F["split"] == "test"]
    scores = {}
    for s in seeds:
        c = json.loads(json.dumps(cfg)); c["anomaly"]["random_state"] = s
        e = HybridRiskEngine(c).fit(tr, va)
        scores[s] = (e.score_frame(te)["risk_score"].to_numpy(), e.thresholds["high"])
        log(f"[eval] stabilitas seed {s} selesai")
    from scipy.stats import spearmanr
    keys = list(scores)
    k = int(0.01 * len(te))
    sp, jac, thr = [], [], [scores[x][1] for x in keys]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = scores[keys[i]][0], scores[keys[j]][0]
            nz = (a > 0) | (b > 0)
            sp.append(spearmanr(a[nz], b[nz]).statistic)
            ta, tb = set(np.argsort(-a, kind="stable")[:k]), set(np.argsort(-b, kind="stable")[:k])
            jac.append(len(ta & tb) / len(ta | tb))
    return {"seeds": list(seeds), "spearman_nonzero_mean": float(np.mean(sp)), "spearman_min": float(np.min(sp)),
            "jaccard_top1pct_mean": float(np.mean(jac)), "jaccard_top1pct_min": float(np.min(jac)),
            "high_threshold_range": [float(min(thr)), float(max(thr))]}


def run_evaluation(cfg=None, stability=True):
    cfg = cfg or load_config()
    t0 = time.time()
    rep_dir = resolve(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    F = load_features(cfg)
    S = pd.read_parquet(resolve("data/scored") / "scores_all.parquet")
    eng = HybridRiskEngine.load(resolve(cfg["paths"]["artifacts_dir"]) / cfg["versions"]["model_version"])
    rc_cols = [f"rc_{c}" for c in REASON_CODES]
    R: dict = {"model_version": cfg["versions"]["model_version"], "thresholds": eng.thresholds, "meta": eng.meta}

    # 1. Profil operasional
    R["band_by_split"] = S.groupby("split")["risk_band"].value_counts(normalize=True).unstack().round(4).to_dict("index")
    R["band_counts_by_split"] = S.groupby("split")["risk_band"].value_counts().unstack().to_dict("index")
    R["reason_rate_by_split"] = S.groupby("split")[rc_cols].mean().round(5).to_dict("index")
    S["month"] = pd.to_datetime(S["admission_date"]).dt.to_period("M").astype(str)
    mon = S[S["month"] >= "2024-01"].groupby("month").agg(n=("claim_id", "size"), high=("risk_band", lambda x: (x == "high").mean()),
                                                          medium_plus=("risk_band", lambda x: (x != "low").mean()),
                                                          **{c: (c, "mean") for c in rc_cols})
    R["monthly"] = mon.round(5).reset_index().to_dict("records")
    hb = S[S["risk_band"] != "low"]
    R["reason_code_coverage"] = {
        b: {"n": int((S["risk_band"] == b).sum()),
            "with_reason_code": float(S.loc[S["risk_band"] == b, rc_cols].any(axis=1).mean())} for b in ["high", "medium"]}
    R["components_mean_by_band"] = S.groupby("risk_band")[["R", "P", "D", "A"]].mean().round(4).to_dict("index")
    # 2. Stabilitas distribusi (PSI)
    bins = np.r_[-1e-9, np.quantile(S.loc[S["split"] == "train", "risk_score"], np.linspace(0.5, 1, 11))[:-1], 1.0 + 1e-9]
    bins = np.unique(bins)
    g = {k: S.loc[S["split"] == k, "risk_score"].to_numpy() for k in ["train", "valid", "test"]}
    R["psi"] = {"train_vs_valid": psi(g["train"], g["valid"], bins), "valid_vs_test": psi(g["valid"], g["test"], bins),
                "train_vs_test": psi(g["train"], g["test"], bins)}
    # 3. Subgroup (test)
    te = S[S["split"] == "test"]
    sub = {}
    for col in ["service_level", "care_class", "provider_type", "provider_region"]:
        x = te.groupby(col).agg(n=("claim_id", "size"), high_rate=("risk_band", lambda s: (s == "high").mean()),
                                medplus_rate=("risk_band", lambda s: (s != "low").mean()))
        x = x[x["n"] >= 500].sort_values("high_rate", ascending=False)
        sub[col] = x.round(4).reset_index().to_dict("records")
    R["subgroup_test"] = sub
    # 4. Skenario sintetis
    R["synthetic"] = synthetic_eval(cfg, eng, F, S)
    # 5. Stabilitas antar seed
    if stability:
        R["seed_stability"] = stability_eval(cfg, F)
    # 6. Rule hit counts (test)
    rule_cols = [c for c in S.columns if c[:2] in ("DQ", "CM") and c[2:4].isdigit()]
    R["rule_hits_all"] = {c: int(S[c].sum()) for c in rule_cols}
    R["elapsed_sec"] = round(time.time() - t0, 1)
    with open(rep_dir / "evaluation_report.json", "w") as f:
        json.dump(R, f, indent=2, default=str, ensure_ascii=False)
    log(f"[eval] laporan -> {rep_dir / 'evaluation_report.json'} ({R['elapsed_sec']} dtk)")
    return R


if __name__ == "__main__":
    run_evaluation()
