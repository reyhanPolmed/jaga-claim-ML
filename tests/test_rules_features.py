"""Golden-case test setiap rule + point-in-time leakage test + engine duplikasi/temporal."""
import pandas as pd

from conftest import frame, make_claim
from jaga_claim.engines.rules import RULES, evaluate_rules, parse_cbg
from jaga_claim.features import build_features


def rules_for(cfg, rows, pt=None):
    F = build_features(frame(rows), cfg)
    return F, evaluate_rules(F, pt)


def test_clean_claim_has_no_rule(cfg):
    _, R = rules_for(cfg, [make_claim()])
    assert R["rule_score"].iloc[0] == 0


GOLDEN = {
    "DQ01_MISSING_CRITICAL": dict(billed_cost=0.0),
    "DQ02_MISSING_CONTEXT": dict(provider_type="99"),
    "DQ03_DATE_INVALID": dict(discharge_date=pd.Timestamp("2024-11-01")),
    "DQ04_AGE_INVALID": dict(birth_date=pd.Timestamp("2025-01-01")),
    "DQ06_TARIFF_RECON": dict(billed_cost=900000.0),
    "DQ07_SPECIAL_CODE_NO_TARIFF": dict(sp_code="SP-01", sp_tariff=0.0),
    "CM01_SEX_DX": dict(primary_icd10="O800", primary_icd10_3="O80", sex="M"),
    "CM02_AGE_DX": dict(primary_icd10="P071", primary_icd10_3="P07", sex="F"),
    "CM03_SEX_CBG": dict(inacbg_code="O-6-10-I", cmg="15", case_type="6", severity="2", service_level="RITL",
                         discharge_date=pd.Timestamp("2024-11-07")),
    "CM04_PROC_CBG_NO_PROC": dict(inacbg_code="N-3-15-0", cmg="14", case_type="3"),
    "CM05_HIGH_SEV_SHORT_LOS": dict(inacbg_code="J-4-16-III", cmg="10", case_type="4", severity="4", service_level="RITL",
                                    discharge_date=pd.Timestamp("2024-11-06")),
    "CM06_SETTING_MISMATCH": dict(service_level="RITL", discharge_date=pd.Timestamp("2024-11-07")),
    "CM07_CBG_COMPONENT_MISMATCH": dict(severity="3"),
}


def test_every_rule_has_golden_case():
    assert {r.rule_id for r in RULES} - {"DQ05_SERVICE_AFTER_DEATH"} == set(GOLDEN)


def test_golden_cases(cfg):
    for rid, kw in GOLDEN.items():
        _, R = rules_for(cfg, [make_claim(**kw)])
        assert bool(R[rid].iloc[0]), rid


def test_service_after_death(cfg):
    a = make_claim(claim_id="CLM_A", service_level="RITL", inacbg_code="J-4-16-I", cmg="10", case_type="4", severity="2",
                   discharge_status="3", discharge_date=pd.Timestamp("2024-11-08"))
    b = make_claim(claim_id="CLM_B", admission_date=pd.Timestamp("2024-11-20"), discharge_date=pd.Timestamp("2024-11-20"))
    F, R = rules_for(cfg, [a, b])
    assert R.loc[F["claim_id"] == "CLM_B", "DQ05_SERVICE_AFTER_DEATH"].item()
    assert not R.loc[F["claim_id"] == "CLM_A", "DQ05_SERVICE_AFTER_DEATH"].item()


def test_parse_cbg():
    p = parse_cbg(pd.Series(["A-4-14-I", "Q-5-44-0", "K-1-12-III"]))
    assert list(p["cmg_from_code"]) == ["1", "17", "11"]
    assert list(p["severity_from_code"]) == ["2", "1", "4"]


def test_point_in_time_no_future_leakage(cfg):
    """Menambah episode MASA DEPAN tidak boleh mengubah fitur episode sebelumnya."""
    rows = [make_claim(claim_id=f"CLM_{i}", admission_date=pd.Timestamp("2024-11-01") + pd.Timedelta(days=3 * i),
                       discharge_date=pd.Timestamp("2024-11-01") + pd.Timedelta(days=3 * i)) for i in range(4)]
    F1 = build_features(frame(rows[:3]), cfg).set_index("claim_id")
    F2 = build_features(frame(rows), cfg).set_index("claim_id").loc[F1.index]
    cols = ["prior_visits_30d", "same_cbg_visits_30d", "days_since_last_visit", "dup_fuzzy_score", "overlap_prior_inpatient"]
    pd.testing.assert_frame_equal(F1[cols], F2[cols])


def test_exact_duplicate_and_overlap(cfg):
    a = make_claim(claim_id="CLM_A")
    b = make_claim(claim_id="CLM_B")                    # identik -> duplikat persis
    inp = make_claim(claim_id="CLM_C", participant_key="PST_2", service_level="RITL", inacbg_code="J-4-16-I", cmg="10",
                     case_type="4", severity="2", discharge_date=pd.Timestamp("2024-11-10"))
    rj = make_claim(claim_id="CLM_D", participant_key="PST_2", admission_date=pd.Timestamp("2024-11-07"),
                    discharge_date=pd.Timestamp("2024-11-07"))
    F = build_features(frame([a, b, inp, rj]), cfg).set_index("claim_id")
    assert F.loc["CLM_B", "dup_exact"] == 1 and F.loc["CLM_B", "dup_exact_ref"] == "CLM_A"
    assert F.loc["CLM_A", "dup_exact"] == 0                    # yang pertama tidak ditandai
    assert F.loc["CLM_D", "overlap_prior_inpatient"] == 1


def test_unknown_categories_do_not_crash(cfg):
    F = build_features(frame([make_claim(inacbg_code="X-9-99-0", provider_type="ZZZ", primary_icd10="U999", primary_icd10_3="U99")]), cfg)
    assert len(F) == 1
