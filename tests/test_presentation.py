from jaga_claim.api.presentation import REASONS, stakeholder_explanation
from jaga_claim.api.schemas import ScoreResponse


def result(**updates):
    data = dict(claim_id='synthetic', risk_score=.6, risk_band='high',
                reason_codes=['POSSIBLE_DUPLICATE'],
                evidence=[dict(feature='dup_exact_ref', value='comparison-id', reference='original evidence')],
                model_version='test', rule_version='test', scored_at='2026-01-01T00:00:00Z')
    data.update(updates)
    return ScoreResponse(**data)


def test_explains_without_mutating_or_fabricating_evidence_links():
    score = result()
    original = score.model_dump()
    text = stakeholder_explanation(score)
    assert text.priority_label == 'Tinggi'
    assert text.reasons[0].code == 'POSSIBLE_DUPLICATE'
    assert text.evidence[0].evidence_index == 0
    assert text.evidence[0].observed_value == 'comparison-id'
    assert text.evidence[0].comparison == 'original evidence'
    assert score.model_dump() == original


def test_empty_flags_does_not_claim_safe():
    text = stakeholder_explanation(result(risk_band='low', reason_codes=[], evidence=[]))
    assert 'tidak menjamin' in text.summary
    assert text.reasons == [] and text.evidence == []


def test_anomaly_without_reason_code_has_distinct_explanation():
    text = stakeholder_explanation(result(reason_codes=[], evidence=[dict(
        feature='anomaly_score', value='persentil 99', reference='kombinasi fitur')]))
    assert 'tidak lazim' in text.summary
    assert text.reasons == []


def test_all_existing_reason_codes_are_supported():
    text = stakeholder_explanation(result(reason_codes=list(REASONS)))
    assert len(text.reasons) == 6
