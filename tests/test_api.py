"""Uji kontrak API v1: 3 fixture wajib (sukses ber-flag, sukses tanpa flag, gagal/ML tidak tersedia) + validasi."""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jaga_claim.api.app import app
from jaga_claim.api.schemas import ScoreResponse

EX = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def load(name):
    return json.loads((EX / name).read_text())


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_flagged_success(client):
    r = client.post("/v1/score", json=load("request_flagged_duplicate.json"))
    assert r.status_code == 200, r.text
    body = ScoreResponse(**r.json())
    assert "POSSIBLE_DUPLICATE" in body.reason_codes and body.risk_band in ("medium", "high")
    assert body.evidence and body.model_version and body.rule_version and body.scored_at.endswith("Z")
    assert body.stakeholder_explanation is not None
    assert body.stakeholder_explanation.source == 'template'
    assert any(r.code == 'POSSIBLE_DUPLICATE' for r in body.stakeholder_explanation.reasons)
    assert len(body.stakeholder_explanation.evidence) == len(body.evidence)


def test_no_flag_success(client):
    r = client.post("/v1/score", json=load("request_low_risk.json"))
    assert r.status_code == 200, r.text
    body = ScoreResponse(**r.json())
    assert body.reason_codes == [] and body.risk_band == "low"


def test_contract_example_demo_codes(client):
    r = client.post("/v1/score", json=load("request_contract_demo.json"))
    assert r.status_code == 200, r.text
    ScoreResponse(**r.json())


def test_validation_error_422(client):
    bad = load("request_low_risk.json"); del bad["diagnoses"]["primary_icd10"]
    r = client.post("/v1/score", json=bad)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_fkl48_forbidden(client):
    bad = load("request_low_risk.json"); bad["financial_pre_submit"]["FKL48"] = 1
    r = client.post("/v1/score", json=bad)
    assert r.status_code == 422


def test_prediction_time_must_be_utc(client):
    bad = load("request_low_risk.json"); bad["prediction_time"] = "2024-11-05T10:00:00+07:00"
    assert client.post("/v1/score", json=bad).status_code == 422


def test_missing_discharge_is_insufficient(client):
    bad = load("request_low_risk.json"); bad["episode"]["discharge_date"] = None
    r = client.post("/v1/score", json=bad)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INSUFFICIENT_DATA"


def test_bad_json_400(client):
    r = client.post("/v1/score", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_service_unavailable_503(client):
    svc = client.app.state.service
    client.app.state.service = None
    try:
        r = client.post("/v1/score", json=load("request_low_risk.json"))
        assert r.status_code == 503 and r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    finally:
        client.app.state.service = svc


def test_deterministic(client):
    a = client.post("/v1/score", json=load("request_flagged_duplicate.json")).json()
    b = client.post("/v1/score", json=load("request_flagged_duplicate.json")).json()
    a.pop("scored_at"); b.pop("scored_at")
    assert a == b
