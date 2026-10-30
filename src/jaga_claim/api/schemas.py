"""Schema request/response sesuai JAGA-CLAIM ML API Contract v1 (POST /v1/score)."""
from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FORBIDDEN_KEYS = {"fkl48", "verified_cost", "biaya_verifikasi", "verification_cost"}


class Episode(BaseModel):
    model_config = ConfigDict(extra="allow")
    admission_date: dt.date = Field(..., description="FKL03 tanggal datang")
    discharge_date: Optional[dt.date] = Field(None, description="FKL04 tanggal pulang (wajib untuk scoring)")
    service_level: Optional[Literal["RJTL", "RITL"]] = Field(None, description="FKL10; diturunkan dari INA-CBG bila kosong")
    provider_type: Optional[str] = Field(None, description="FKL09 kode tipe FKRTL")
    provider_region: Optional[str] = Field(None, description="FKL05_NEW kode provinsi FKRTL")
    care_class: Optional[str] = Field(None, description="FKL13 kelas rawat (1/2/3)")
    tariff_regional: Optional[str] = Field(None, description="FKL31 regional tarif INA-CBG (1-5)")
    provider_kind: Optional[str] = Field(None, description="FKL08")
    provider_ownership: Optional[str] = Field(None, description="FKL07")
    service_unit: Optional[str] = Field(None, description="FKL11 poli")
    segment: Optional[str] = Field(None, description="FKL12")
    discharge_status: Optional[str] = Field(None, description="FKL14 (1 sehat, 2 rujuk, 3 meninggal, 4 pulang paksa, 5 tidak tahu)")
    patient_sex: Optional[Literal["M", "F"]] = Field(None, description="Opsional; default dari referensi kepesertaan tim ML")
    patient_birth_date: Optional[dt.date] = Field(None, description="Opsional; default dari referensi kepesertaan tim ML")


class Diagnoses(BaseModel):
    model_config = ConfigDict(extra="allow")
    primary_icd10: str = Field(..., min_length=3, description="FKL18")
    admission_icd10: Optional[str] = Field(None, description="FKL16")
    secondary_icd10: list[str] = Field(default_factory=list, description="FKL24 per FKL02")


class Casemix(BaseModel):
    model_config = ConfigDict(extra="allow")
    inacbg_code: str = Field(..., min_length=3, description="FKL19, contoh 'A-4-14-I'")
    severity: Optional[str] = Field(None, description="Opsional: 0/I/II/III atau kode FKL23 1-4")


class Procedures(BaseModel):
    model_config = ConfigDict(extra="allow")
    procedure_type: Optional[str] = Field(None, description="FKL30 (teks/kode, dipisah ';')")
    special_codes: dict[str, Optional[str]] = Field(default_factory=dict, description="sa/sp/rr/si/sd -> kode special")


class FinancialPreSubmit(BaseModel):
    model_config = ConfigDict(extra="allow")
    billed_cost: float = Field(..., ge=0, description="FKL47 biaya tagih provider")
    cbg_tariff: Optional[float] = Field(None, ge=0, description="FKL32")
    special_tariffs: dict[str, float] = Field(default_factory=dict, description="sa/sp/rr/si/sd -> tarif")

    @model_validator(mode="before")
    @classmethod
    def no_post_adjudication(cls, v):
        if isinstance(v, dict) and any(str(k).lower() in FORBIDDEN_KEYS for k in v):
            raise ValueError("FKL48/biaya verifikasi dilarang sebagai input pra-submit")
        return v


class ScoreRequest(BaseModel):
    claim_id: str = Field(..., min_length=1)
    participant_key: str = Field(..., min_length=1)
    prediction_time: dt.datetime
    episode: Episode
    diagnoses: Diagnoses
    casemix: Casemix
    procedures: Procedures
    financial_pre_submit: FinancialPreSubmit
    history_features: dict = Field(default_factory=dict)
    peer_features: dict = Field(default_factory=dict)

    @field_validator("prediction_time")
    @classmethod
    def utc_required(cls, v: dt.datetime):
        if v.tzinfo is None or v.utcoffset() != dt.timedelta(0):
            raise ValueError("prediction_time harus ISO 8601 UTC (akhiran Z)")
        return v


class Evidence(BaseModel):
    feature: str
    value: str
    reference: str


class StakeholderReason(BaseModel):
    code: str
    title: str
    meaning: str
    suggested_check: str


class StakeholderEvidence(BaseModel):
    evidence_index: int
    title: str
    observed_value: str
    comparison: str


class StakeholderExplanation(BaseModel):
    version: Literal['stakeholder-v1'] = 'stakeholder-v1'
    source: Literal['template'] = 'template'
    priority_label: str
    summary: str
    score_meaning: str
    reasons: list[StakeholderReason]
    evidence: list[StakeholderEvidence]
    suggested_next_steps: list[str]
    limitations: list[str]


class ScoreResponse(BaseModel):
    claim_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_band: Literal["low", "medium", "high"]
    reason_codes: list[Literal["COST_PEER_OUTLIER", "LOS_PEER_OUTLIER", "TEMPORAL_REPEAT",
                               "POSSIBLE_DUPLICATE", "CODING_MISMATCH", "DATA_QUALITY"]]
    evidence: list[Evidence]
    model_version: str
    rule_version: str
    scored_at: str
    stakeholder_explanation: Optional[StakeholderExplanation] = None


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
