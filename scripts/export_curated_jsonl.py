"""Export bounded curated FKRTL snapshots for the JAGA-CLAIM website importer.

The website owns its audit/review database, while this ML project owns source
curation and pseudonymization. This command exports only the transport fields
allowed by ``jaga-claim-curated-v1``; it never exports raw IDs, FKL48, history
features, peer features, scores, or model artefacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


CLAIM_ID = re.compile(r"^CLM_[0-9a-f]{20}$")
PARTICIPANT_KEY = re.compile(r"^PST_[0-9a-f]{20}$")
SPECIAL_KEYS = ("sa", "sp", "rr", "si", "sd")
OPTIONAL_EPISODE_FIELDS = (
    "service_level", "provider_type", "provider_region", "care_class",
    "tariff_regional", "provider_kind", "provider_ownership", "service_unit",
    "segment", "discharge_status",
)


def nonempty(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def date_text(value: Any, field: str) -> str:
    if isinstance(value, dt.datetime):
        value = value.date()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = nonempty(value)
    if not text:
        raise ValueError(f"{field} wajib tersedia")
    try:
        return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} bukan tanggal ISO valid") from exc


def number(value: Any, field: str, required: bool = False) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if required:
            raise ValueError(f"{field} wajib tersedia")
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} harus finite dan >= 0")
    return parsed


def normalized_snapshot(value: str) -> str:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--snapshot-at harus ISO 8601 UTC dengan akhiran Z") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0) or not value.endswith("Z"):
        raise argparse.ArgumentTypeError("--snapshot-at harus ISO 8601 UTC dengan akhiran Z")
    return parsed.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_record(row: dict[str, Any], *, source_revision: int, snapshot_at: str) -> dict[str, Any]:
    claim_id = nonempty(row.get("claim_id"))
    participant_key = nonempty(row.get("participant_key"))
    if not claim_id or not CLAIM_ID.fullmatch(claim_id):
        raise ValueError("claim_id tidak sesuai ^CLM_[0-9a-f]{20}$")
    if not participant_key or not PARTICIPANT_KEY.fullmatch(participant_key):
        raise ValueError("participant_key tidak sesuai ^PST_[0-9a-f]{20}$")

    episode = {
        "admission_date": date_text(row.get("admission_date"), "admission_date"),
        "discharge_date": date_text(row.get("discharge_date"), "discharge_date"),
    }
    for field in OPTIONAL_EPISODE_FIELDS:
        value = nonempty(row.get(field))
        if value is not None:
            episode[field] = value

    secondary = [code for code in (nonempty(row.get("sec_dx")) or "").split("|") if code]
    special_codes = {
        key: value for key in SPECIAL_KEYS
        if (value := nonempty(row.get(f"{key}_code"))) is not None
    }
    special_tariffs = {
        key: value for key in SPECIAL_KEYS
        if (value := number(row.get(f"{key}_tariff"), f"{key}_tariff")) not in (None, 0)
    }

    primary_icd10 = nonempty(row.get("primary_icd10"))
    inacbg_code = nonempty(row.get("inacbg_code"))
    if not primary_icd10 or not inacbg_code:
        raise ValueError("primary_icd10 dan inacbg_code wajib tersedia")

    diagnoses: dict[str, Any] = {"primary_icd10": primary_icd10, "secondary_icd10": secondary}
    admission_icd10 = nonempty(row.get("admission_icd10"))
    if admission_icd10 is not None:
        diagnoses["admission_icd10"] = admission_icd10

    casemix: dict[str, Any] = {"inacbg_code": inacbg_code}
    severity = nonempty(row.get("severity"))
    if severity is not None:
        casemix["severity"] = severity

    procedures: dict[str, Any] = {"special_codes": special_codes}
    procedure_type = nonempty(row.get("procedure_text"))
    if procedure_type is not None:
        procedures["procedure_type"] = procedure_type

    financial: dict[str, Any] = {
        "billed_cost": number(row.get("billed_cost"), "billed_cost", required=True),
        "special_tariffs": special_tariffs,
    }
    cbg_tariff = number(row.get("cbg_tariff"), "cbg_tariff")
    if cbg_tariff is not None:
        financial["cbg_tariff"] = cbg_tariff

    return {
        "schema_version": "jaga-claim-curated-v1",
        "claim_id": claim_id,
        "participant_key": participant_key,
        "source_revision": source_revision,
        "snapshot_at": snapshot_at,
        "episode": episode,
        "diagnoses": diagnoses,
        "casemix": casemix,
        "procedures": procedures,
        "financial_pre_submit": financial,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export curated ML records as bounded website JSONL.")
    parser.add_argument("--input", type=Path, default=Path("data/curated/claims_curated.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-at", type=normalized_snapshot, required=True,
                        help="source snapshot UTC, e.g. 2026-10-27T05:00:00Z")
    parser.add_argument("--source-revision", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000,
                        help="maximum 50,000 rows to match the website import limit")
    parser.add_argument("--force", action="store_true", help="allow replacing an existing output file")
    args = parser.parse_args()

    if args.source_revision < 1 or args.offset < 0 or not 1 <= args.limit <= 50_000:
        parser.error("source-revision minimal 1, offset minimal 0, dan limit harus 1..50000")
    if not args.input.is_file():
        parser.error(f"input tidak ditemukan: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output sudah ada: {args.output}; gunakan --force bila memang akan diganti")

    columns = [
        "claim_id", "participant_key", "admission_date", "discharge_date", *OPTIONAL_EPISODE_FIELDS,
        "primary_icd10", "admission_icd10", "sec_dx", "inacbg_code", "severity", "procedure_text",
        "billed_cost", "cbg_tariff", *[f"{key}_code" for key in SPECIAL_KEYS],
        *[f"{key}_tariff" for key in SPECIAL_KEYS],
    ]
    parquet = pq.ParquetFile(args.input)
    missing = sorted(set(columns) - set(parquet.schema.names))
    if missing:
        parser.error(f"schema curated tidak lengkap: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    row_number = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for batch in parquet.iter_batches(batch_size=1_024, columns=columns):
            for row in batch.to_pylist():
                row_number += 1
                if skipped < args.offset:
                    skipped += 1
                    continue
                if written >= args.limit:
                    break
                try:
                    record = to_record(row, source_revision=args.source_revision, snapshot_at=args.snapshot_at)
                except ValueError as exc:
                    raise SystemExit(f"baris curated {row_number} tidak dapat diekspor: {exc}") from exc
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
            if written >= args.limit:
                break

    print(f"export selesai: {written} baris -> {args.output}")


if __name__ == "__main__":
    main()
