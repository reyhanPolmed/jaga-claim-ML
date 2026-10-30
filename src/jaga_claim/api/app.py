"""JAGA-CLAIM ML Scoring API v1 (FastAPI).

Jalankan:
    PYTHONPATH=src uvicorn jaga_claim.api.app:app --host 0.0.0.0 --port 8000

Endpoint kontrak : POST /v1/score
Endpoint operasional: GET /health, GET /v1/metadata
Endpoint opsional (di luar kontrak MVP): POST /v1/explain (ringkasan bahasa reviewer;
  LLM hanya merangkum reason code yang sudah dihitung, dengan fallback deterministik).
Autentikasi opsional: set JAGA_ML_SERVICE_TOKEN untuk mewajibkan Bearer token pada /v1/*.

Prinsip: decision-support. risk_score = prioritas review, bukan probabilitas fraud.
"""
from __future__ import annotations

import logging
import os
import time
from hmac import compare_digest
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from .schemas import ErrorResponse, ScoreRequest, ScoreResponse
from .service import InsufficientData, ScoringService
from .explanation import explain_score
from .presentation import stakeholder_explanation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("jaga-claim.api")


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    """Require a server-to-server bearer token when the service is configured with one.

    Local loopback development can leave ``JAGA_ML_SERVICE_TOKEN`` empty. Any
    non-local deployment must set it and use the same value as the website's
    ``ML_SERVICE_TOKEN``; the browser never receives this secret.
    """
    expected = os.getenv("JAGA_ML_SERVICE_TOKEN")
    if not expected:
        return
    provided = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not provided or not compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Service authentication diperlukan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    app.state.load_error = None
    try:  # Load once, serve many
        app.state.service = ScoringService()
    except Exception as exc:  # fail-safe: API hidup, scoring 503
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        log.error("Model/artefak gagal dimuat: %s", app.state.load_error)
    yield


app = FastAPI(title="JAGA-CLAIM ML Scoring API", version="v1", lifespan=lifespan,
              description="Prioritisasi review klaim FKRTL pra-submit. Bukan keputusan fraud otomatis.")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errs):
        return _err(400, "BAD_REQUEST", "Body bukan JSON yang valid")
    msg = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}" for e in errs[:5])
    return _err(422, "VALIDATION_ERROR", msg)


@app.get("/health")
def health(request: Request):
    svc = request.app.state.service
    return {"status": "ok" if svc else "unavailable", "api_version": "v1", "package_version": __version__,
            "model_version": svc.engine.versions["model_version"] if svc else None,
            "error": request.app.state.load_error}


@app.get("/v1/metadata", dependencies=[Depends(require_service_token)])
def metadata(request: Request):
    svc = request.app.state.service
    if not svc:
        return _err(503, "SERVICE_UNAVAILABLE", "Model belum siap")
    e = svc.engine
    return {"versions": e.versions, "thresholds": e.thresholds, "weights": e.cfg["fusion"]["weights"],
            "floors": e.cfg["fusion"]["floors"], "trained": e.meta, "loaded_at": svc.loaded_at,
            "history_episodes": int(len(svc.history)),
            "reason_codes": ["COST_PEER_OUTLIER", "LOS_PEER_OUTLIER", "TEMPORAL_REPEAT", "POSSIBLE_DUPLICATE",
                             "CODING_MISMATCH", "DATA_QUALITY"]}


@app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(require_service_token)],
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def score(req: ScoreRequest, request: Request, include_components: bool = Query(False, description="debug: skor R/P/D/A")):
    svc: ScoringService | None = request.app.state.service
    if svc is None:
        return _err(503, "SERVICE_UNAVAILABLE", "ML service belum siap/tidak tersedia")
    t = time.perf_counter()
    try:
        out = svc.score(req)
        out['stakeholder_explanation'] = stakeholder_explanation(ScoreResponse(**out)).model_dump()
    except InsufficientData as exc:
        return _err(422, "INSUFFICIENT_DATA", str(exc))
    except Exception as exc:  # jangan pernah mengganti kegagalan dengan skor rekaan
        log.exception("scoring gagal")
        return _err(503, "SCORING_FAILED", f"{type(exc).__name__}")
    # log TANPA identifier mentah (claim_id sudah ter-pseudonim)
    log.info("scored claim=%s band=%s score=%.4f codes=%s %.0fms", out["claim_id"], out["risk_band"], out["risk_score"],
             ",".join(out["reason_codes"]) or "-", 1000 * (time.perf_counter() - t))
    if include_components:
        return JSONResponse({**out, "_components": svc.components(req)})
    return out


@app.post("/v1/explain", dependencies=[Depends(require_service_token)])
async def explain(resp: ScoreResponse):
    """Ringkasan opsional, tidak mengubah skor atau keputusan reviewer."""
    return await explain_score(resp)
