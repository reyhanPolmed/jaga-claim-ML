"""Pemuatan konfigurasi dan utilitas bersama (pseudonimisasi, path)."""
from __future__ import annotations

import hashlib
import hmac
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

REASON_CODES = (
    "COST_PEER_OUTLIER",
    "LOS_PEER_OUTLIER",
    "TEMPORAL_REPEAT",
    "POSSIBLE_DUPLICATE",
    "CODING_MISMATCH",
    "DATA_QUALITY",
)


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    p = Path(path or os.getenv("JAGA_CONFIG", DEFAULT_CONFIG))
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def _salt(cfg: dict) -> bytes:
    env = cfg["pseudonym"]["salt_env"]
    return os.getenv(env, cfg["pseudonym"]["dev_salt"]).encode()


def pseudonymize(values, cfg: dict, prefix: str):
    """HMAC-SHA256 ter-salt; satu arah. Dipakai untuk PSTV01, FKL02, FKP02.

    Identifier mentah tidak boleh keluar dari zona raw (Dokumentasi hlm. 16).
    """
    key = _salt(cfg)
    n = cfg["pseudonym"]["length"]
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "":
            out.append(None)
            continue
        d = hmac.new(key, f"{prefix}:{v}".encode(), hashlib.sha256).hexdigest()[:n]
        out.append(f"{prefix}_{d}")
    return out
