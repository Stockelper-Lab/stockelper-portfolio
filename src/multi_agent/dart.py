from __future__ import annotations

import ast
import os
import re
from typing import Any


_SPLIT_RE = re.compile(r"[,\s]+")


def parse_open_dart_api_keys() -> list[str]:
    """OPEN_DART_API_KEYS/OPEN_DART_API_KEY에서 DART API Key 리스트를 파싱합니다.

    지원 포맷:
    - OPEN_DART_API_KEYS=key1,key2,key3
    - OPEN_DART_API_KEYS=key1 key2 key3
    - (하위호환) OPEN_DART_API_KEY=key1[,key2,...]
    """
    raw = (os.getenv("OPEN_DART_API_KEYS") or os.getenv("OPEN_DART_API_KEY") or "").strip()
    if not raw:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(raw) if p and p.strip()]

    # Deduplicate (preserve order)
    seen: set[str] = set()
    keys: list[str] = []
    for k in parts:
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
    return keys


def mask_api_key(key: str) -> str:
    """로그에 키를 그대로 남기지 않기 위한 마스킹."""
    k = (key or "").strip()
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def _try_parse_dict_from_error(err: Any) -> dict[str, Any] | None:
    if isinstance(err, dict):
        return err

    text = str(err)
    if not text:
        return None

    # 가장 흔한 형태: "{'status': '020', 'message': '...'}"
    if text.lstrip().startswith("{") and text.rstrip().endswith("}"):
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def is_dart_quota_exceeded_error(err: Any) -> bool:
    """OpenDartReader에서 status=020(사용한도 초과)인지 판별합니다."""
    data = _try_parse_dict_from_error(err)
    if isinstance(data, dict):
        status = str(data.get("status", "") or "").strip()
        if status == "020":
            return True

    text = str(err)
    if not text:
        return False

    # 보수적으로 문자열도 체크
    if "020" in text and ("사용한도" in text or "status" in text):
        return True
    return False


