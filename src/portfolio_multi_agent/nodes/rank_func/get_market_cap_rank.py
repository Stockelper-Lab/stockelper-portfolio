"""
한국투자증권 시가총액 순위 API - 시가총액 순위
"""

from __future__ import annotations

from typing import Dict, List
import os
import aiohttp
import asyncio
import random
import time


# ---- module-level cache / rate limit ----
# 시가총액 랭킹은 초 단위로 변하지 않으므로, 짧은 TTL 캐시로 호출 폭주를 방지합니다.
_CACHE_AT: float = 0.0  # monotonic seconds
_CACHE_DATA: list[dict[str, str]] = []

# 동시 호출을 직렬화하고, 최소 호출 간격을 강제합니다.
_FETCH_LOCK = asyncio.Lock()
_NEXT_ALLOWED_AT: float = 0.0  # monotonic seconds


def _now() -> float:
    return time.monotonic()


def _analysis_rps() -> float:
    """분석용 KIS RPS(초당 최대 호출 수).

    - 우선: KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND
    - 폴백: KIS_MAX_REQUESTS_PER_SECOND
    - 최종: 1.0
    """
    for name, default in (
        ("KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND", ""),
        ("KIS_MAX_REQUESTS_PER_SECOND", "1"),
    ):
        raw = (os.getenv(name) or default).strip()
        if not raw:
            continue
        try:
            v = float(raw)
            if v > 0:
                return v
        except Exception:
            continue
    return 1.0


def _cache_ttl_seconds() -> float:
    raw = (os.getenv("KIS_RANK_CACHE_TTL_SECONDS") or "30").strip()
    try:
        v = float(raw)
        return v if v > 0 else 30.0
    except Exception:
        return 30.0


def _cache_max_age_seconds() -> float:
    """API 호출 실패 시 사용할 수 있는 'stale 캐시' 최대 나이."""
    raw = (os.getenv("KIS_RANK_CACHE_MAX_AGE_SECONDS") or "300").strip()
    try:
        v = float(raw)
        return v if v > 0 else 300.0
    except Exception:
        return 300.0


def _is_rate_limit_error(api_response: dict) -> bool:
    msg = str(api_response.get("msg1", "") or "")
    msg_cd = str(api_response.get("msg_cd", "") or "")
    # KIS에서 자주 보이는 rate limit 메시지/코드 패턴
    if "초당" in msg and "초과" in msg:
        return True
    if "거래건수" in msg and "초과" in msg:
        return True
    if msg_cd in {"EGW00133"}:
        return True
    return False


async def get_market_cap_rank(
    top_n: int = 30,
    app_key: str = "",
    app_secret: str = "",
    access_token: str = "",
) -> List[Dict[str, str]]:
    """
    시가총액 순위 조회 및 종목 정보 리스트 반환

    Args:
        top_n: 상위 몇 개의 종목을 가져올지 (기본값: 30)
        app_key: 한국투자증권 API App Key
        app_secret: 한국투자증권 API App Secret
        access_token: 한국투자증권 API Access Token

    Returns:
        시가총액 순위 상위 N개 종목 정보 리스트 [{"code": "종목코드", "name": "종목명"}, ...]
    """
    global _CACHE_AT, _CACHE_DATA, _NEXT_ALLOWED_AT
    n = int(top_n or 0)
    if n <= 0:
        return []

    # 빠른 경로: 캐시가 유효하면 즉시 반환
    now = _now()
    ttl = _cache_ttl_seconds()
    if _CACHE_DATA and (now - _CACHE_AT) <= ttl:
        return list(_CACHE_DATA[:n])

    _app_key = app_key
    _app_secret = app_secret
    _access_token = access_token

    base_url = (
        os.getenv("KIS_API_BASE_URL") or "https://openapivts.koreainvestment.com:29443"
    ).rstrip("/")
    url = f"{base_url}/uapi/domestic-stock/v1/ranking/market-cap"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {_access_token}",
        "appkey": _app_key,
        "appsecret": _app_secret,
        "tr_id": "FHPST01740000",
        "custtype": "P",
    }
    params = {
        "fid_input_price_2": "",
        "fid_cond_mrkt_div_code": "J",
        "fid_cond_scr_div_code": "20174",
        "fid_div_cls_code": "0",
        "fid_input_iscd": "0000",
        "fid_trgt_cls_code": "0",
        "fid_trgt_exls_cls_code": "0",
        "fid_input_price_1": "",
        "fid_vol_cnt": "",
    }

    # 호출 폭주 방지: 직렬화 + 최소 간격 + 백오프 재시도
    async with _FETCH_LOCK:
        # lock 획득 후 캐시가 갱신되었을 수도 있으니 재확인
        now = _now()
        if _CACHE_DATA and (now - _CACHE_AT) <= ttl:
            return list(_CACHE_DATA[:n])

        rps = _analysis_rps()
        min_interval = 1.0 / float(rps)
        max_retries = int((os.getenv("KIS_RANK_RETRY_MAX") or "3").strip() or 3)

        last_err_msg = ""
        for attempt in range(max_retries + 1):
            # 최소 호출 간격 보장
            now = _now()
            if now < _NEXT_ALLOWED_AT:
                await asyncio.sleep(_NEXT_ALLOWED_AT - now)
            _NEXT_ALLOWED_AT = max(_NEXT_ALLOWED_AT, _now()) + min_interval

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(url, headers=headers, params=params) as response:
                        api_response = await response.json(content_type=None)
                except Exception as e:
                    last_err_msg = str(e)
                    # 네트워크/파싱 오류 시, stale 캐시가 있으면 그것으로 폴백
                    if _CACHE_DATA and (_now() - _CACHE_AT) <= _cache_max_age_seconds():
                        return list(_CACHE_DATA[:n])
                    raise

            # 응답 코드 확인
            if api_response.get("rt_cd") == "0":
                # Output: 시가총액 순위 리스트
                output = api_response.get("output", [])
                if not isinstance(output, list):
                    output = []

                # 상위 N개의 종목 정보 추출 (중복 제거, 순서 유지)
                seen = set()
                stock_list: list[dict[str, str]] = []
                for stock in output:
                    code = str(stock.get("mksc_shrn_iscd", "") or "").strip()
                    name = str(stock.get("hts_kor_isnm", "") or "").strip()
                    if code and code not in seen:
                        seen.add(code)
                        stock_list.append({"code": code, "name": name})

                # 캐시 갱신
                _CACHE_DATA = stock_list
                _CACHE_AT = _now()
                return list(stock_list[:n])

            # 오류: 메시지 저장
            last_err_msg = str(api_response.get("msg1", "알 수 없는 오류"))

            # rate limit이면 백오프 후 재시도, 또는 stale 캐시로 폴백
            if _is_rate_limit_error(api_response):
                if _CACHE_DATA and (_now() - _CACHE_AT) <= _cache_max_age_seconds():
                    return list(_CACHE_DATA[:n])
                if attempt < max_retries:
                    base = max(1.0, min_interval)
                    delay = base * (2**attempt) + random.uniform(0.0, 0.25)
                    await asyncio.sleep(delay)
                    continue

            # 기타 오류는 즉시 실패(토큰/권한/파라미터 등)
            raise ValueError(f"API 오류: {last_err_msg}")

        # 재시도 초과: stale 캐시가 있으면 폴백, 없으면 에러
        if _CACHE_DATA and (_now() - _CACHE_AT) <= _cache_max_age_seconds():
            return list(_CACHE_DATA[:n])
        raise ValueError(f"API 오류: {last_err_msg or '알 수 없는 오류'}")


# ===== 사용 예시 =====
if __name__ == "__main__":
    # 환경변수에서 인증 정보 로드
    app_key = os.getenv("KIS_APP_KEY", "")
    app_secret = os.getenv("KIS_APP_SECRET", "")
    access_token = os.getenv("KIS_ACCESS_TOKEN", "")

    # 시가총액 순위 상위 10개 종목 조회
    stock_codes = asyncio.run(
        get_market_cap_rank(
            top_n=10, app_key=app_key, app_secret=app_secret, access_token=access_token
        )
    )

    print("=" * 80)
    print(f"시가총액 순위 Top {len(stock_codes)} 종목 코드")
    print("=" * 80)
    print(stock_codes)
