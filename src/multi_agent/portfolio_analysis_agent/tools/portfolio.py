from typing import Dict, List, Optional, Tuple, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import time
import pandas as pd
import numpy as np
from ...utils import get_user_kis_credentials, get_access_token, Industy

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
import aiohttp
import logging
import os
import asyncio
import json
import OpenDartReader
from datetime import datetime

logger = logging.getLogger(__name__)

from multi_agent.dart import (
    is_dart_quota_exceeded_error,
    mask_api_key,
    parse_open_dart_api_keys,
)

_async_engine: AsyncEngine | None = None


def _to_async_db_url(url: str | None) -> str | None:
    """Convert Prisma-style DATABASE_URL to SQLAlchemy async URL if needed."""
    if not url:
        return None
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def get_async_engine() -> AsyncEngine:
    """Lazily create the primary async engine.

    이 모듈이 import 되는 순간에 환경변수 유무로 서버가 바로 죽지 않도록,
    실제 호출 시점에만 `ASYNC_DATABASE_URL`을 확인/초기화합니다.
    """
    global _async_engine

    if _async_engine is not None:
        return _async_engine

    async_db_url = os.getenv("ASYNC_DATABASE_URL") or _to_async_db_url(
        os.getenv("DATABASE_URL")
    )
    if not async_db_url:
        raise RuntimeError("Missing required environment variable: ASYNC_DATABASE_URL")

    _async_engine = create_async_engine(async_db_url, echo=False)
    return _async_engine

_async_engine_ksic: AsyncEngine | None = None
_async_engine_ksic_disabled: bool = False


def get_async_engine_ksic() -> AsyncEngine | None:
    """Lazily create the KSIC async engine.

    We intentionally do NOT hard-crash the whole API server when KSIC DB is not configured,
    because portfolio recommendations are not executed from the chatbot in this project.
    """
    global _async_engine_ksic, _async_engine_ksic_disabled

    # 런타임에서 KSIC DB 연결 실패가 한 번이라도 발생하면, 프로세스 생명주기 동안 비활성화합니다.
    # (대량 병렬 분석 시 매번 커넥션 실패로 전체 추천이 죽는 것을 방지)
    if _async_engine_ksic_disabled:
        return None

    if _async_engine_ksic is not None:
        return _async_engine_ksic

    async_database_url_ksic = os.getenv("ASYNC_DATABASE_URL_KSIC")
    if not async_database_url_ksic:
        logger.warning("ASYNC_DATABASE_URL_KSIC is not set; KSIC industry lookups will be disabled.")
        return None

    _async_engine_ksic = create_async_engine(async_database_url_ksic, echo=False)
    return _async_engine_ksic


MARKET_MAP = {
    "Y": "유가",
    "K": "코스닥",
    "N": "코넥스",
    "E": "기타"
}


class PortfolioAnalysisInput(BaseModel):
    user_investor_type: str = Field(
        description="The investor type of the user. It indicates the user's investment style or risk profile."
    )
    portfolio_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="추천받을 종목 개수(1~20). 지정하지 않으면 투자성향 기본값을 사용합니다.",
    )

class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyzes and recommends portfolio based on user's investor type. Evaluates stocks using market value, stability, profitability, and growth metrics, then suggests optimal portfolio composition tailored to the user's investment style."
    # KIS 모의/실전 base URL
    # - 기본값은 모의투자(openapivts)
    # - 실전 환경은 KIS_API_BASE_URL=https://openapi.koreainvestment.com:9443 로 설정
    url_base: str = os.getenv("KIS_API_BASE_URL", "https://openapivts.koreainvestment.com:29443")
    args_schema: Type[BaseModel] = PortfolioAnalysisInput

    return_direct: bool = True

    def _ensure_rate_limiter(self):
        # NOTE:
        # KIS는 초당 거래건수(TPS) 제한이 있으며, 이를 초과하면
        # rt_cd=1, msg1="초당 거래건수를 초과하였습니다." 형태로 응답이 내려옵니다.
        # 특히 다수 종목을 병렬로 분석할 때 burst가 나기 쉬워, "균등 간격" 제한을 둡니다.
        if not hasattr(self, "_kis_rate_lock"):
            self._kis_rate_lock = asyncio.Lock()
        if not hasattr(self, "_kis_next_allowed_at"):
            self._kis_next_allowed_at = 0.0
        if not hasattr(self, "_kis_min_interval"):
            # 전용 환경변수로 조절 (기본 1 rps)
            raw = (os.getenv("KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND") or "").strip()
            try:
                max_rps = float(raw) if raw else 1.0
            except Exception:
                max_rps = 1.0
            max_rps = max(0.1, max_rps)
            self._kis_min_interval = 1.0 / max_rps

    async def _throttle(self):
        self._ensure_rate_limiter()
        async with self._kis_rate_lock:
            now = time.monotonic()
            wait_s = float(self._kis_next_allowed_at) - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            # 다음 요청 가능 시각을 갱신(균등 간격)
            self._kis_next_allowed_at = time.monotonic() + float(self._kis_min_interval)
            return

    @staticmethod
    def _is_kis_rate_limit_message(msg: str) -> bool:
        m = (msg or "").strip()
        return "초당 거래건수를 초과" in m or "초당거래건수를 초과" in m

    def _make_headers(self, tr_id: str, user_info: dict) -> dict:
        """공통 헤더 생성 함수"""
        logger.debug("Creating headers for transaction ID: %s", tr_id)

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {user_info['kis_access_token']}",
            "appkey": user_info['kis_app_key'],
            "appsecret": user_info['kis_app_secret'],
            "tr_id": tr_id,
            "tr_cont": "N"
        }
        logger.debug("Headers created: %s", headers)
        return headers

    async def _kis_get_json(
        self,
        url: str,
        headers: dict,
        params: dict,
        user_info: dict,
        *,
        endpoint: str,
        symbol: str | None = None,
        max_retries: int = 3,
    ) -> tuple[dict | None, bool, dict]:
        """KIS GET 호출 공통 처리(토큰 만료/레이트리밋/JSON 파싱).

        Returns:
            (data|None, update_access_token_flag, user_info)
        """
        update_access_token_flag = False
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(max(1, int(max_retries))):
                await self._throttle()
                async with session.get(url, headers=headers, params=params) as response:
                    status_code = response.status
                    text = await response.text()

                # 토큰 만료/무효: 1회 재발급 후 재시도
                if status_code in (401, 403, 500) and (
                    "기간이 만료된 token" in text or "유효하지 않은 token" in text
                ):
                    user_info["kis_access_token"] = await get_access_token(
                        user_info["kis_app_key"], user_info["kis_app_secret"]
                    )
                    update_access_token_flag = True
                    headers["authorization"] = f"Bearer {user_info['kis_access_token']}"
                    continue

                try:
                    data = json.loads(text)
                except Exception:
                    logger.warning(
                        "KIS %s JSON parse failed: symbol=%s status=%s text=%s",
                        endpoint,
                        symbol,
                        status_code,
                        text[:200],
                    )
                    return None, update_access_token_flag, user_info

                if data.get("rt_cd") != "0":
                    msg1 = str(data.get("msg1", ""))
                    # 레이트리밋: 잠시 대기 후 재시도
                    if self._is_kis_rate_limit_message(msg1) and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue

                    logger.warning(
                        "KIS %s API error: symbol=%s rt_cd=%s msg1=%s",
                        endpoint,
                        symbol,
                        data.get("rt_cd"),
                        msg1,
                    )
                    return None, update_access_token_flag, user_info

                return data, update_access_token_flag, user_info

        return None, update_access_token_flag, user_info

    def _get_kis_trading_id(self) -> str:
        """KIS 계좌 조회/주문 등 trading API용 tr_id를 반환합니다.

        - 모의투자(openapivts): VTTC8434R
        - 실전(openapi): TTTC8434R
        """
        base = (self.url_base or "").lower()
        if "openapivts" in base:
            return "VTTC8434R"
        return "TTTC8434R"

    async def _fetch_user_holdings(self, base_url: str, tr_id: str, user_info: dict) -> dict:
        """(내부) 특정 base_url/TR로 기보유 종목을 조회합니다."""
        account_no = user_info.get("account_no") or ""
        if "-" not in account_no:
            raise ValueError("account_no 형식이 올바르지 않습니다. (예: 12345678-01)")

        url = base_url.rstrip("/") + "/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {user_info['kis_access_token']}",
            "appkey": user_info["kis_app_key"],
            "appsecret": user_info["kis_app_secret"],
            "tr_id": tr_id,
            "custtype": "P",
        }
        params = {
            "CANO": account_no.split("-")[0],
            "ACNT_PRDT_CD": account_no.split("-")[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._throttle()
            async with session.get(url, headers=headers, params=params) as response:
                text = await response.text()
                try:
                    res_data = json.loads(text)
                except Exception:
                    raise ValueError(f"KIS holdings 응답 파싱 실패: {text[:200]}")

                if res_data.get("rt_cd") != "0":
                    raise ValueError(
                        f"KIS holdings API 오류: {res_data.get('msg1', '알 수 없는 오류')}"
                    )

                holdings = []
                for item in res_data.get("output1", []) or []:
                    try:
                        quantity = int(item.get("hldg_qty", "0"))
                    except Exception:
                        quantity = 0
                    if quantity <= 0:
                        continue

                    code = str(item.get("pdno", "") or "")
                    name = str(item.get("prdt_name", "") or "")
                    try:
                        avg_buy_price = float(item.get("pchs_avg_pric", "0") or 0)
                    except Exception:
                        avg_buy_price = 0.0
                    try:
                        current_price = float(item.get("prpr", "0") or 0)
                    except Exception:
                        current_price = 0.0
                    try:
                        evaluated_amount = float(item.get("evlu_amt", "0") or 0)
                    except Exception:
                        evaluated_amount = 0.0
                    try:
                        profit_loss = float(item.get("evlu_pfls_amt", "0") or 0)
                    except Exception:
                        profit_loss = 0.0

                    return_rate = (
                        (current_price - avg_buy_price) / avg_buy_price
                        if avg_buy_price > 0
                        else 0.0
                    )
                    holdings.append(
                        {
                            "code": code,
                            "name": name,
                            "quantity": quantity,
                            "avg_buy_price": avg_buy_price,
                            "current_price": current_price,
                            "return_rate": return_rate,
                            "evaluated_amount": evaluated_amount,
                            "profit_loss": profit_loss,
                        }
                    )

                summary = None
                try:
                    out2 = (res_data.get("output2") or [])
                    if out2:
                        summary = dict(out2[0])
                except Exception:
                    summary = None

                return {"holdings": holdings, "summary": summary}

    async def get_user_holdings(self, user_info: dict) -> dict:
        """유저의 기보유 종목(계좌 보유 현황)을 조회합니다.

        Returns:
            {
              "holdings": [ {code,name,quantity,avg_buy_price,current_price,return_rate,evaluated_amount,profit_loss}, ... ],
              "summary": { ... } | None
            }
        """
        # 1) 현재 설정(base_url/tr_id)로 먼저 시도
        base_url = self.url_base
        tr_id = self._get_kis_trading_id()
        try:
            return await self._fetch_user_holdings(base_url, tr_id, user_info)
        except Exception as e:
            msg = str(e)

            # 2) 환경(TR) 불일치 시 자동 폴백
            # - "모의투자 TR 이 아닙니다." : 실전 TR을 모의 환경에 보낸 경우 → 모의 환경으로 재시도
            # - "실전투자 TR 이 아닙니다." : 모의 TR을 실전 환경에 보낸 경우 → 실전 환경으로 재시도
            if "모의투자 TR" in msg:
                try:
                    return await self._fetch_user_holdings(
                        "https://openapivts.koreainvestment.com:29443",
                        "VTTC8434R",
                        user_info,
                    )
                except Exception:
                    raise e
            if "실전투자 TR" in msg:
                try:
                    return await self._fetch_user_holdings(
                        "https://openapi.koreainvestment.com:9443",
                        "TTTC8434R",
                        user_info,
                    )
                except Exception:
                    raise e

            raise e

    def _ensure_dart_client(self):
        """DART 클라이언트/락을 지연 초기화합니다.

        OpenDartReader는 초기화 시 corp_codes를 조회하는데, 이를 종목마다 반복하면
        DART API 한도 초과(status=020)로 이어질 수 있습니다. (요청당 1회만 초기화)
        """
        if not hasattr(self, "_dart_lock"):
            self._dart_lock = asyncio.Lock()
        if not hasattr(self, "_dart_client"):
            self._dart_client = None
        if not hasattr(self, "_dart_api_key"):
            self._dart_api_key = None
        if not hasattr(self, "_dart_api_keys"):
            self._dart_api_keys = []
        if not hasattr(self, "_dart_key_index"):
            self._dart_key_index = 0
        if not hasattr(self, "_dart_exhausted_keys"):
            self._dart_exhausted_keys = set()

    async def _get_dart_client(self):
        """가능하면 OpenDartReader 인스턴스를 재사용하여 반환합니다."""
        api_keys = parse_open_dart_api_keys()
        if not api_keys:
            return None

        self._ensure_dart_client()

        # env 변경(키 추가/삭제)을 반영
        if getattr(self, "_dart_api_keys", None) != api_keys:
            self._dart_api_keys = api_keys
            self._dart_key_index = 0
            self._dart_exhausted_keys = set()
            self._dart_client = None
            self._dart_api_key = None

        async with self._dart_lock:
            # double-check
            if getattr(self, "_dart_client", None) is not None and getattr(self, "_dart_api_key", None) in api_keys:
                return self._dart_client

            keys: list[str] = list(self._dart_api_keys or [])
            if not keys:
                return None

            # 현재 인덱스부터 순환하며 초기화 시도 (status=020이면 다음 키로 폴백)
            tried = 0
            start_idx = int(self._dart_key_index or 0) % len(keys)
            idx = start_idx

            while tried < len(keys):
                api_key = keys[idx]
                idx = (idx + 1) % len(keys)
                tried += 1

                if api_key in self._dart_exhausted_keys:
                    continue

                try:
                    self._dart_client = OpenDartReader(api_key)
                    self._dart_api_key = api_key
                    self._dart_key_index = idx
                    return self._dart_client
                except Exception as e:
                    if is_dart_quota_exceeded_error(e):
                        # 한도 초과 키는 건너뛰고 다음 키로 계속
                        self._dart_exhausted_keys.add(api_key)
                        logger.warning(
                            "OpenDartReader quota exceeded(status=020). rotate key: %s",
                            mask_api_key(api_key),
                        )
                        continue

                    # 기타 오류(키 오류 등): 해당 키는 제외하고 다음 키 시도
                    self._dart_exhausted_keys.add(api_key)
                    logger.warning(
                        "Failed to initialize OpenDartReader (key=%s): %s",
                        mask_api_key(api_key),
                        e,
                    )
                    continue

            # 모든 키 실패
            self._dart_client = None
            self._dart_api_key = None
            return None
        return self._dart_client


    async def get_top_market_value(self, fid_rank_sort_cls_code, user_info):
        """시가총액 상위 종목을 조회합니다.
        - fid_rank_sort_cls_code: 순위 정렬 구분 코드 (23:PER, 24:PBR, 25:PCR, 26:PSR, 27: EPS, 28:EVA, 29: EBITDA, 30: EV/EBITDA, 31:EBITDA/금융비율)
        """
        logger.info("Fetching top market value stocks with sort code: %s", fid_rank_sort_cls_code)
        path = "/uapi/domestic-stock/v1/ranking/market-value"
        url = self.url_base + path
        headers = self._make_headers("FHPST01790000", user_info)

        params = {
            "fid_trgt_cls_code": "0",
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20179",
            "fid_input_iscd": "0000",
            "fid_div_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
            "fid_input_option_1": "2024",
            "fid_input_option_2": "3",
            "fid_rank_sort_cls_code": fid_rank_sort_cls_code,
            "fid_blng_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._throttle()
            async with session.get(url, headers=headers, params=params) as response:
                status_code = response.status
                text = await response.text()

                update_access_token_flag = False
                if status_code in (401, 403, 500) and ("기간이 만료된 token" in text or "유효하지 않은 token" in text):
                    user_info['kis_access_token'] = await get_access_token(user_info['kis_app_key'], user_info['kis_app_secret'])
                    update_access_token_flag = True

                    headers["authorization"] = (
                        f"Bearer {user_info['kis_access_token']}"
                    )
                    await self._throttle()
                    async with session.get(
                        url, headers=headers, params=params
                    ) as res_refresh:
                        status_code = res_refresh.status
                        text = await res_refresh.text()

                data = json.loads(text)

                logger.debug("Top market value stocks fetched: %s", data)
                return data.get("output", []), update_access_token_flag, user_info

    async def get_stock_basic_info(self, symbol):
        """OpenDART API를 사용하여 기업 정보를 조회하고, 산업분류명을 추가합니다."""
        dart = await self._get_dart_client()
        if dart is None:
            # DART 키가 없거나 초기화 실패 시 최소 정보만 반환(추천은 계속 진행)
            return {
                "corp_name": symbol,
                "corp_cls": None,
                "market": "N/A",
                "induty_code": None,
                "induty_name": "N/A",
            }

        # DART 사용량 초과/키 오류 등으로 실패할 수 있으므로, 추천이 전체적으로 죽지 않게 처리
        result = None
        try:
            # OpenDartReader는 동기 호출이므로 thread로 실행
            result = await asyncio.to_thread(dart.company, symbol)
        except Exception as e:
            # 한도초과(status=020)면 다른 키로 1회 재시도
            if is_dart_quota_exceeded_error(e):
                logger.warning(
                    "DART company quota exceeded(status=020) for %s; retry with rotated key",
                    symbol,
                )
                async with self._dart_lock:
                    # 현재 클라이언트를 무효화 → 다음 _get_dart_client()에서 다음 키를 선택
                    if self._dart_api_key:
                        self._dart_exhausted_keys.add(self._dart_api_key)
                    self._dart_client = None
                    self._dart_api_key = None

                dart2 = await self._get_dart_client()
                if dart2 is not None:
                    try:
                        result = await asyncio.to_thread(dart2.company, symbol)
                    except Exception as e2:
                        logger.warning(
                            "DART company lookup failed for %s after retry: %s",
                            symbol,
                            e2,
                        )
                        return {
                            "corp_name": symbol,
                            "corp_cls": None,
                            "market": "N/A",
                            "induty_code": None,
                            "induty_name": "N/A",
                        }
                else:
                    return {
                        "corp_name": symbol,
                        "corp_cls": None,
                        "market": "N/A",
                        "induty_code": None,
                        "induty_name": "N/A",
                    }
            else:
                logger.warning("DART company lookup failed for %s: %s", symbol, e)
                return {
                    "corp_name": symbol,
                    "corp_cls": None,
                    "market": "N/A",
                    "induty_code": None,
                    "induty_name": "N/A",
                }

        result["market"] = MARKET_MAP.get(result.get("corp_cls"), result.get("corp_cls"))
        
        # result에서 induty_code 추출
        induty_code = result.get("induty_code")
        
        # KSIC DB를 사용하여 industy 테이블에서 induty_name 조회 (옵션)
        if induty_code:
            engine_ksic = get_async_engine_ksic()
            if engine_ksic is None:
                result["induty_name"] = "N/A"
            else:
                try:
                    async with AsyncSession(engine_ksic) as session:
                        stmt = select(Industy).where(Industy.industy_code == induty_code)
                        db_result = await session.execute(stmt)
                        industy = db_result.scalar_one_or_none()

                        if industy:
                            result["induty_name"] = industy.industy_name
                        else:
                            result["induty_name"] = "N/A"
                except Exception as e:
                    # KSIC DB가 없거나(예: database "ksic" does not exist), 네트워크/권한 문제가 있더라도
                    # 추천 전체가 죽지 않도록 폴백합니다.
                    logger.warning(
                        "KSIC industry lookup failed (induty_code=%s). Disable KSIC lookups for this process. err=%s",
                        induty_code,
                        e,
                    )
                    # 이후 호출에서 반복 실패를 피하기 위해 비활성화
                    global _async_engine_ksic, _async_engine_ksic_disabled
                    _async_engine_ksic_disabled = True
                    _async_engine_ksic = None
                    result["induty_name"] = "N/A"
        else:
            result["induty_name"] = "N/A"
        
        return result
        

    async def get_stability_ratio(self, symbol: str, div_cd: str = "0", user_info=None):
        """국내주식 안정성 비율 조회"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/finance/stability-ratio"
        headers = self._make_headers("FHKST66430600", user_info)
        params = {
            "fid_input_iscd": symbol,
            "FID_DIV_CLS_CODE": div_cd,
            "fid_cond_mrkt_div_code": 'J'
        }

        data, update_access_token_flag, user_info = await self._kis_get_json(
            url,
            headers,
            params,
            user_info,
            endpoint="stability-ratio",
            symbol=symbol,
        )
        api_output = ((data or {}).get("output") or [])[:4]
        n = len(api_output)

        if n == 0:
            return 0.0, [], update_access_token_flag, user_info

        df = pd.DataFrame(api_output)
        cols = ["lblt_rate", "bram_depn", "crnt_rate", "quck_rate"]
        
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            min_value = df[c].min()
            max_value = df[c].max() if df[c].max() > min_value else min_value + 1
            df[c] = (df[c] - min_value) / (max_value - min_value)

        weights = [0.5, 0.3, 0.15, 0.05]
        if n < 4:
            weights = weights[:n] + [0] * (n - len(weights))  # 부족한 부분은 0으로 채움
        df["StabilityScore"] = df[cols].mean(axis=1)
        df["weight"] = weights
        df["weighted_score"] = df["StabilityScore"] * df["weight"]
        final_score = df["weighted_score"].sum()

        return final_score, api_output, update_access_token_flag, user_info


    async def get_profit_ratio(self, symbol: str, div_cd: str = "1", user_info=None):
        """수익성 비율 조회"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/finance/profit-ratio"
        headers = self._make_headers("FHKST66430400", user_info)
        params = {
            "fid_input_iscd": symbol,
            "FID_DIV_CLS_CODE": div_cd,
            "fid_cond_mrkt_div_code": 'J'
        }

        data, update_access_token_flag, user_info = await self._kis_get_json(
            url,
            headers,
            params,
            user_info,
            endpoint="profit-ratio",
            symbol=symbol,
        )
        api_output = ((data or {}).get("output") or [])[:4]
        n = len(api_output)

        if n == 0:
            return 0.0, [], update_access_token_flag, user_info

        df = pd.DataFrame(api_output)
        cols = ["cptl_ntin_rate","self_cptl_ntin_inrt","sale_ntin_rate","sale_totl_rate"]
        
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            min_value = df[c].min()
            max_value = df[c].max() if df[c].max() > min_value else min_value + 1
            df[c] = (df[c] - min_value) / (max_value - min_value)

        weights = [0.5, 0.3, 0.15, 0.05]
        if n < 4:
            weights = weights[:n] + [0] * (n - len(weights))  # 부족한 부분은 0으로 채움
        df["StabilityScore"] = df[cols].mean(axis=1)
        df["weight"] = weights
        df["weighted_score"] = df["StabilityScore"] * df["weight"]
        final_score = df["weighted_score"].sum()

        return final_score, api_output, update_access_token_flag, user_info

    async def get_growth_ratio(self, symbol: str, div_cd: str = "1", user_info=None):
        """성장성 비율 조회"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/finance/growth-ratio"
        headers = self._make_headers("FHKST66430800", user_info)
        params = {
            "fid_input_iscd": symbol,
            "FID_DIV_CLS_CODE": div_cd,
            "fid_cond_mrkt_div_code": 'J'
        }

        data, update_access_token_flag, user_info = await self._kis_get_json(
            url,
            headers,
            params,
            user_info,
            endpoint="growth-ratio",
            symbol=symbol,
        )
        api_output = ((data or {}).get("output") or [])[:4]
        n = len(api_output)

        if n == 0:
            return 0.0, [], update_access_token_flag, user_info

        df = pd.DataFrame(api_output)
        cols = ["grs","bsop_prfi_inrt","equt_inrt","totl_aset_inrt"] # 매출액 증가율, 영업 이익 증가율, 자기자본 증가율, 총자산 증가율
        
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            min_value = df[c].min()
            max_value = df[c].max() if df[c].max() > min_value else min_value + 1
            df[c] = (df[c] - min_value) / (max_value - min_value)

        weights = [0.5, 0.3, 0.15, 0.05]
        if n < 4:
            weights = weights[:n] + [0] * (n - len(weights))  # 부족한 부분은 0으로 채움

        df["StabilityScore"] = df[cols].mean(axis=1)
        df["weight"] = weights
        df["weighted_score"] = df["StabilityScore"] * df["weight"]
        final_score = df["weighted_score"].sum()

        return final_score, api_output, update_access_token_flag, user_info

    async def get_major_ratio(self, symbol: str, div_cd: str = "1", user_info=None):
        """기타 주요 비율 조회"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/finance/other-major-ratios"
        headers = self._make_headers("FHKST66430500", user_info)
        params = {
            "fid_input_iscd": symbol,
            "FID_DIV_CLS_CODE": div_cd,
            "fid_cond_mrkt_div_code": 'J'
        }

        data, update_access_token_flag, user_info = await self._kis_get_json(
            url,
            headers,
            params,
            user_info,
            endpoint="other-major-ratios",
            symbol=symbol,
        )
        api_output = ((data or {}).get("output") or [])[:4]
        n = len(api_output)

        if n == 0:
            return 0.0, [], update_access_token_flag, user_info

        df = pd.DataFrame(api_output)
        cols = ["payout_rate","eva","ebitda","ev_ebitda"] # 배당 성향, EVA, EBITDA, EV_EBITDA
        
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            min_value = df[c].min()
            max_value = df[c].max() if df[c].max() > min_value else min_value + 1
            df[c] = (df[c] - min_value) / (max_value - min_value)

        weights = [0.5, 0.3, 0.15, 0.05]
        if n < 4:
            weights = weights[:n] + [0] * (n - len(weights))  # 부족한 부분은 0으로 채움
        df["StabilityScore"] = df[cols].mean(axis=1)
        df["weight"] = weights
        df["weighted_score"] = df["StabilityScore"] * df["weight"]
        final_score = df["weighted_score"].sum()

        return final_score, api_output, update_access_token_flag, user_info

    async def get_financial_ratio(self, symbol: str, div_cd: str = "1", user_info=None):
        """재무 비율 조회"""
        url = f"{self.url_base}/uapi/domestic-stock/v1/finance/financial-ratio"
        headers = self._make_headers("FHKST66430300", user_info)
        params = {
            "fid_input_iscd": symbol,
            "FID_DIV_CLS_CODE": div_cd,
            "fid_cond_mrkt_div_code": 'J'
        }

        data, update_access_token_flag, user_info = await self._kis_get_json(
            url,
            headers,
            params,
            user_info,
            endpoint="financial-ratio",
            symbol=symbol,
        )
        api_output = ((data or {}).get("output") or [])[:4]
        n = len(api_output)

        if n == 0:
            return 0.0, [], update_access_token_flag, user_info

        df = pd.DataFrame(api_output)
        cols = ["grs","bsop_prfi_inrt","ntin_inrt","roe_val", "eps", "sps", "bps", "rsrv_rate", "lblt_rate"] # 매출액 증가율, 영업이익증가율, 순이익증가율, ROE, EPS, 주당매출액, BPS, 유보비율, 부채비율
        
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            min_value = df[c].min()
            max_value = df[c].max() if df[c].max() > min_value else min_value + 1
            df[c] = (df[c] - min_value) / (max_value - min_value)

        weights = [0.5, 0.3, 0.15, 0.05]
        if n < 4:
            weights = weights[:n] + [0] * (n - len(weights))  # 부족한 부분은 0으로 채움

        df["StabilityScore"] = df[cols].mean(axis=1)
        df["weight"] = weights
        df["weighted_score"] = df["StabilityScore"] * df["weight"]
        final_score = df["weighted_score"].sum()

        return final_score, api_output, update_access_token_flag, user_info

    async def analyze_stock(self, symbol: str, user_info: dict, risk_level: str):
        should_update_access_token = False

        stock_info = await self.get_stock_basic_info(symbol)

        stability_score, stability_data, update_access_token_flag, user_info = await self.get_stability_ratio(symbol, user_info=user_info)
        should_update_access_token |= update_access_token_flag

        profit_score, profit_data, update_access_token_flag, user_info = await self.get_profit_ratio(symbol, user_info=user_info)
        should_update_access_token |= update_access_token_flag
        
        growth_score, growth_data, update_access_token_flag, user_info = await self.get_growth_ratio(symbol, user_info=user_info)
        should_update_access_token |= update_access_token_flag
        
        major_score, major_data, update_access_token_flag, user_info = await self.get_major_ratio(symbol, user_info=user_info)
        should_update_access_token |= update_access_token_flag
        
        fin_score, fin_data, update_access_token_flag, user_info = await self.get_financial_ratio(symbol, user_info=user_info)
        should_update_access_token |= update_access_token_flag

        total_score = self._calculate_total_score(
            stability_score, profit_score, growth_score,
            major_score, fin_score, risk_level
        )

        analysis_result = {
            "symbol": symbol,
            "name": stock_info.get("corp_name"),
            "market": stock_info.get("market"),
            "sector": stock_info.get("induty_name"),
            "total_score": float(total_score),
            "stability_score": float(stability_score),
            "profit_score": float(profit_score),
            "growth_score": float(growth_score),
            "details": {
                "stability": stability_data,
                "profit": profit_data,
                "growth": growth_data,
                "major": major_data,
                "financial": fin_data
            }
        }

        return analysis_result, should_update_access_token

    async def analyze_portfolio(
        self,
        risk_level: str,
        user_info: dict,
        top_n: int = 30,
        portfolio_size: Optional[int] = None,
    ) -> Dict:
        """
        투자 성향에 따른 포트폴리오 분석 및 추천

        risk_level: "안정형" | "안정추구형" | "위험중립형" | "적극투자형" | "공격투자형"
        """
        logger.info("Analyzing portfolio for risk level: %s with top N: %d", risk_level, top_n)
        # 0. 기보유 종목 조회(가능하면)
        holdings_payload: dict | None = None
        try:
            holdings_payload = await self.get_user_holdings(user_info)
        except Exception as e:
            # holdings 조회 실패해도 추천 전체는 계속 진행
            logger.warning("Failed to fetch user holdings; proceed without holdings. err=%s", e)

        holdings_list = (holdings_payload or {}).get("holdings") or []
        holdings_map = {h["code"]: h for h in holdings_list if h.get("code")}

        # 1. 시가총액 상위 종목 조회
        ranking, update_access_token_flag, user_info = await self.get_top_market_value(fid_rank_sort_cls_code='23', user_info=user_info)
        portfolio_data = []
        should_update_access_token = update_access_token_flag

        tasks = []
        # 2. 각 종목별 지표 분석
        # - 후보군: 시가총액 상위(top_n) + 기보유 종목(중복 제거)
        candidate_symbols: list[str] = []
        seen: set[str] = set()

        for item in ranking[:top_n]:
            symbol = item.get("mksc_shrn_iscd")
            if not symbol:
                logger.warning("No symbol found for item: %s", item)
                continue
            symbol = str(symbol)
            if symbol in seen:
                continue
            seen.add(symbol)
            candidate_symbols.append(symbol)

        for code in holdings_map.keys():
            if code in seen:
                continue
            seen.add(code)
            candidate_symbols.append(code)

        for symbol in candidate_symbols:
            tasks.append(self.analyze_stock(symbol, user_info, risk_level))

        results = await asyncio.gather(*tasks)
        for analysis_result, flag in results:
            should_update_access_token |= flag
            sym = str(analysis_result.get("symbol", "") or "")
            is_holding = sym in holdings_map
            analysis_result["is_holding"] = is_holding
            if is_holding:
                analysis_result["holding"] = holdings_map.get(sym)
            portfolio_data.append(analysis_result)

        logger.info("Portfolio analysis completed. Total stocks analyzed: %d", len(portfolio_data))
        # 3. 투자 성향에 따른 포트폴리오 구성
        if should_update_access_token:
            # 토큰이 갱신된 경우 DB(user.kis_access_token)에 저장
            try:
                from multi_agent.utils import update_user_kis_credentials

                await update_user_kis_credentials(
                    get_async_engine(), user_info["id"], user_info["kis_access_token"]
                )
            except Exception as e:
                logger.warning(
                    "Failed to persist refreshed KIS access token to DB: %s",
                    e,
                )
        rec = self._build_portfolio_recommendation_with_holdings(
            portfolio_data,
            risk_level,
            holdings_map=holdings_map,
            portfolio_size=portfolio_size,
        )

        # 보고서용 메타/기보유 종목 포함
        rec["holdings"] = holdings_list
        rec["holdings_summary"] = (holdings_payload or {}).get("summary")

        # 기보유 종목이 최종 추천에 포함됐는지 표시
        rec_codes = {x.get("symbol") for x in rec.get("recommendations", [])}
        rec["holdings_included"] = [h for h in holdings_list if h.get("code") in rec_codes]
        rec["holdings_excluded"] = [h for h in holdings_list if h.get("code") not in rec_codes]
        rec["analysis_universe_size"] = len(candidate_symbols)
        rec["analysis_top_n_market_cap"] = top_n

        return rec

    def _build_portfolio_recommendation_with_holdings(
        self,
        data: List[Dict],
        risk_level: str,
        holdings_map: dict[str, dict] | None = None,
        portfolio_size: Optional[int] = None,
    ) -> Dict:
        """기보유 종목을 '기준'으로 최종 포트폴리오를 구성합니다.

        정책:
        - 기보유 종목이 있고, (보유 종목 수 <= 목표 개수)면: **보유 종목 전부 유지 + 부족분만 신규 추천**
        - 보유 종목 수가 목표 개수보다 많으면: 보유 종목 중 종합점수 상위 N개로 축소
        - 보유 종목이 없으면: 기존 로직(상위 N개)과 동일
        """
        holdings_map = holdings_map or {}

        # 기본 로직으로 target_size 계산(성향별 default 포함)
        base = self._build_portfolio_recommendation(
            data, risk_level, portfolio_size=portfolio_size
        )
        target_size = int(base.get("portfolio_size", 0) or 0)
        if target_size <= 0:
            return base

        sorted_data = sorted(data, key=lambda x: x["total_score"], reverse=True)
        if not sorted_data:
            return {"risk_level": risk_level, "portfolio_size": 0, "recommendations": []}

        if not holdings_map:
            return base

        holding_items = [x for x in sorted_data if str(x.get("symbol", "")) in holdings_map]
        new_items = [x for x in sorted_data if str(x.get("symbol", "")) not in holdings_map]

        if len(holding_items) >= target_size:
            final_items = holding_items[:target_size]
        else:
            need = target_size - len(holding_items)
            final_items = holding_items + new_items[:need]

        total_score = sum(item["total_score"] for item in final_items)
        if total_score > 0:
            for item in final_items:
                item["weight"] = round(item["total_score"] / total_score * 100, 2)
        else:
            equal = round(100 / len(final_items), 2)
            for item in final_items:
                item["weight"] = equal

        return {
            "risk_level": risk_level,
            "portfolio_size": len(final_items),
            "recommendations": final_items,
        }

    def _calculate_total_score(self, stability: float, profit: float, 
                             growth: float, major: float, fin: float, 
                             risk_level: str) -> float:
        """투자 성향에 따른 종합 점수 계산"""
        if risk_level == "위험중립형":
            weights = {
                "stability": 0.3,
                "profit": 0.2,
                "growth": 0.2,
                "major": 0.2,
                "financial": 0.1
            }
        elif risk_level == "안정추구형":
            weights = {
                "stability": 0.4,
                "profit": 0.2,
                "growth": 0.1,
                "major": 0.2,
                "financial": 0.1
            }
        elif risk_level == "안정형":
            weights = {
                "stability": 0.3,
                "profit": 0.3,
                "growth": 0.2,
                "major": 0.1,
                "financial": 0.1
            }
        elif risk_level == "적극투자형":
            weights = {
                "stability": 0.2,
                "profit": 0.3,
                "growth": 0.3,
                "major": 0.1,
                "financial": 0.1
            }
        else:  # 공격투자형
            weights = {
                "stability": 0.1,
                "profit": 0.3,
                "growth": 0.4,
                "major": 0.1,
                "financial": 0.1
            }

        return (
            stability * weights["stability"] +
            profit * weights["profit"] +
            growth * weights["growth"] +
            major * weights["major"] +
            fin * weights["financial"]
        )

    def _build_portfolio_recommendation(
        self,
        data: List[Dict],
        risk_level: str,
        portfolio_size: Optional[int] = None,
    ) -> Dict:
        """투자 성향에 따른 포트폴리오 추천"""
        # 점수 기준 정렬
        sorted_data = sorted(data, key=lambda x: x["total_score"], reverse=True)

        # 기본: 투자 성향별 포트폴리오 크기
        if portfolio_size is None:
            if risk_level == "위험중립형":
                portfolio_size = 4
            elif risk_level == "안정추구형":
                portfolio_size = 3
            elif risk_level == "안정형":
                portfolio_size = 3
            elif risk_level == "적극투자형":
                portfolio_size = 5
            else:  # 공격투자형
                portfolio_size = 6

        # 안전장치: 분석된 종목 수를 초과하면 가능한 만큼만
        if not sorted_data:
            return {"risk_level": risk_level, "portfolio_size": 0, "recommendations": []}

        portfolio_size = max(1, int(portfolio_size))
        portfolio_size = min(portfolio_size, len(sorted_data))

        # 상위 종목 선정
        recommended_portfolio = sorted_data[:portfolio_size]

        # 투자 비중 계산
        total_score = sum(item["total_score"] for item in recommended_portfolio)
        if total_score > 0:
            for item in recommended_portfolio:
                item["weight"] = round(item["total_score"] / total_score * 100, 2)
        else:
            # 점수가 모두 0이면 균등 분배
            equal = round(100 / len(recommended_portfolio), 2)
            for item in recommended_portfolio:
                item["weight"] = equal

        return {
            "risk_level": risk_level,
            "portfolio_size": portfolio_size,
            "recommendations": recommended_portfolio
        }

    def _format_analysis_result_to_markdown(self, analysis_result: Dict) -> str:
        """분석 결과를 '보고서' 형태의 한국어 마크다운으로 변환"""
        risk_level = analysis_result.get("risk_level", "N/A")
        portfolio_size = analysis_result.get("portfolio_size", 0)
        recommendations = analysis_result.get("recommendations", [])
        holdings = analysis_result.get("holdings", []) or []
        holdings_included = analysis_result.get("holdings_included", []) or []
        holdings_excluded = analysis_result.get("holdings_excluded", []) or []
        universe_size = analysis_result.get("analysis_universe_size", None)
        top_n_market_cap = analysis_result.get("analysis_top_n_market_cap", None)
        generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # 마크다운 시작(보고서)
        markdown = "# 포트폴리오 추천 보고서\n\n"
        markdown += f"- 생성 시각: {generated_at}\n"
        markdown += f"- 투자 성향(설문 기반): **{risk_level}**\n"
        markdown += f"- 최종 추천 종목 수: **{portfolio_size}개**\n"
        if universe_size is not None and top_n_market_cap is not None:
            markdown += f"- 분석 후보군: 시가총액 상위 {top_n_market_cap} + 기보유 종목 → **{universe_size}개**\n"
        markdown += "\n---\n\n"
        
        # 1) 기보유 종목 섹션
        markdown += "## 1) 기보유 종목 현황\n\n"
        if not holdings:
            markdown += "- 현재 기보유 종목이 없거나, 계좌 조회 결과가 없습니다.\n\n"
        else:
            markdown += f"- 기보유 종목: **{len(holdings)}개**\n"
            markdown += f"- 최종 추천 포트폴리오에 포함: **{len(holdings_included)}개**, 제외: **{len(holdings_excluded)}개**\n\n"

            markdown += "| 종목명 | 종목코드 | 보유수량 | 평균단가 | 현재가 | 수익률 | 평가금액 | 평가손익 | 최종포트폴리오 |\n"
            markdown += "|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|\n"
            included_codes = {h.get("code") for h in holdings_included}
            for h in holdings:
                name = h.get("name", "N/A")
                code = h.get("code", "N/A")
                qty = h.get("quantity", 0)
                avgp = h.get("avg_buy_price", 0.0)
                curp = h.get("current_price", 0.0)
                rr = h.get("return_rate", 0.0)
                eva = h.get("evaluated_amount", 0.0)
                pnl = h.get("profit_loss", 0.0)
                flag = "포함" if code in included_codes else "미포함"
                markdown += (
                    f"| {name} | {code} | {qty:,} | {avgp:,.0f} | {curp:,.0f} | {rr*100:.2f}% | {eva:,.0f} | {pnl:,.0f} | {flag} |\n"
                )
            markdown += "\n---\n\n"

        # 2) 추천 프로세스 설명
        markdown += "## 2) 추천 프로세스(요약)\n\n"
        markdown += "- **투자 성향 산출**: `public.survey.answer(q1~q8)`를 점수화하여 투자 성향(안정형~공격투자형)을 결정합니다.\n"
        markdown += "- **기보유 종목 반영**: 계좌 보유 종목을 조회한 뒤, **보유 종목을 최종 포트폴리오의 베이스로 포함**하고 부족분만 신규로 추천합니다.\n"
        markdown += "  - 단, 보유 종목 수가 요청 개수보다 많으면 보유 종목 중 종합점수 상위 N개로 축소합니다.\n"
        markdown += "- **후보군 구성**: 시가총액 상위 종목 + 기보유 종목을 합쳐 분석 후보군을 만듭니다.\n"
        markdown += "- **지표 계산/스코어링**: 안정성/수익성/성장성/주요지표/재무비율 지표를 계산하고, 투자 성향별 가중치로 종합점수를 산출합니다.\n"
        markdown += "- **최종 선정/비중 산출**: 종합점수 상위 **N개(요청 개수)**를 선정하고, 종합점수 비율로 투자 비중(%)을 계산합니다.\n\n"
        markdown += "---\n\n"
        
        # 3) 최종 추천 포트폴리오(마지막)
        markdown += "## 3) 최종 추천 포트폴리오\n\n"
        markdown += "| 순위 | 구분 | 종목명 | 종목코드 | 업종 | 시장 | 투자비중 | 종합점수 | 안정성 | 수익성 | 성장성 |\n"
        markdown += "|:---:|:---:|:---|:---:|:---|:---|---:|---:|---:|---:|---:|\n"
        
        for idx, stock in enumerate(recommendations, 1):
            name = stock.get("name", "N/A")
            symbol = stock.get("symbol", "N/A")
            sector = stock.get("sector", "N/A")
            market = stock.get("market", "N/A")
            weight = stock.get("weight", 0)
            total_score = stock.get("total_score", 0)
            stability_score = stock.get("stability_score", 0)
            profit_score = stock.get("profit_score", 0)
            growth_score = stock.get("growth_score", 0)
            kind = "기보유" if stock.get("is_holding") else "신규"
            
            markdown += (
                f"| {idx} | {kind} | {name} | {symbol} | {sector} | {market} | {weight}% | "
                f"{float(total_score):.3f} | {float(stability_score):.3f} | {float(profit_score):.3f} | {float(growth_score):.3f} |\n"
            )
        
        return markdown
    
    def _run(
        self,
        user_investor_type: str,
        portfolio_size: Optional[int] = None,
        config: RunnableConfig = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return asyncio.run(self._arun(user_investor_type, portfolio_size, config, run_manager))


    async def _arun(
        self, 
        user_investor_type: str,
        portfolio_size: Optional[int] = None,
        config: RunnableConfig = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        비동기 포트폴리오 분석 실행 메서드
        risk_profile: 투자자의 위험 성향 (선택적)
        top_n: 분석할 종목 수
        """
        # user_id 기반으로 stockelper_web.user 테이블에서 KIS 자격증명/계좌를 조회합니다.
        user_id = (config or {}).get("configurable", {}).get("user_id")
        if user_id is None:
            raise ValueError("user_id가 없습니다. 요청에 user_id를 포함해주세요.")

        user_info = await get_user_kis_credentials(
            async_engine=get_async_engine(), user_id=user_id
        )
        if not user_info:
            raise ValueError(f"user_id={user_id} 사용자를 DB에서 찾지 못했습니다.")

        # DB에 저장된 토큰이 있으면 재사용하고,
        # 없으면 app_key/app_secret으로 발급받아 user.kis_access_token에 저장합니다.
        access_token = user_info.get("kis_access_token")
        if not access_token:
            access_token = await get_access_token(
                user_info["kis_app_key"], user_info["kis_app_secret"]
            )
            if not access_token:
                raise ValueError(
                    "KIS access token 발급에 실패했습니다. KIS 키를 확인해주세요."
                )
            user_info["kis_access_token"] = access_token
            try:
                from multi_agent.utils import update_user_kis_credentials

                await update_user_kis_credentials(
                    get_async_engine(), user_id, access_token
                )
            except Exception as e:
                logger.warning("Failed to persist issued KIS access token to DB: %s", e)
        
        # 포트폴리오 분석 실행
        analysis_result = await self.analyze_portfolio(
            user_investor_type,
            user_info,
            top_n=20,
            portfolio_size=portfolio_size,
        )

        # 마크다운 형식으로 변환하여 반환
        markdown_result = self._format_analysis_result_to_markdown(analysis_result)
        
        return markdown_result
