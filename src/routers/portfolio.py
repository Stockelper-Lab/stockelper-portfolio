from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from multi_agent.utils import (
    create_portfolio_recommendation_job,
    get_user_survey_answer,
    survey_answer_to_investor_type,
    update_portfolio_recommendation_job,
)
from portfolio_agents import PortfolioAgentsManager
from portfolio_multi_agent.state import BuyInputState, SellInputState

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_ENGINE = None
_AGENTS_MANAGER = None


def _to_async_db_url(url: Optional[str]) -> Optional[str]:
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


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    async_db_url = os.getenv("ASYNC_DATABASE_URL") or _to_async_db_url(
        os.getenv("DATABASE_URL")
    )
    if not async_db_url:
        raise RuntimeError("ASYNC_DATABASE_URL 또는 DATABASE_URL 이 설정되어 있지 않습니다.")
    # NOTE:
    # 운영 환경에서 DB/네트워크가 idle 커넥션을 끊는 경우가 있어,
    # 풀에서 꺼낸 커넥션이 이미 닫혀있으면 자동으로 재연결하도록 pre-ping을 활성화합니다.
    _ENGINE = create_async_engine(async_db_url, echo=False, pool_pre_ping=True)
    return _ENGINE


def _get_agents_manager() -> PortfolioAgentsManager:
    global _AGENTS_MANAGER
    if _AGENTS_MANAGER is None:
        _AGENTS_MANAGER = PortfolioAgentsManager()
    return _AGENTS_MANAGER


class PortfolioRecommendationRequest(BaseModel):
    user_id: int = Field(description="User ID")
    portfolio_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="추천 종목 개수(1~20). 채팅에서 '10개 종목 추천'처럼 요청하면 이 값이 전달됩니다.",
    )
    include_web_search: Optional[bool] = Field(
        default=None,
        description="(선택) 웹검색 기반 신호를 포함할지 여부. 미지정 시 서버 환경변수 기본값을 따릅니다.",
    )
    risk_free_rate: float = Field(
        default=0.03,
        description="(선택) 무위험 이자율(연율, 기본 3%). Agents 기반 최적화에서 사용됩니다.",
    )


@router.post("/recommendations", status_code=status.HTTP_200_OK)
async def recommend_portfolio(body: PortfolioRecommendationRequest):
    engine = _get_engine()

    # 1) 요청을 받는 순간 "빈" 레코드를 먼저 생성 (job_id UUID 생성)
    job = await create_portfolio_recommendation_job(body.user_id)
    rec_id = job["id"]
    job_id = job["job_id"]

    investor_type: str = ""
    result: str = ""

    try:
        # 2) 설문 기반 투자성향 산출
        survey_answer = await get_user_survey_answer(engine, body.user_id)
        if not survey_answer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="설문 정보(survey.answer)가 없습니다. 먼저 설문을 완료해주세요.",
            )
        investor_type = survey_answer_to_investor_type(survey_answer)

        # 3) 포트폴리오 추천 생성
        manager = _get_agents_manager()
        result = await manager.recommend_markdown(
            engine=engine,
            user_id=body.user_id,
            investor_type=investor_type,
            portfolio_size=body.portfolio_size,
            request_id=job_id,
            include_web_search=body.include_web_search,
            risk_free_rate=body.risk_free_rate,
        )

        # 4) 결과 생성 완료 후, 최초 레코드를 업데이트
        await update_portfolio_recommendation_job(rec_id, investor_type, result)
    except Exception as e:
        # 실패하더라도 DB에 "빈 레코드"만 남지 않도록 에러를 적재(최선 시도)
        try:
            detail = str(e)
            if isinstance(e, HTTPException):
                detail = str(e.detail)
            await update_portfolio_recommendation_job(
                rec_id, investor_type or "", f"ERROR: {detail}"
            )
        except Exception:
            pass
        # 클라이언트가 원인을 알 수 있도록 500 대신 의미있는 상태코드로 변환합니다.
        if isinstance(e, HTTPException):
            raise

        msg = str(e)
        if ("KIS 자격증명" in msg) or ("유효하지 않은 AppKey" in msg) or ("EGW00103" in msg):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from e
        if ("초당" in msg and "초과" in msg) or ("EGW00133" in msg) or ("EGW00201" in msg):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=msg) from e
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=msg) from e

    return {"id": rec_id, "job_id": job_id, "investor_type": investor_type, "result": result}


@router.post("/buy", status_code=status.HTTP_200_OK)
async def buy_portfolio(body: BuyInputState):
    """포트폴리오 매수 워크플로우 실행(OpenAI Agents SDK)."""
    engine = _get_engine()
    try:
        request_id = str(uuid.uuid4())
        manager = _get_agents_manager()
        result = await manager.buy(
            engine=engine,
            user_id=body.user_id,
            request_id=request_id,
            max_portfolio_size=body.max_portfolio_size,
            rank_weight=body.rank_weight,
            risk_free_rate=body.risk_free_rate,
        )
    except Exception as e:
        # 워크플로우 내부(외부 API/토큰/입력값) 오류를 500이 아닌 502로 노출
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    return jsonable_encoder(result)


@router.post("/sell", status_code=status.HTTP_200_OK)
async def sell_portfolio(body: SellInputState):
    """포트폴리오 매도 워크플로우 실행(OpenAI Agents SDK)."""
    engine = _get_engine()
    try:
        request_id = str(uuid.uuid4())
        manager = _get_agents_manager()
        result = await manager.sell(
            engine=engine,
            user_id=body.user_id,
            request_id=request_id,
            loss_threshold=body.loss_threshold,
            profit_threshold=body.profit_threshold,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    return jsonable_encoder(result)


