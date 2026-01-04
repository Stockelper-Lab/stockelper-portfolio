from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from multi_agent.portfolio_analysis_agent.tools.portfolio import PortfolioAnalysisTool
from multi_agent.utils import (
    get_user_survey_answer,
    survey_answer_to_investor_type,
)
from portfolio_multi_agent.builder import build_buy_workflow, build_sell_workflow
from portfolio_multi_agent.state import BuyInputState, SellInputState

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_ENGINE = None
_BUY_WORKFLOW = None
_SELL_WORKFLOW = None


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
    _ENGINE = create_async_engine(async_db_url, echo=False)
    return _ENGINE


def _get_buy_workflow():
    global _BUY_WORKFLOW
    if _BUY_WORKFLOW is None:
        _BUY_WORKFLOW = build_buy_workflow()
    return _BUY_WORKFLOW


def _get_sell_workflow():
    global _SELL_WORKFLOW
    if _SELL_WORKFLOW is None:
        _SELL_WORKFLOW = build_sell_workflow()
    return _SELL_WORKFLOW


class PortfolioRecommendationRequest(BaseModel):
    user_id: int = Field(description="User ID")


@router.post("/recommendations", status_code=status.HTTP_200_OK)
async def recommend_portfolio(body: PortfolioRecommendationRequest):
    engine = _get_engine()

    survey_answer = await get_user_survey_answer(engine, body.user_id)
    if not survey_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="설문 정보(survey.answer)가 없습니다. 먼저 설문을 완료해주세요.",
        )
    investor_type = survey_answer_to_investor_type(survey_answer)

    tool = PortfolioAnalysisTool()
    result = await tool.ainvoke(
        {"user_investor_type": investor_type},
        config={"configurable": {"user_id": body.user_id}},
    )

    return {"investor_type": investor_type, "result": result}


@router.post("/buy", status_code=status.HTTP_200_OK)
async def buy_portfolio(body: BuyInputState):
    """포트폴리오 매수 워크플로우 실행(LangGraph)."""
    workflow = _get_buy_workflow()
    try:
        result = await workflow.ainvoke(body.model_dump())
    except Exception as e:
        # 워크플로우 내부(외부 API/토큰/입력값) 오류를 500이 아닌 502로 노출
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    return jsonable_encoder(result)


@router.post("/sell", status_code=status.HTTP_200_OK)
async def sell_portfolio(body: SellInputState):
    """포트폴리오 매도 워크플로우 실행(LangGraph)."""
    workflow = _get_sell_workflow()
    try:
        result = await workflow.ainvoke(body.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    return jsonable_encoder(result)


