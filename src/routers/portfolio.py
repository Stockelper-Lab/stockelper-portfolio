from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from multi_agent.portfolio_analysis_agent.tools.portfolio import PortfolioAnalysisTool
from multi_agent.utils import get_user_kis_credentials
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
    investor_type: Optional[str] = Field(
        default=None,
        description="투자 성향(없으면 DB의 사용자 investor_type 사용)",
    )


@router.post("/recommendations", status_code=status.HTTP_200_OK)
async def recommend_portfolio(body: PortfolioRecommendationRequest):
    engine = _get_engine()

    investor_type = body.investor_type
    if investor_type is None:
        user_info = await get_user_kis_credentials(engine, body.user_id)
        if not user_info or not user_info.get("investor_type"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="투자 성향 정보(investor_type)가 없습니다. 먼저 사용자 프로필을 설정해주세요.",
            )
        investor_type = user_info["investor_type"]

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
    result = await workflow.ainvoke(body.model_dump())
    return jsonable_encoder(result)


@router.post("/sell", status_code=status.HTTP_200_OK)
async def sell_portfolio(body: SellInputState):
    """포트폴리오 매도 워크플로우 실행(LangGraph)."""
    workflow = _get_sell_workflow()
    result = await workflow.ainvoke(body.model_dump())
    return jsonable_encoder(result)


