from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from observability.langfuse import build_langfuse_runnable_config
from multi_agent.portfolio_analysis_agent.tools.portfolio import PortfolioAnalysisTool
from multi_agent.utils import (
    create_portfolio_recommendation_job,
    get_user_survey_answer,
    survey_answer_to_investor_type,
    update_portfolio_recommendation_job,
)
from portfolio_agents import PortfolioAgentsManager
from portfolio_multi_agent.builder import build_buy_workflow, build_sell_workflow
from portfolio_multi_agent.state import BuyInputState, SellInputState

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_ENGINE = None
_BUY_WORKFLOW = None
_SELL_WORKFLOW = None
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


def _get_agents_manager() -> PortfolioAgentsManager:
    global _AGENTS_MANAGER
    if _AGENTS_MANAGER is None:
        _AGENTS_MANAGER = PortfolioAgentsManager()
    return _AGENTS_MANAGER


def _use_openai_agents_for_recommendations() -> bool:
    # 기본은 기존 로직(PortfolioAnalysisTool) 유지.
    # 운영에서 단계적 전환을 위해 환경변수로만 스위치합니다.
    raw = (os.getenv("PORTFOLIO_RECOMMENDATIONS_ENGINE") or "").strip().lower()
    if raw in {"agents", "openai-agents", "openai_agents"}:
        return True
    raw2 = (os.getenv("PORTFOLIO_USE_OPENAI_AGENTS") or "").strip().lower()
    return raw2 in {"1", "true", "yes", "y", "on"}


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
        if _use_openai_agents_for_recommendations():
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
        else:
            tool = PortfolioAnalysisTool()
            lf_cfg = build_langfuse_runnable_config(
                run_name="portfolio_recommendations",
                trace_id=job_id,
                user_id=body.user_id,
                session_id=str(body.user_id),
                tags=[
                    "stockelper-portfolio",
                    "api",
                    "portfolio",
                    "recommendations",
                ],
                metadata={
                    "endpoint": "/portfolio/recommendations",
                    "rec_id": rec_id,
                    "job_id": job_id,
                    "investor_type": investor_type,
                },
            )
            cfg = {"configurable": {"user_id": body.user_id}}
            cfg.update(lf_cfg)
            result = await tool.ainvoke(
                {
                    "user_investor_type": investor_type,
                    "portfolio_size": body.portfolio_size,
                },
                config=cfg,
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
        raise

    return {"id": rec_id, "job_id": job_id, "investor_type": investor_type, "result": result}


@router.post("/buy", status_code=status.HTTP_200_OK)
async def buy_portfolio(body: BuyInputState):
    """포트폴리오 매수 워크플로우 실행(LangGraph)."""
    workflow = _get_buy_workflow()
    try:
        trace_id = str(uuid.uuid4())
        lf_cfg = build_langfuse_runnable_config(
            run_name="portfolio_buy_agent",
            trace_id=trace_id,
            user_id=body.user_id,
            session_id=str(body.user_id),
            tags=["stockelper-portfolio", "api", "portfolio", "buy"],
            metadata={
                "endpoint": "/portfolio/buy",
                "trace_id": trace_id,
            },
        )
        result = await workflow.ainvoke(body.model_dump(), config=lf_cfg)
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
        trace_id = str(uuid.uuid4())
        lf_cfg = build_langfuse_runnable_config(
            run_name="portfolio_sell_agent",
            trace_id=trace_id,
            user_id=body.user_id,
            session_id=str(body.user_id),
            tags=["stockelper-portfolio", "api", "portfolio", "sell"],
            metadata={
                "endpoint": "/portfolio/sell",
                "trace_id": trace_id,
            },
        )
        result = await workflow.ainvoke(body.model_dump(), config=lf_cfg)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    return jsonable_encoder(result)


