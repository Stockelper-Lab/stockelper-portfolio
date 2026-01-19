from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class InvestorViewResponse(BaseModel):
    """단일 종목에 대한 투자자 뷰(LLM 출력)"""

    expected_return: float = Field(
        ...,
        ge=-0.20,
        le=0.20,
        description="향후 1년간 예상 초과 수익률 (소수, 예: 0.10=10%). 범위 -0.20~0.20",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="예측 신뢰도(0~1). 데이터/신호 일치도가 높을수록 높게",
    )
    reasoning: str = Field(..., description="투자 판단 근거(2~3문장, 간결)")


class PortfolioReportData(BaseModel):
    """최종 리포트(LLM 출력)"""

    short_summary: str = Field(description="2~3문장 요약")
    markdown_report: str = Field(description="전체 마크다운 보고서")


class VerificationResult(BaseModel):
    """리포트/결과 정합성 점검 결과(선택)"""

    verified: bool = Field(description="리포트가 내부적으로 일관적인지 여부")
    issues: str = Field(default="", description="문제점/우려 사항(있으면)")
    retry_suggestion: Optional[str] = Field(
        default=None, description="재시도/수정이 필요하면 제안"
    )


class SellDecisionResponse(BaseModel):
    """단일 종목에 대한 매도 결정(LLM 출력)."""

    decision: Literal["SELL", "HOLD"] = Field(description="매도 결정(SELL/HOLD)")
    reasoning: str = Field(description="결정 사유(2~3문장, 간결)")

