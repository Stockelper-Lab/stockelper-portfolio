from __future__ import annotations

import os

from agents import Agent

from ..schemas import InvestorViewResponse


_MODEL = (os.getenv("OPENAI_AGENTS_VIEW_MODEL") or "gpt-4.1-mini").strip()

PROMPT = """당신은 전문 투자 분석가입니다.
제공된 세 가지 분석 결과(웹 검색, 재무제표, 기술적 지표)를 종합하여 단일 종목의 투자자 뷰를 생성하세요.

요구사항:
- expected_return: -0.20 ~ 0.20 범위의 소수(예: 0.10=10%)
- confidence: 0.0 ~ 1.0 범위
- reasoning: 2~3문장, 간결. 데이터가 부족하면 중립(0.0, 0.3)로 두고 이유를 적으세요.

가이드라인(권장):
- 매우 강세: 0.10 ~ 0.15
- 강세: 0.05 ~ 0.10
- 약한 강세: 0.02 ~ 0.05
- 중립: -0.02 ~ 0.02
- 약한 약세: -0.05 ~ -0.02
- 약세: -0.10 ~ -0.05
- 매우 약세: -0.15 ~ -0.10

신뢰도(권장):
- 0.8~1.0: 세 분석이 모두 일치하고 명확
- 0.6~0.8: 대부분 일치
- 0.4~0.6: 혼재
- 0.2~0.4: 상충/불확실
- 0.0~0.2: 정보 부족
"""

view_agent = Agent(
    name="PortfolioViewAgent",
    instructions=PROMPT,
    model=_MODEL,
    output_type=InvestorViewResponse,
)

