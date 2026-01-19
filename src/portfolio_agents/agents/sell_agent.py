from __future__ import annotations

from agents import Agent

from ..schemas import SellDecisionResponse

SELL_DECISION_PROMPT = """당신은 전문 투자 분석가입니다.
주어진 보유 종목 정보와 분석 결과(웹 검색/재무제표/기술적 지표)를 종합해서,
이 종목을 지금 매도할지(HOLD/SELL) 결정하세요.

결정 기준(가이드라인):
- 손절: 수익률이 손절 기준 이하이면 SELL을 강하게 고려
- 익절: 수익률이 익절 기준 이상이면 SELL을 고려
- 펀더멘털/기술/뉴스 리스크가 명확하면 SELL, 긍정적 요인이 우세하면 HOLD

출력 요구사항:
- decision: "SELL" 또는 "HOLD"
- reasoning: 2~3문장으로 간결하게 (핵심 근거만)
"""

sell_agent = Agent(
    name="PortfolioSellDecisionAgent",
    instructions=SELL_DECISION_PROMPT,
    output_type=SellDecisionResponse,
)

