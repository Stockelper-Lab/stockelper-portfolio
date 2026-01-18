from __future__ import annotations

import os

from agents import Agent, WebSearchTool


_MODEL = (os.getenv("OPENAI_AGENTS_SEARCH_MODEL") or "gpt-4.1-mini").strip()

INSTRUCTIONS = (
    "당신은 금융 리서치 어시스턴트입니다. 주어진 검색어로 웹 검색을 수행하고, "
    "최근 뉴스/핵심 이벤트/숫자(실적, 가이던스 등)/리스크를 중심으로 "
    "최대 300단어(한국어)로 간결히 요약하세요. 출처(도메인/매체)를 가능하면 함께 언급하세요."
)

search_agent = Agent(
    name="PortfolioWebSearchAgent",
    model=_MODEL,
    instructions=INSTRUCTIONS,
    tools=[WebSearchTool()],
)

