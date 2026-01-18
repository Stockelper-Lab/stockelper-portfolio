from __future__ import annotations

import os

from agents import Agent

from ..schemas import PortfolioReportData


_MODEL = (os.getenv("OPENAI_AGENTS_REPORT_MODEL") or "gpt-4.1-mini").strip()

WRITER_PROMPT = """당신은 포트폴리오 리포트 작성자입니다.
입력으로 제공되는 (1) 투자 성향, (2) 기보유 종목 현황, (3) 최종 추천 포트폴리오(비중/업종/시장/근거),
그리고 (4) 추천 프로세스(요약)를 사용하여, 사용자가 바로 읽을 수 있는 마크다운 보고서를 작성하세요.

형식 요구사항:
- 반드시 아래 섹션 헤더를 포함하세요(순서 유지):
  - '포트폴리오 추천 보고서' (문서 제목)
  - '## 1) 기보유 종목 현황'
  - '## 2) 추천 프로세스(요약)'
  - '## 3) 최종 추천 포트폴리오'
- 표는 마크다운 테이블로 작성하세요.
- '추천 프로세스(요약)'는 큰 기능만 불릿으로 나열하세요(상세 구현/세부 파라미터 설명 금지).
"""

writer_agent = Agent(
    name="PortfolioReportWriterAgent",
    instructions=WRITER_PROMPT,
    model=_MODEL,
    output_type=PortfolioReportData,
)

