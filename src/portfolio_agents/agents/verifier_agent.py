from __future__ import annotations

import os

from agents import Agent

from ..schemas import VerificationResult


_MODEL = (os.getenv("OPENAI_AGENTS_VERIFY_MODEL") or "gpt-4.1-mini").strip()

VERIFIER_PROMPT = """당신은 꼼꼼한 감사자(auditor)입니다.
입력으로 주어진 포트폴리오 추천 리포트를 읽고, 다음을 점검하세요:
- 내부 모순(비중 합계, 표의 값/설명 불일치)
- '기보유 종목 현황' 누락 또는 실패 사유가 있는 경우의 안내 부족
- 근거 없는 단정/과도한 확신

문제가 없으면 verified=true로, 문제가 있으면 issues에 핵심만 적고 retry_suggestion을 제안하세요."""

verifier_agent = Agent(
    name="PortfolioReportVerifierAgent",
    instructions=VERIFIER_PROMPT,
    model=_MODEL,
    output_type=VerificationResult,
)

