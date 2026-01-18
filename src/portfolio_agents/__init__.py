"""OpenAI Agents SDK 기반 포트폴리오 에이전트 패키지.

이 패키지는 `openai-agents`(import 경로: `agents`)를 사용해
포트폴리오 추천/리포트 생성을 코드 기반 오케스트레이션 + LLM(Structured Outputs) 조합으로 구현합니다.
"""

from .manager import PortfolioAgentsManager

__all__ = ["PortfolioAgentsManager"]

