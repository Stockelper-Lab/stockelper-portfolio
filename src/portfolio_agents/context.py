from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class PortfolioAgentContext:
    """Agents SDK run 에 전달되는 로컬 컨텍스트.

    - 주의: 이 객체 자체는 LLM에 전송되지 않습니다. (도구/훅에서만 접근)
    - LLM이 봐야 하는 데이터는 입력 메시지 또는 도구 출력으로 전달해야 합니다.
    """

    user_id: int
    engine: AsyncEngine
    request_id: str
    now_utc: datetime

    kis_base_url: str
    include_web_search: bool = False
    risk_free_rate: float = 0.03

