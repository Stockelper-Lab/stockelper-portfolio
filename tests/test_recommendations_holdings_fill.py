from __future__ import annotations

from datetime import datetime, timezone

import pytest

from portfolio_agents.manager import PortfolioAgentsManager
from portfolio_agents.context import PortfolioAgentContext
from portfolio_multi_agent.state import PortfolioMetrics, PortfolioResult, PortfolioWeight, Stock


def _extract_final_portfolio_codes(markdown: str) -> list[str]:
    """결정적 리포트의 '## 3) 최종 추천 포트폴리오' 표에서 종목코드를 추출합니다."""
    lines = [ln.rstrip("\n") for ln in (markdown or "").splitlines()]
    try:
        start = lines.index("## 3) 최종 추천 포트폴리오")
    except ValueError:
        return []

    # 표 헤더 2줄("| 순위 | ...", "|---:|...") 이후부터 데이터 행
    rows = []
    for ln in lines[start + 1 :]:
        if not ln.startswith("|"):
            # 표가 끝나면 종료
            if rows:
                break
            continue
        # 헤더/구분선은 스킵
        if "종목코드" in ln or ln.startswith("|---"):
            continue
        rows.append(ln)

    codes: list[str] = []
    for r in rows:
        parts = [p.strip() for p in r.strip("|").split("|")]
        if len(parts) < 4:
            continue
        code = parts[3]
        if code:
            codes.append(code)
    return codes


@pytest.mark.asyncio
async def test_recommendations_fill_to_target_including_holdings():
    manager = PortfolioAgentsManager()

    ctx = PortfolioAgentContext(
        user_id=1,
        engine=object(),  # 실제로 사용되지 않음
        request_id="test",
        now_utc=datetime.now(timezone.utc),
        kis_base_url="https://example.invalid",
        include_web_search=False,
        risk_free_rate=0.03,
    )

    # 보유 5개 + 후보 30개
    holdings = [
        {"code": f"H{i:02d}", "name": f"HOLD{i:02d}", "quantity": 1, "evaluated_amount": 1000 + i}
        for i in range(1, 6)
    ]
    universe = [Stock(code=h["code"], name=h["name"]) for h in holdings] + [
        Stock(code=f"N{i:02d}", name=f"NEW{i:02d}") for i in range(1, 31)
    ]

    # 최적화 결과는 신규 종목 위주로 가중치가 높게 나오더라도,
    # 최종 추천은 "보유 + 신규" 합이 target(20)이 되도록 해야 함.
    weights = []
    for idx, s in enumerate(universe):
        # 신규에 더 높은 weight를 줌
        w = 0.10 if s.code.startswith("N") else 0.001
        weights.append(PortfolioWeight(code=s.code, name=s.name, weight=w, reasoning=""))

    portfolio_result = PortfolioResult(
        weights=weights,
        metrics=PortfolioMetrics(expected_return=0.1, volatility=0.2, sharpe_ratio=0.3),
    )

    md = await manager._write_report(
        ctx=ctx,
        investor_type="위험중립형",
        universe=universe,
        portfolio_size=20,
        stock_scores={s.code: 1.0 for s in universe},
        portfolio_result=portfolio_result,
        holdings=holdings,
        holdings_summary=None,
        holdings_error=None,
        user_info={},  # 업종/시장 enrich 스킵
        run_cfg=object(),  # LLM 비사용 경로에서 사용되지 않음
    )

    codes = _extract_final_portfolio_codes(md)
    assert len(codes) == 20
    # 보유 종목은 최종 20개에 포함되어야 함
    for h in holdings:
        assert h["code"] in codes


@pytest.mark.asyncio
async def test_recommendations_fill_after_holdings_reduced():
    manager = PortfolioAgentsManager()

    ctx = PortfolioAgentContext(
        user_id=1,
        engine=object(),
        request_id="test",
        now_utc=datetime.now(timezone.utc),
        kis_base_url="https://example.invalid",
        include_web_search=False,
        risk_free_rate=0.03,
    )

    # (매도 후) 보유 3개만 남은 상황을 가정 → 나머지 17개를 신규로 채워야 함
    holdings = [
        {"code": f"H{i:02d}", "name": f"HOLD{i:02d}", "quantity": 1, "evaluated_amount": 1000 + i}
        for i in range(1, 4)
    ]
    universe = [Stock(code=h["code"], name=h["name"]) for h in holdings] + [
        Stock(code=f"N{i:02d}", name=f"NEW{i:02d}") for i in range(1, 40)
    ]

    weights = [PortfolioWeight(code=s.code, name=s.name, weight=0.01, reasoning="") for s in universe]
    portfolio_result = PortfolioResult(
        weights=weights,
        metrics=PortfolioMetrics(expected_return=0.1, volatility=0.2, sharpe_ratio=0.3),
    )

    md = await manager._write_report(
        ctx=ctx,
        investor_type="위험중립형",
        universe=universe,
        portfolio_size=20,
        stock_scores={s.code: 1.0 for s in universe},
        portfolio_result=portfolio_result,
        holdings=holdings,
        holdings_summary=None,
        holdings_error=None,
        user_info={},
        run_cfg=object(),
    )

    codes = _extract_final_portfolio_codes(md)
    assert len(codes) == 20
    for h in holdings:
        assert h["code"] in codes

