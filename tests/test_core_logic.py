from __future__ import annotations

import pytest

from multi_agent.portfolio_analysis_agent.tools.portfolio import PortfolioAnalysisTool
from portfolio_multi_agent.nodes.portfolio_builder import PortfolioBuilder, InputState
from portfolio_multi_agent.state import InvestorView, MarketData, Stock


@pytest.mark.asyncio
async def test_portfolio_builder_optimizes_weights_and_metrics():
    # 간단한 3종목 샘플 (수익률 시계열은 길이만 동일하면 됨)
    stocks = [
        Stock(code="000001", name="A"),
        Stock(code="000002", name="B"),
        Stock(code="000003", name="C"),
    ]
    market_data_list = [
        MarketData(code="000001", name="A", returns=[0.01, -0.005, 0.002, 0.003, 0.004]),
        MarketData(code="000002", name="B", returns=[0.008, -0.002, 0.001, 0.002, 0.003]),
        MarketData(code="000003", name="C", returns=[0.012, -0.007, 0.003, 0.001, 0.002]),
    ]
    investor_views = [
        InvestorView(
            stock_indices=[0],
            expected_return=0.05,
            confidence=0.7,
            reasoning="테스트 뷰",
        )
    ]

    builder = PortfolioBuilder(max_weight=0.8)
    out = await builder(
        InputState(
            portfolio_list=stocks,
            market_data_list=market_data_list,
            investor_views=investor_views,
            stock_scores={s.code: 1.0 for s in stocks},
            risk_free_rate=0.03,
        )
    )

    result = out["portfolio_result"]
    assert result is not None
    assert result.weights

    weight_sum = sum(w.weight for w in result.weights)
    # 아주 작은 비중 필터링(>0.001) 때문에 1.0에서 약간 벗어날 수 있어 느슨하게 체크
    assert 0.98 <= weight_sum <= 1.0
    for w in result.weights:
        assert 0.0 <= w.weight <= 0.8

    assert result.metrics.expected_return is not None
    assert result.metrics.volatility is not None
    assert result.metrics.sharpe_ratio is not None


def test_portfolio_analysis_recommendation_postprocess():
    tool = PortfolioAnalysisTool()

    # 종합 점수 기반 정렬/비중 산출 로직만 검증 (외부 API 호출 없음)
    sample = [
        {
            "symbol": "000001",
            "name": "A",
            "market": "유가",
            "sector": "N/A",
            "total_score": 0.9,
            "stability_score": 0.2,
            "profit_score": 0.2,
            "growth_score": 0.2,
            "details": {},
        },
        {
            "symbol": "000002",
            "name": "B",
            "market": "유가",
            "sector": "N/A",
            "total_score": 0.6,
            "stability_score": 0.2,
            "profit_score": 0.2,
            "growth_score": 0.2,
            "details": {},
        },
        {
            "symbol": "000003",
            "name": "C",
            "market": "유가",
            "sector": "N/A",
            "total_score": 0.3,
            "stability_score": 0.2,
            "profit_score": 0.2,
            "growth_score": 0.2,
            "details": {},
        },
        {
            "symbol": "000004",
            "name": "D",
            "market": "유가",
            "sector": "N/A",
            "total_score": 0.1,
            "stability_score": 0.2,
            "profit_score": 0.2,
            "growth_score": 0.2,
            "details": {},
        },
    ]

    rec = tool._build_portfolio_recommendation(sample, "안정형")
    assert rec["risk_level"] == "안정형"
    assert rec["portfolio_size"] == 3  # 안정형은 3개
    assert len(rec["recommendations"]) == 3

    weights = [x["weight"] for x in rec["recommendations"]]
    assert all(isinstance(w, float) for w in weights)
    # 반올림 때문에 100에서 약간 어긋날 수 있음
    assert abs(sum(weights) - 100.0) <= 0.2

    # 요청 개수 오버라이드: 2개로 강제
    rec2 = tool._build_portfolio_recommendation(sample, "안정형", portfolio_size=2)
    assert rec2["portfolio_size"] == 2
    assert len(rec2["recommendations"]) == 2



