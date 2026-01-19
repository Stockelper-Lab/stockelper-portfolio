from __future__ import annotations

import pytest

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
