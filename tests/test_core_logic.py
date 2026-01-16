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


def test_portfolio_analysis_markdown_includes_holdings_section_and_flags():
    tool = PortfolioAnalysisTool()

    analysis_result = {
        "risk_level": "안정형",
        "portfolio_size": 2,
        "analysis_universe_size": 22,
        "analysis_top_n_market_cap": 20,
        "holdings": [
            {
                "code": "000001",
                "name": "A",
                "quantity": 10,
                "avg_buy_price": 1000.0,
                "current_price": 1100.0,
                "return_rate": 0.1,
                "evaluated_amount": 11000.0,
                "profit_loss": 1000.0,
            },
            {
                "code": "000009",
                "name": "X",
                "quantity": 5,
                "avg_buy_price": 2000.0,
                "current_price": 1800.0,
                "return_rate": -0.1,
                "evaluated_amount": 9000.0,
                "profit_loss": -1000.0,
            },
        ],
        "holdings_included": [{"code": "000001"}],
        "holdings_excluded": [{"code": "000009"}],
        "recommendations": [
            {
                "symbol": "000001",
                "name": "A",
                "sector": "N/A",
                "market": "유가",
                "weight": 60.0,
                "total_score": 0.9,
                "stability_score": 0.2,
                "profit_score": 0.2,
                "growth_score": 0.2,
                "is_holding": True,
            },
            {
                "symbol": "000002",
                "name": "B",
                "sector": "N/A",
                "market": "유가",
                "weight": 40.0,
                "total_score": 0.6,
                "stability_score": 0.2,
                "profit_score": 0.2,
                "growth_score": 0.2,
                "is_holding": False,
            },
        ],
    }

    markdown = tool._format_analysis_result_to_markdown(analysis_result)

    # 섹션/요약 라인 존재
    assert "## 1) 기보유 종목 현황" in markdown
    assert "- 기보유 종목: **2개**" in markdown
    assert "최종 추천 포트폴리오에 포함: **1개**, 제외: **1개**" in markdown

    # 테이블 헤더/행/포함 플래그 검증
    assert "| 종목명 | 종목코드 | 보유수량 |" in markdown
    assert "| A | 000001 |" in markdown
    assert "| X | 000009 |" in markdown
    assert "| A | 000001 |" in markdown and "| 포함 |" in markdown
    assert "| X | 000009 |" in markdown and "| 미포함 |" in markdown


@pytest.mark.asyncio
async def test_portfolio_analysis_end_to_end_holdings_to_markdown_with_mocks(monkeypatch):
    """외부 API/DB 없이 holdings → analyze_portfolio → markdown 흐름을 검증."""
    tool = PortfolioAnalysisTool()

    async def fake_get_user_holdings(self, user_info):  # noqa: ANN001
        return {
            "holdings": [
                {
                    "code": "000001",
                    "name": "A",
                    "quantity": 10,
                    "avg_buy_price": 1000.0,
                    "current_price": 1100.0,
                    "return_rate": 0.1,
                    "evaluated_amount": 11000.0,
                    "profit_loss": 1000.0,
                },
                {
                    "code": "000009",
                    "name": "X",
                    "quantity": 5,
                    "avg_buy_price": 2000.0,
                    "current_price": 1800.0,
                    "return_rate": -0.1,
                    "evaluated_amount": 9000.0,
                    "profit_loss": -1000.0,
                },
            ],
            "summary": {"dnca_tot_amt": "1000000"},
        }

    async def fake_get_top_market_value(self, fid_rank_sort_cls_code, user_info):  # noqa: ANN001
        ranking = [
            {"mksc_shrn_iscd": "000002"},
            {"mksc_shrn_iscd": "000003"},
        ]
        return ranking, False, user_info

    async def fake_analyze_stock(self, symbol, user_info, risk_level):  # noqa: ANN001
        scores = {"000001": 0.9, "000009": 0.1, "000002": 0.8, "000003": 0.7}
        names = {"000001": "A", "000009": "X", "000002": "B", "000003": "C"}
        return (
            {
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "market": "유가",
                "sector": "N/A",
                "total_score": scores.get(symbol, 0.0),
                "stability_score": 0.2,
                "profit_score": 0.2,
                "growth_score": 0.2,
                "details": {},
            },
            False,
        )

    # NOTE: PortfolioAnalysisTool은 Pydantic 모델(BaseTool)이므로 인스턴스에 setattr로 주입이 막힙니다.
    # 따라서 클래스 메서드를 패치하여 흐름을 모킹합니다.
    monkeypatch.setattr(
        PortfolioAnalysisTool, "get_user_holdings", fake_get_user_holdings, raising=True
    )
    monkeypatch.setattr(
        PortfolioAnalysisTool, "get_top_market_value", fake_get_top_market_value, raising=True
    )
    monkeypatch.setattr(PortfolioAnalysisTool, "analyze_stock", fake_analyze_stock, raising=True)

    # 목표 1개로 설정 → holding 2개 중 상위 1개만 최종 추천에 포함되고 1개는 제외되도록 유도
    rec = await tool.analyze_portfolio(
        risk_level="안정형",
        user_info={"id": 1, "account_no": "12345678-01"},
        top_n=2,
        portfolio_size=1,
    )
    markdown = tool._format_analysis_result_to_markdown(rec)

    assert "## 1) 기보유 종목 현황" in markdown
    assert "최종 추천 포트폴리오에 포함: **1개**, 제외: **1개**" in markdown
    assert "| A | 000001 |" in markdown and "| 포함 |" in markdown
    assert "| X | 000009 |" in markdown and "| 미포함 |" in markdown


