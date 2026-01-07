from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
import routers.portfolio as portfolio_router


def test_portfolio_recommendations_mocked(monkeypatch):
    async def fake_ainvoke(self, input, config=None):  # noqa: ANN001
        return "MOCK_RECOMMENDATION_RESULT"

    # DB/외부 API 없이도 라우터의 request/response 형태를 검증하기 위해 모킹
    monkeypatch.setattr(
        portfolio_router.PortfolioAnalysisTool, "ainvoke", fake_ainvoke, raising=True
    )
    monkeypatch.setattr(portfolio_router, "_get_engine", lambda: object(), raising=True)
    async def fake_get_user_survey_answer(engine, user_id):  # noqa: ANN001
        return {
            "q1": 3,
            "q2": 5,
            "q3": 5,
            "q4": 1,
            "q5": [1, 2, 3],
            "q6": 3,
            "q7": 3,
            "q8": 3,
        }

    monkeypatch.setattr(
        portfolio_router,
        "get_user_survey_answer",
        fake_get_user_survey_answer,
        raising=True,
    )
    monkeypatch.setattr(
        portfolio_router,
        "survey_answer_to_investor_type",
        lambda answer: "안정형",
        raising=True,
    )
    async def fake_create_job(user_id):  # noqa: ANN001
        return {"id": "test-id", "job_id": "test-job-id"}

    async def fake_update_job(rec_id, investor_type, result):  # noqa: ANN001
        return True

    monkeypatch.setattr(
        portfolio_router,
        "create_portfolio_recommendation_job",
        fake_create_job,
        raising=True,
    )
    monkeypatch.setattr(
        portfolio_router,
        "update_portfolio_recommendation_job",
        fake_update_job,
        raising=True,
    )

    client = TestClient(app)
    res = client.post(
        "/portfolio/recommendations",
        json={"user_id": 1, "portfolio_size": 10},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["id"] == "test-id"
    assert payload["job_id"] == "test-job-id"
    assert payload["investor_type"] == "안정형"
    assert payload["result"] == "MOCK_RECOMMENDATION_RESULT"


def test_portfolio_buy_mocked(monkeypatch):
    class DummyWorkflow:
        async def ainvoke(self, payload, config=None):  # noqa: ANN001
            return {"ok": True, "input": payload}

    monkeypatch.setattr(portfolio_router, "_get_buy_workflow", lambda: DummyWorkflow())

    client = TestClient(app)
    res = client.post("/portfolio/buy", json={"user_id": 1})
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert "input" in payload


def test_portfolio_sell_mocked(monkeypatch):
    class DummyWorkflow:
        async def ainvoke(self, payload, config=None):  # noqa: ANN001
            return {"ok": True, "input": payload}

    monkeypatch.setattr(portfolio_router, "_get_sell_workflow", lambda: DummyWorkflow())

    client = TestClient(app)
    res = client.post("/portfolio/sell", json={"user_id": 1})
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] is True
    assert "input" in payload


