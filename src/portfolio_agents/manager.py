from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Iterable

import aiohttp
from agents import Runner, RunConfig, custom_span, trace

from multi_agent.utils import ensure_user_kis_access_token, get_user_kis_credentials
from portfolio_multi_agent.nodes.get_financial_statement import (
    FinancialStatement,
    InputState as FinancialStatementInputState,
)
from portfolio_multi_agent.nodes.get_technical_indicator import (
    TechnicalIndicator,
    InputState as TechnicalIndicatorInputState,
)
from portfolio_multi_agent.nodes.portfolio_builder import (
    PortfolioBuilder,
    InputState as PortfolioBuilderInputState,
)
from portfolio_multi_agent.nodes.rank_func.get_market_cap_rank import get_market_cap_rank
from portfolio_multi_agent.state import (
    AnalysisResult,
    InvestorView,
    MarketData,
    PortfolioWeight,
    PortfolioResult,
    Stock,
)

from .agents import search_agent, verifier_agent, view_agent, writer_agent
from .context import PortfolioAgentContext
from .schemas import InvestorViewResponse, PortfolioReportData, VerificationResult


def _bool_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _kis_base_url() -> str:
    return (os.getenv("KIS_API_BASE_URL") or "https://openapivts.koreainvestment.com:29443").rstrip(
        "/"
    )


def _is_etf_or_etn(name: str) -> bool:
    raw = (name or "").strip()
    if not raw:
        return False
    up = raw.upper()
    if "ETF" in up or "ETN" in up:
        return True
    if any(
        k in up
        for k in [
            "KODEX",
            "TIGER",
            "KOSEF",
            "KBSTAR",
            "ARIRANG",
            "HANARO",
            "KINDEX",
            "TIMEFOLIO",
            "ACE",
            "SOL",
        ]
    ):
        return True
    if any(k in raw for k in ["레버리지", "인버스"]):
        return True
    return False


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return default


class PortfolioAgentsManager:
    """OpenAI Agents SDK 기반 추천/리포트 오케스트레이터.

    - 결정적(외부 API/계산) 단계는 코드로 수행
    - LLM은 (선택적으로) WebSearch/InvestorView/Report/Verify 단계에만 사용
    """

    def __init__(self) -> None:
        self._enable_web_search = _bool_env("PORTFOLIO_ENABLE_WEBSEARCH", default=False)
        self._enable_llm_views = _bool_env("PORTFOLIO_ENABLE_LLM_VIEWS", default=True)
        self._enable_llm_report = _bool_env("PORTFOLIO_ENABLE_LLM_REPORT", default=True)
        self._enable_llm_verify = _bool_env("PORTFOLIO_ENABLE_LLM_VERIFY", default=False)

        self._analysis_top_n = _safe_int(os.getenv("PORTFOLIO_ANALYSIS_TOP_N", "20"), 20)
        self._default_portfolio_size = _safe_int(
            os.getenv("PORTFOLIO_DEFAULT_SIZE", "10"), 10
        )
        self._websearch_max = _safe_int(os.getenv("PORTFOLIO_WEBSEARCH_MAX_STOCKS", "6"), 6)
        self._view_max = _safe_int(os.getenv("PORTFOLIO_VIEW_MAX_STOCKS", "10"), 10)

    @staticmethod
    def _llm_available() -> bool:
        return bool((os.getenv("OPENAI_API_KEY") or "").strip())

    @staticmethod
    def _to_trace_id(request_id: str) -> str:
        """OpenAI Agents trace_id 형식(`trace_<32_alphanumeric>`)으로 변환."""
        rid = str(request_id or "").strip()
        if rid.startswith("trace_") and len(rid) > 6:
            return rid
        candidate = rid.replace("-", "")
        if len(candidate) == 32 and all(c in "0123456789abcdefABCDEF" for c in candidate):
            return f"trace_{candidate}"
        # 형식이 불명확하면 SDK가 생성하도록 빈 값 반환(또는 자동 생성)
        return ""

    @staticmethod
    def _build_run_config(ctx: PortfolioAgentContext) -> RunConfig:
        # 민감 데이터(trace) 포함 방지: 기본적으로 false 권장
        trace_id = PortfolioAgentsManager._to_trace_id(ctx.request_id) or None
        return RunConfig(
            workflow_name="Stockelper Portfolio Agent",
            trace_id=trace_id,
            group_id=str(ctx.user_id),
            # OPENAI_AGENTS_DISABLE_TRACING=1 이거나 OPENAI_API_KEY가 없으면 트레이싱을 끕니다.
            tracing_disabled=_bool_env("OPENAI_AGENTS_DISABLE_TRACING", default=False)
            or not PortfolioAgentsManager._llm_available(),
            trace_include_sensitive_data=False,
            trace_metadata={
                "user_id": ctx.user_id,
                "request_id": ctx.request_id,
                "kis_base_url": ctx.kis_base_url,
            },
        )

    async def recommend_markdown(
        self,
        *,
        engine: Any,
        user_id: int,
        investor_type: str,
        portfolio_size: int | None,
        request_id: str,
        include_web_search: bool | None = None,
        risk_free_rate: float = 0.03,
    ) -> str:
        """추천 보고서(마크다운) 생성."""
        now_utc = datetime.now(timezone.utc)
        ctx = PortfolioAgentContext(
            user_id=user_id,
            engine=engine,
            request_id=request_id,
            now_utc=now_utc,
            kis_base_url=_kis_base_url(),
            include_web_search=(
                self._enable_web_search if include_web_search is None else bool(include_web_search)
            ),
            risk_free_rate=float(risk_free_rate),
        )

        # 포트폴리오 사이즈 디폴트
        n_portfolio = int(portfolio_size or 0) if portfolio_size else 0
        if n_portfolio <= 0:
            n_portfolio = self._default_portfolio_size

        run_cfg = self._build_run_config(ctx)

        def span(name: str):
            return custom_span(name) if not run_cfg.tracing_disabled else nullcontext()

        async def run_pipeline() -> str:
            # 1) DB에서 user KIS 자격증명 조회 + 토큰 보장
            with span("Load user context"):
                user_info = await get_user_kis_credentials(engine, user_id)
                if not user_info:
                    raise ValueError(f"user_id={user_id} 사용자를 DB에서 찾지 못했습니다.")
                if not user_info.get("account_no"):
                    raise ValueError("user.account_no가 비어있습니다.")

                token = await ensure_user_kis_access_token(engine, user_id, user_info, validate=True)
                user_info["kis_access_token"] = token

            # 2) 기보유 종목 조회(모의투자)
            with span("Fetch holdings (KIS)"):
                holdings, holdings_summary, holdings_error = await self._fetch_holdings(user_info, ctx)

            # 3) 후보군(시총 상위 + 기보유 종목)
            with span("Build universe"):
                universe, stock_scores = await self._build_universe(
                    user_info=user_info,
                    holdings=holdings,
                    target_portfolio_size=n_portfolio,
                )

            # 4) 신호 수집(재무/기술 + 선택: 웹검색)
            with span("Collect signals"):
                analysis_results, market_data_list = await self._collect_signals(
                    ctx=ctx, user_info=user_info, universe=universe, run_cfg=run_cfg
                )

            # 5) Investor views(LLM, 선택)
            with span("Generate investor views"):
                investor_views = await self._generate_investor_views(
                    ctx=ctx,
                    universe=universe,
                    analysis_results=analysis_results,
                    run_cfg=run_cfg,
                )

            # 6) Black-Litterman 최적화(결정적)
            with span("Optimize portfolio (Black-Litterman)"):
                portfolio_result = await self._build_portfolio(
                    universe=universe,
                    stock_scores=stock_scores,
                    market_data_list=market_data_list,
                    investor_views=investor_views,
                    risk_free_rate=float(ctx.risk_free_rate),
                )

            # 7) 리포트 작성(LLM, 선택; 실패 시 결정적 포맷으로 폴백)
            with span("Write report"):
                markdown = await self._write_report(
                    ctx=ctx,
                    investor_type=investor_type,
                    universe=universe,
                    portfolio_size=n_portfolio,
                    stock_scores=stock_scores,
                    portfolio_result=portfolio_result,
                    holdings=holdings,
                    holdings_summary=holdings_summary,
                    holdings_error=holdings_error,
                    user_info=user_info,
                    run_cfg=run_cfg,
                )

            # 8) 리포트 검증(선택)
            if self._enable_llm_verify and self._llm_available():
                with span("Verify report"):
                    try:
                        v = await Runner.run(verifier_agent, markdown, run_config=run_cfg)
                        ver = v.final_output_as(VerificationResult)
                        if not ver.verified and ver.retry_suggestion:
                            markdown = await self._write_report(
                                ctx=ctx,
                                investor_type=investor_type,
                                universe=universe,
                                portfolio_size=n_portfolio,
                                stock_scores=stock_scores,
                                portfolio_result=portfolio_result,
                                holdings=holdings,
                                holdings_summary=holdings_summary,
                                holdings_error=holdings_error,
                                user_info=user_info,
                                run_cfg=run_cfg,
                                extra_instruction=ver.retry_suggestion,
                            )
                    except Exception:
                        pass

            return markdown

        if run_cfg.tracing_disabled:
            return await run_pipeline()

        with trace(
            run_cfg.workflow_name,
            trace_id=getattr(run_cfg, "trace_id", None) or None,
            group_id=getattr(run_cfg, "group_id", None) or None,
        ):
            return await run_pipeline()

    async def _fetch_holdings(
        self, user_info: dict, ctx: PortfolioAgentContext
    ) -> tuple[list[dict], dict | None, str | None]:
        """KIS inquire-balance로 보유 종목을 조회합니다.

        IMPORTANT: 계좌/키 매칭 오류(INVALID_CHECK_ACNO)는 토큰 재발급으로 해결되지 않는 경우가 많아
        여기서는 재발급/재시도를 하지 않습니다(토큰은 Load 단계에서 이미 보장됨).
        """
        account_no = str(user_info.get("account_no", "") or "").strip()
        token = str(user_info.get("kis_access_token", "") or "").strip()
        app_key = str(user_info.get("kis_app_key", "") or "").strip()
        app_secret = str(user_info.get("kis_app_secret", "") or "").strip()

        if not account_no or "-" not in account_no:
            return [], None, "계좌번호 형식이 올바르지 않습니다."
        if not token or not app_key or not app_secret:
            return [], None, "KIS 자격증명이 없습니다."

        cano, prdt = account_no.split("-", 1)
        url = f"{ctx.kis_base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": "VTTC8434R",
            "custtype": "P",
        }
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as res:
                data = await res.json(content_type=None)

        if str(data.get("rt_cd", "")).strip() != "0":
            return [], None, str(data.get("msg1", "알 수 없는 오류"))

        output1 = data.get("output1") or []
        output2 = data.get("output2") or []
        summary = output2[0] if isinstance(output2, list) and output2 else None

        holdings: list[dict] = []
        for item in output1:
            qty = _safe_int(item.get("hldg_qty", 0), 0)
            if qty <= 0:
                continue
            avg = _safe_float(item.get("pchs_avg_pric", 0.0), 0.0)
            cur = _safe_float(item.get("prpr", 0.0), 0.0)
            evaluated_amount = _safe_float(item.get("evlu_amt", 0.0), float(cur * qty))
            profit_loss = _safe_float(item.get("evlu_pfls_amt", 0.0), float((cur - avg) * qty))
            return_rate = (cur - avg) / avg if avg > 0 else 0.0

            holdings.append(
                {
                    "code": str(item.get("pdno", "") or "").strip(),
                    "name": str(item.get("prdt_name", "") or "").strip(),
                    "quantity": qty,
                    "avg_buy_price": float(avg),
                    "current_price": float(cur),
                    "return_rate": float(return_rate),
                    "evaluated_amount": float(evaluated_amount),
                    "profit_loss": float(profit_loss),
                }
            )

        return holdings, summary, None

    async def _build_universe(
        self, *, user_info: dict, holdings: list[dict], target_portfolio_size: int
    ) -> tuple[list[Stock], dict[str, float]]:
        """시총 상위 종목 + 기보유 종목으로 분석 후보군을 구성합니다."""
        # 분석 후보군은 목표 포트폴리오 수의 2배(최소 10)로 잡되, 상한은 analysis_top_n으로 제한
        universe_target = min(max(int(target_portfolio_size) * 2, 10), int(self._analysis_top_n))
        fetch_n = min(int(self._analysis_top_n) * 2, 60)  # ETF/ETN 필터링을 고려해 여유 있게 호출

        market_cap = await get_market_cap_rank(
            top_n=fetch_n,
            app_key=str(user_info.get("kis_app_key", "") or ""),
            app_secret=str(user_info.get("kis_app_secret", "") or ""),
            access_token=str(user_info.get("kis_access_token", "") or ""),
        )
        filtered = [x for x in market_cap if not _is_etf_or_etn(str(x.get("name", "") or ""))]
        filtered = filtered[:universe_target]

        # 점수: 순위 기반 정규화(합=1)
        n = len(filtered)
        total_sum = n * (n + 1) / 2 if n > 0 else 1
        stock_scores: dict[str, float] = {}
        code_to_name: dict[str, str] = {}
        for rank, s in enumerate(filtered, start=1):
            code = str(s.get("code", "") or "").strip()
            name = str(s.get("name", "") or "").strip()
            if not code:
                continue
            code_to_name[code] = name
            stock_scores[code] = float((n - rank + 1) / total_sum)

        # 기보유 종목을 후보군에 포함(있으면)
        # - 순위 점수가 없는 종목은 매우 작은 값으로 부여(시장가중치 계산에서 0 방지)
        for h in holdings:
            code = str(h.get("code", "") or "").strip()
            name = str(h.get("name", "") or "").strip()
            if not code:
                continue
            if code not in code_to_name:
                code_to_name[code] = name or code
            if code not in stock_scores:
                stock_scores[code] = 0.001

        # 최종 후보군(중복 제거, 순서 유지: 시총 → 보유)
        universe_codes: list[str] = []
        for s in filtered:
            c = str(s.get("code", "") or "").strip()
            if c and c not in universe_codes:
                universe_codes.append(c)
        for h in holdings:
            c = str(h.get("code", "") or "").strip()
            if c and c not in universe_codes:
                universe_codes.append(c)

        # 안전 상한(과도한 LLM 호출 방지)
        hard_cap = _safe_int(os.getenv("PORTFOLIO_UNIVERSE_MAX", "30"), 30)
        universe_codes = universe_codes[: max(1, hard_cap)]

        universe = [Stock(code=c, name=code_to_name.get(c, c)) for c in universe_codes]
        return universe, stock_scores

    async def _collect_signals(
        self,
        *,
        ctx: PortfolioAgentContext,
        user_info: dict,
        universe: list[Stock],
        run_cfg: RunConfig,
    ) -> tuple[list[AnalysisResult], list[MarketData]]:
        """재무/기술(+선택 웹검색) 신호 수집."""
        # 재무/기술은 결정적(툴)로 실행
        financial = FinancialStatement()
        technical = TechnicalIndicator()

        fin_task = asyncio.create_task(
            financial(FinancialStatementInputState(portfolio_list=universe))
        )
        tech_task = asyncio.create_task(
            technical(
                TechnicalIndicatorInputState(
                    user_id=ctx.user_id,
                    kis_app_key=user_info.get("kis_app_key"),
                    kis_app_secret=user_info.get("kis_app_secret"),
                    kis_access_token=user_info.get("kis_access_token"),
                    account_no=user_info.get("account_no"),
                    portfolio_list=universe,
                )
            )
        )

        # 웹검색은 비용/환경에 따라 선택
        web_results: dict[str, str] = {}
        if ctx.include_web_search and self._llm_available():
            web_targets = universe[: max(0, int(self._websearch_max))]
            tasks = [
                asyncio.create_task(self._web_search_one(stock, run_cfg)) for stock in web_targets
            ]
            for t in asyncio.as_completed(tasks):
                code, summary = await t
                if code and summary:
                    web_results[code] = summary

        fin_out, tech_out = await asyncio.gather(fin_task, tech_task)

        analysis_results: list[AnalysisResult] = []
        analysis_results.extend(list(fin_out.get("analysis_results") or []))
        analysis_results.extend(list(tech_out.get("analysis_results") or []))

        # 웹검색 결과를 AnalysisResult로 합치기
        for stock in universe:
            summary = web_results.get(stock.code, "")
            if not summary:
                continue
            analysis_results.append(
                AnalysisResult(code=stock.code, name=stock.name, type="web_search", result=summary)
            )

        market_data_list: list[MarketData] = list(tech_out.get("market_data_list") or [])
        return analysis_results, market_data_list

    async def _web_search_one(self, stock: Stock, run_cfg: RunConfig) -> tuple[str, str]:
        query = f"{stock.name}({stock.code}) 최근 실적 뉴스 리스크"
        try:
            r = await Runner.run(
                search_agent,
                input=f"Search term: {query}\nReason: 포트폴리오 추천을 위한 최신 뉴스/리스크 요약",
                run_config=run_cfg,
            )
            return stock.code, str(r.final_output)
        except Exception:
            return stock.code, ""

    async def _generate_investor_views(
        self,
        *,
        ctx: PortfolioAgentContext,
        universe: list[Stock],
        analysis_results: list[AnalysisResult],
        run_cfg: RunConfig,
    ) -> list[InvestorView]:
        """종목별 InvestorView를 생성합니다.

        - LLM 미사용/불가 시: 중립 뷰로 폴백
        """
        # 기본: 중립 뷰
        neutral = [
            InvestorView(
                stock_indices=[i],
                expected_return=0.0,
                confidence=0.3,
                reasoning="분석 데이터가 부족하거나 LLM 비활성화로 중립적 입장을 유지합니다.",
            )
            for i in range(len(universe))
        ]

        if not self._enable_llm_views or not self._llm_available():
            return neutral

        # 종목별 분석 결과 그룹화
        grouped: dict[str, dict[str, str]] = {}
        for s in universe:
            grouped[s.code] = {
                "web_search": "",
                "financial_statement": "",
                "technical_indicator": "",
            }
        for r in analysis_results:
            if r.code not in grouped:
                continue
            grouped[r.code][r.type] = r.result or ""

        # 비용 제한: 상위 score 기준으로 일부만 LLM 호출
        # (여기서는 universe 순서를 유지하되, 호출 대상만 제한)
        max_views = max(1, int(self._view_max))
        tasks: list[asyncio.Task] = []
        for i, stock in enumerate(universe[:max_views]):
            analyses = grouped.get(stock.code, {})
            tasks.append(
                asyncio.create_task(self._view_one(i, stock, analyses, run_cfg=run_cfg))
            )

        out = list(neutral)
        for t in asyncio.as_completed(tasks):
            idx, view = await t
            if view is not None and 0 <= idx < len(out):
                out[idx] = view
        return out

    async def _view_one(
        self, idx: int, stock: Stock, analyses: dict[str, str], *, run_cfg: RunConfig
    ) -> tuple[int, InvestorView | None]:
        web = (analyses.get("web_search") or "").strip() or "N/A"
        fin = (analyses.get("financial_statement") or "").strip() or "N/A"
        tech = (analyses.get("technical_indicator") or "").strip() or "N/A"

        prompt = (
            f"종목 정보:\n- 종목명: {stock.name}\n- 종목코드: {stock.code}\n\n"
            f"분석 결과:\n"
            f"1) 웹 검색: {web}\n\n"
            f"2) 재무제표: {fin}\n\n"
            f"3) 기술적 지표: {tech}\n"
        )
        try:
            r = await Runner.run(view_agent, prompt, run_config=run_cfg)
            data = r.final_output_as(InvestorViewResponse)
            return (
                idx,
                InvestorView(
                    stock_indices=[idx],
                    expected_return=float(data.expected_return),
                    confidence=float(data.confidence),
                    reasoning=str(data.reasoning or ""),
                ),
            )
        except Exception:
            return idx, None

    async def _build_portfolio(
        self,
        *,
        universe: list[Stock],
        stock_scores: dict[str, float],
        market_data_list: list[MarketData],
        investor_views: list[InvestorView],
        risk_free_rate: float,
    ) -> PortfolioResult | None:
        builder = PortfolioBuilder()
        state = PortfolioBuilderInputState(
            portfolio_list=universe,
            market_data_list=market_data_list,
            investor_views=investor_views,
            stock_scores=stock_scores,
            risk_free_rate=float(risk_free_rate),
        )
        out = await builder(state)
        return out.get("portfolio_result")

    async def _write_report(
        self,
        *,
        ctx: PortfolioAgentContext,
        investor_type: str,
        universe: list[Stock],
        portfolio_size: int,
        stock_scores: dict[str, float],
        portfolio_result: PortfolioResult | None,
        holdings: list[dict],
        holdings_summary: dict | None,
        holdings_error: str | None,
        user_info: dict,
        run_cfg: RunConfig,
        extra_instruction: str | None = None,
    ) -> str:
        # 최종 추천 종목: weight desc 상위 N
        recommended = list((portfolio_result.weights if portfolio_result else []) or [])
        recommended = sorted(recommended, key=lambda w: w.weight, reverse=True)
        recommended = recommended[: max(0, int(portfolio_size))]

        # 최적화 결과가 없거나 비어 있으면, 점수 기반 균등배분으로 폴백
        if not recommended:
            fallback = sorted(
                universe, key=lambda s: float(stock_scores.get(s.code, 0.0)), reverse=True
            )[: max(1, int(portfolio_size))]
            if fallback:
                w = 1.0 / float(len(fallback))
                recommended = [
                    PortfolioWeight(
                        code=s.code,
                        name=s.name,
                        weight=w,
                        reasoning="최적화 결과가 없어 점수 상위 종목을 균등 비중으로 구성했습니다.",
                    )
                    for s in fallback
                ]

        # 상위 N개로 자른 후, 비중이 1이 되도록 정규화(표시용)
        sum_w = float(sum(float(w.weight) for w in recommended)) if recommended else 0.0
        if sum_w > 0 and abs(sum_w - 1.0) > 1e-6:
            recommended = [
                PortfolioWeight(
                    code=w.code,
                    name=w.name,
                    weight=float(w.weight) / sum_w,
                    reasoning=str(getattr(w, "reasoning", "") or ""),
                )
                for w in recommended
            ]

        # 업종/시장(선택): KIS inquire-price로 채움 (최종 추천 종목만)
        code_to_sector: dict[str, str] = {}
        code_to_market: dict[str, str] = {}
        try:
            code_to_sector, code_to_market = await self._enrich_sector_market(
                ctx=ctx, user_info=user_info, symbols=[w.code for w in recommended]
            )
        except Exception:
            # 업종/시장 조회 실패는 리포트 전체를 죽이지 않음
            code_to_sector, code_to_market = {}, {}

        holding_codes = {str(h.get("code", "") or "").strip() for h in holdings}

        # 결정적 리포트(폴백) 본문 데이터
        base_markdown = self._format_markdown_fallback(
            ctx=ctx,
            investor_type=investor_type,
            universe=universe,
            portfolio_size=portfolio_size,
            recommended=recommended,
            holding_codes=holding_codes,
            holdings=holdings,
            holdings_summary=holdings_summary,
            holdings_error=holdings_error,
            code_to_sector=code_to_sector,
            code_to_market=code_to_market,
        )

        if not self._enable_llm_report or not self._llm_available():
            return base_markdown

        # LLM writer: 폴백 본문을 '소스 데이터'로 주고 더 자연스러운 요약/문장화 수행
        prompt = (
            "아래는 포트폴리오 추천에 필요한 원본 데이터(초안)입니다. "
            "이 내용을 바탕으로 사용자 친화적인 마크다운 보고서를 작성하세요.\n\n"
            f"{base_markdown}\n"
        )
        if extra_instruction:
            prompt += f"\n\n추가 수정 지시:\n{extra_instruction}\n"

        try:
            r = await Runner.run(writer_agent, prompt, run_config=run_cfg)
            out = r.final_output_as(PortfolioReportData)
            md = str(out.markdown_report or "").strip()
            # 필수 섹션이 없으면 폴백
            if "## 1) 기보유 종목 현황" not in md or "## 3) 최종 추천 포트폴리오" not in md:
                return base_markdown
            return md
        except Exception:
            return base_markdown

    async def _enrich_sector_market(
        self,
        *,
        ctx: PortfolioAgentContext,
        user_info: dict,
        symbols: Iterable[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """KIS inquire-price로 업종/시장 정보를 조회합니다(최종 추천 종목만)."""
        token = str(user_info.get("kis_access_token", "") or "").strip()
        app_key = str(user_info.get("kis_app_key", "") or "").strip()
        app_secret = str(user_info.get("kis_app_secret", "") or "").strip()

        if not token or not app_key or not app_secret:
            return {}, {}

        url = f"{ctx.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": "FHKST01010100",
            "custtype": "P",
        }

        code_to_sector: dict[str, str] = {}
        code_to_market: dict[str, str] = {}

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for sym in symbols:
                sym = str(sym or "").strip()
                if not sym:
                    continue
                params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": sym}
                async with session.get(url, headers=headers, params=params) as res:
                    data = await res.json(content_type=None)
                if str(data.get("rt_cd", "")).strip() != "0":
                    continue
                output = (data.get("output") or {}) or {}
                if isinstance(output, list):
                    output = output[0] if output else {}
                if not isinstance(output, dict):
                    continue
                sector = str(output.get("bstp_kor_isnm", "") or "").strip()
                market = str(output.get("rprs_mrkt_kor_name", "") or "").strip()
                if sector:
                    code_to_sector[sym] = sector
                if market:
                    code_to_market[sym] = market
                # 과도한 burst 방지(소량 호출이므로 짧게)
                await asyncio.sleep(0.05)

        return code_to_sector, code_to_market

    @staticmethod
    def _format_markdown_fallback(
        *,
        ctx: PortfolioAgentContext,
        investor_type: str,
        universe: list[Stock],
        portfolio_size: int,
        recommended: list[Any],
        holding_codes: set[str],
        holdings: list[dict],
        holdings_summary: dict | None,
        holdings_error: str | None,
        code_to_sector: dict[str, str],
        code_to_market: dict[str, str],
    ) -> str:
        generated_at = ctx.now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: list[str] = []
        lines.append("포트폴리오 추천 보고서")
        lines.append(f"생성 시각: {generated_at}")
        lines.append(f"투자 성향(설문 기반): {investor_type}")
        lines.append(f"최종 추천 종목 수: {min(len(recommended), int(portfolio_size))}개")
        lines.append(
            f"분석 후보군: 시가총액 상위 {len([s for s in universe if s.code not in holding_codes])} + 기보유 종목 → {len(universe)}개"
        )
        lines.append("")

        # 1) holdings
        lines.append("## 1) 기보유 종목 현황")
        if holdings:
            included = [h for h in holdings if h.get("code") in {w.code for w in recommended}]
            excluded = [h for h in holdings if h.get("code") not in {w.code for w in recommended}]
            lines.append(f"- 기보유 종목: **{len(holdings)}개**")
            lines.append(
                f"- 최종 추천 포트폴리오에 포함: **{len(included)}개**, 제외: **{len(excluded)}개**"
            )
            if holdings_summary:
                dnca = holdings_summary.get("dnca_tot_amt")
                tot_evlu = holdings_summary.get("tot_evlu_amt")
                if dnca or tot_evlu:
                    lines.append(
                        f"- 예수금/총평가(요약): 예수금={dnca or 'N/A'}, 총평가={tot_evlu or 'N/A'}"
                    )
            lines.append("")
            lines.append("| 종목명 | 종목코드 | 보유수량 | 평균매입가 | 현재가 | 수익률 | 평가금액 | 평가손익 | 추천포함 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
            rec_codes = {w.code for w in recommended}
            for h in holdings:
                code = str(h.get("code", "") or "")
                name = str(h.get("name", "") or "")
                qty = h.get("quantity", 0)
                avg = h.get("avg_buy_price", 0.0)
                cur = h.get("current_price", 0.0)
                rr = h.get("return_rate", 0.0)
                ev = h.get("evaluated_amount", 0.0)
                pl = h.get("profit_loss", 0.0)
                flag = "포함" if code in rec_codes else "미포함"
                lines.append(
                    f"| {name} | {code} | {qty} | {avg:,.0f} | {cur:,.0f} | {rr*100:,.2f}% | {ev:,.0f} | {pl:,.0f} | {flag} |"
                )
            lines.append("")
        else:
            lines.append("현재 기보유 종목이 없거나, 계좌 조회 결과가 없습니다.")
            if holdings_error:
                lines.append(f"계좌 조회 실패 사유: {holdings_error}")
            if holdings_summary:
                dnca = holdings_summary.get("dnca_tot_amt")
                tot_evlu = holdings_summary.get("tot_evlu_amt")
                if dnca or tot_evlu:
                    lines.append(
                        f"(요약) 예수금={dnca or 'N/A'}, 총평가={tot_evlu or 'N/A'}"
                    )
            lines.append("")

        # 2) process summary (큰 기능만)
        lines.append("## 2) 추천 프로세스(요약)")
        lines.append("- **투자 성향 산출**: 설문(`public.survey.answer`) 기반으로 투자 성향을 결정합니다.")
        lines.append("- **(모의) 계좌/토큰 로드**: DB에서 KIS 자격증명/토큰을 조회하고 만료 시 갱신합니다.")
        lines.append("- **기보유 종목 조회(모의투자)**: KIS 잔고 조회로 기보유 종목을 확인합니다.")
        lines.append("- **후보군 구성**: 시가총액 상위 종목 + 기보유 종목으로 분석 후보군을 구성합니다.")
        lines.append("- **신호 수집**: 재무제표(DART)·기술지표(KIS)·(선택)웹검색을 수집합니다.")
        lines.append("- **투자자 뷰 생성**: 분석 결과를 종합해 종목별 예상수익/신뢰도를 생성합니다.")
        lines.append("- **최종 추천/비중 산출**: Black-Litterman으로 비중을 최적화해 최종 포트폴리오를 구성합니다.")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 3) final portfolio
        lines.append("## 3) 최종 추천 포트폴리오")
        lines.append("| 순위 | 구분 | 종목명 | 종목코드 | 업종 | 시장 | 투자비중 | 근거 |")
        lines.append("|---:|---|---|---:|---|---|---:|---|")
        for i, w in enumerate(recommended, start=1):
            kind = "기보유" if w.code in holding_codes else "신규"
            sector = code_to_sector.get(w.code, "N/A")
            market = code_to_market.get(w.code, "N/A")
            pct = float(w.weight) * 100.0
            reasoning = str(getattr(w, "reasoning", "") or "")
            lines.append(
                f"| {i} | {kind} | {w.name} | {w.code} | {sector} | {market} | {pct:.2f}% | {reasoning} |"
            )
        lines.append("")

        return "\n".join(lines)

