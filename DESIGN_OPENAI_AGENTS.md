# Stockelper Portfolio - OpenAI Agents SDK 기반 설계 문서

- **문서 목적**: 기존 `stockelper-portfolio` 서비스(추천/매수/매도 API, DB 연동, KIS/DART/웹검색)를 유지하면서, 포트폴리오 추천/매매 워크플로를 **OpenAI Agents Python SDK(`openai-agents`)** 기반으로 재구성하기 위한 설계안을 정리합니다.
- **참조 레포**: `/home/jys/workspace/Stockelper-Lab/openai-agents-python`
  - 핵심 개념: `Agent`, `Runner`, `function_tool`, `WebSearchTool`, `output_type`(Structured Outputs), `trace`, `RunConfig`, `guardrails`, `handoffs`, `Agents as tools`
  - 특히 `examples/financial_research_agent`의 “코드 기반 오케스트레이션 + 서브 에이전트/툴” 패턴을 따릅니다.
- **핵심 원칙**: **결정적 계산/외부 API 호출은 도구(코드)로 고정**하고, LLM은 **계획/요약/정합성 점검/서술(Structured Output 포함)**에만 사용합니다.

---

## 현재 시스템 요약(현행 구현 기반)

### API 레이어
- `src/routers/portfolio.py`
  - `POST /portfolio/recommendations`: 포트폴리오 추천 보고서 생성(마크다운)
  - `POST /portfolio/buy`: 매수 워크플로(LangGraph)
  - `POST /portfolio/sell`: 매도 워크플로(LangGraph)

### 추천 로직(단일 Tool 기반)
- `src/multi_agent/portfolio_analysis_agent/tools/portfolio.py`
  - DB(`stockelper_web.public.users`, `public.survey`)에서 사용자 정보/설문을 읽고,
  - KIS(모의투자) 보유종목/시세/랭킹/재무비율 등 + DART 재무제표 + (선택) 웹검색을 조합하여
  - 추천 결과를 마크다운 보고서로 출력

### 매수/매도 로직(LangGraph 기반)
- `src/portfolio_multi_agent/builder.py`
  - **매수**: LoadUserContext → Ranking → (WebSearch/FinancialStatement/TechnicalIndicator 병렬) → ViewGenerator → PortfolioBuilder(BL) → PortfolioTrader(주문)
  - **매도**: LoadUserContext → GetPortfolioHoldings → (WebSearch/FinancialStatement/TechnicalIndicator 병렬) → SellDecisionMaker → PortfolioSeller(주문)

---

## 목표/비목표

### 목표
- **Agents SDK 표준 방식으로 재구성**: `Agent/Tool/Runner/Tracing/Guardrails` 중심의 모듈형 설계
- **추천(리포트) 품질/재현성 개선**
  - 입력 구조화, 오류 분류, 재시도, 결과 정합성 감사(verify) 강화
- **운영 가시성(Tracing) 강화**
  - user_id/job_id 단위로 “어떤 도구 호출/어떤 판단”이 있었는지 추적 가능하게
- **모의투자(VTS) 기준 명확화**
  - base url/TR ID/토큰 발급 도메인/레이트리밋 정책을 명시하고, 구현이 이를 따르게

### 비목표
- LLM이 수익률/최적화/지표 계산을 직접 수행(금지)
- LLM이 임의의 트레이딩 전략/코드를 생성해 실행(운영 리스크)
- 계좌/키 매칭 문제(`INVALID_CHECK_ACNO`)를 “코드만으로 해결”(대부분 키-계좌 세트 문제로 분류하여 사용자/운영 대응 영역)

---

## 전체 오케스트레이션 설계(권장)

### 설계 선택
- **코드 기반 오케스트레이션(결정적) + LLM은 Structured Outputs** 형태로만 사용
  - 추천/매매는 안전성·재현성·비용 통제가 중요하므로, LLM은 “서술/계획/검증”으로 제한
  - `openai-agents-python/docs/ko/multi_agent.md`의 코드 기반 오케스트레이션 권장과 일치

### Mermaid: 추천(리포트) End-to-End

```mermaid
flowchart TD
  A[입력 수신\n(user_id, risk_level, portfolio_size, rank_weight)] --> B[LoadUserContext Tool\n(DB 조회 + 토큰 검증/갱신)]
  B --> C[FetchHoldings Tool\n(KIS inquire-balance, 모의)]
  C --> D[BuildUniverse Tool\n(랭킹 후보 + 보유종목 + ETF/ETN 제외)]
  D --> E[CollectSignals\n(병렬: WebSearch Agent / Financial Tool / Technical Tool)]
  E --> F[ViewGenerator Agent\n(Structured Output: expected_return/confidence/reasoning)]
  F --> G[PortfolioBuilder Tool\n(Black-Litterman 최적화)]
  G --> H[ReportWriter Agent\n(Structured Output: markdown_report + short_summary)]
  H --> I[Verifier Agent(선택)\n(정합성 점검: pass/retry/fail)]
  I -->|ok| J[최종 보고서 반환]
  I -->|retry| K[AdjustPlan Agent\n(재시도 수정안)]
  K --> D
  I -->|fail| L[실패 사유 반환/저장]
```

### Mermaid: 매수/매도(요약)

```mermaid
flowchart TD
  subgraph BUY[매수 워크플로]
    B1[LoadUserContext Tool] --> B2[Ranking Tool]
    B2 --> B3[Signals 병렬]
    B3 --> B4[ViewGenerator Agent]
    B4 --> B5[PortfolioBuilder Tool]
    B5 --> B6[OrderPlanner Tool\n(가격/현금/수량)]
    B6 --> B7[PlaceOrders Tool\n(KIS 모의 매수)]
  end
  subgraph SELL[매도 워크플로]
    S1[LoadUserContext Tool] --> S2[FetchHoldings Tool]
    S2 --> S3[Signals 병렬]
    S3 --> S4[SellDecision Agent]
    S4 --> S5[PlaceOrders Tool\n(KIS 모의 매도)]
  end
```

---

## 제안 코드 구조(Portfolio 서비스 레포 내)

`examples/financial_research_agent`(매니저 + agents/ + tools/) 패턴을 그대로 따릅니다.

- `src/portfolio_agents/`
  - `context.py`: 컨텍스트(dataclass/Pydantic)
  - `schemas.py`: LLM `output_type`(Pydantic) + tool args/result 스키마
  - `tools/`
    - `db.py`: 유저 조회/토큰 갱신/추천 결과 저장(선택)
    - `kis.py`: KIS 시세/보유/랭킹/주문(모의)
    - `dart.py`: DART 재무제표/지표(결정적)
    - `technicals.py`: 180일 OHLCV + 기술지표(결정적)
    - `optimizer.py`: Black-Litterman
  - `agents/`
    - `planner_agent.py`(선택): 요청 해석/파라미터 구조화
    - `search_agent.py`: `WebSearchTool()` 기반 최신 뉴스 요약
    - `view_agent.py`: InvestorView 생성(Structured Outputs)
    - `writer_agent.py`: 보고서 마크다운 생성(Structured Outputs)
    - `verifier_agent.py`(선택): 보고서/결과 정합성 감사
    - `sell_agent.py`(선택): 매도 판단(Structured Outputs)
  - `manager.py`: 코드 기반 오케스트레이터(추천/매수/매도)

> 기존 LangGraph 워크플로는 **기능 플래그**로 병행 운영하면서, 안정화 후 Agents 파이프라인으로 교체하는 것을 권장합니다.

---

## 컨텍스트 설계(Agents SDK `context`)

`openai-agents-python/docs/ko/context.md`를 따릅니다.

### 로컬 컨텍스트(LLM에게 직접 노출되지 않음)

```python
@dataclass
class PortfolioContext:
    user_id: int
    db_engine: AsyncEngine
    kis_base_url: str  # 기본: https://openapivts.koreainvestment.com:29443
    now_utc: datetime
    # rate limiters / caches / clients
    kis_max_rps: float
    dart_api_keys: list[str]
```

- **중요**: `context`는 도구 함수/훅/오케스트레이터에서만 사용하고, LLM이 봐야 하는 데이터는 **도구 출력/입력 메시지**로 전달합니다.

---

## 스키마 계약(Structured Outputs)

Agents SDK의 `Agent(output_type=...)`를 적극 사용합니다.

### 핵심 Pydantic 모델(예시)
- `RecommendationRequest`
  - `user_id: int`
  - `risk_level: Literal["안정형", ...]`
  - `portfolio_size: int`
  - `rank_weight: RankWeight`(11개 가중치)
- `HoldingPosition`
  - `code, name, quantity, avg_buy_price, current_price, return_rate, evaluated_amount, profit_loss`
- `UniverseItem`
  - `code, name, source: Literal["rank", "holding"]`
- `SignalBundle`
  - `web_summary: str | None`
  - `financial_metrics: dict`
  - `technical_metrics: dict`
- `InvestorView`
  - `expected_return: float` (-0.20~0.20)
  - `confidence: float` (0~1)
  - `reasoning: str` (2~3문장)
- `PortfolioWeight`
  - `code, name, weight: float, sector, market, is_holding`
- `PortfolioReportData`
  - `short_summary: str`
  - `markdown_report: str`
  - `follow_up_questions: list[str]`(선택)
- `VerificationResult`
  - `verified: bool`
  - `issues: str`
  - `retry_suggestion: str | None`

---

## Tools 설계(함수 도구: 결정적 실행)

Agents SDK의 `@function_tool`로 래핑합니다(`docs/ko/tools.md` 참조).

### 1) LoadUserContext Tool (DB + 토큰 보장)
- **입력**: `user_id`
- **처리**:
  - `stockelper_web.public.users`에서 `kis_app_key/kis_app_secret/kis_access_token/account_no` 조회
  - 저장된 토큰 유효성 검증(계좌 불필요 endpoint로 확인)
  - 만료/무효면 재발급(`/oauth2/tokenP`, VTS 도메인) 후 DB 업데이트
- **출력**: `{app_key, app_secret, access_token, account_no}`
- **오류 분류**:
  - 토큰발급 레이트리밋(EGW00133): “재시도 가능”으로 분류

### 2) FetchHoldings Tool (KIS 잔고/보유종목)
- **입력**: `{access_token, account_no}`
- **출력**: `holdings: list[HoldingPosition]` + `summary`(예수금/총평가 등)
- **오류 분류**:
  - `INVALID_CHECK_ACNO`: **계좌/키 매칭 문제**로 분류(재시도해도 대부분 불가)

### 3) BuildUniverse Tool (후보군 구성)
- **입력**: 랭킹 결과 + 보유종목
- **규칙**:
  - ETF/ETN 제외(명시적 표기 + 대표 브랜드 휴리스틱)
  - 중복 제거, 최대 후보 수 상한

### 4) Financial Tool (DART 기반 지표)
- **입력**: `symbols`
- **출력**: 종목별 재무지표 dict
- **정책**:
  - 병렬 처리하되 DART 쿼터 초과 시 키 로테이션/폴백

### 5) Technical Tool (KIS OHLCV 180일 + 지표)
- **입력**: `symbols`
- **출력**: 종목별 MACD/BB/RSI/Stoch/MA20·60·120 등
- **정책**:
  - KIS 초당 제한 준수(세마포어/균등 간격)

### 6) Optimizer Tool (Black-Litterman)
- **입력**: 랭킹 점수 + views + 공분산(가격수익률) + 제약
- **출력**: 비중/성과 지표
- **제약**:
  - 합=1
  - 개별 0~30%
  - 0.1% 미만 제외

### 7) Order Tools (모의 주문: Buy/Sell)
- **가격 조회**: inquire-price
- **주문 수량 산출**: 현금/비중/현재가 기반(결정적)
- **주문 실행**: `/trading/order-cash`, TR ID는 모의(VTTC...) 사용

---

## Agents 설계(LLM: 서술/검증 중심)

`examples/financial_research_agent`처럼 “writer/verification” 구조가 유효합니다.

### A) WebSearch Agent
- **도구**: `WebSearchTool()`
- **역할**: 종목별 최신 뉴스/리스크 요약(300단어 내)

### B) ViewGenerator Agent
- **입력**: 종목별 신호 번들(웹/재무/기술)
- **output_type**: `InvestorView`(종목별)
- **가이드라인**:
  - 데이터 부족 시 `expected_return=0.0`, `confidence=0.3`

### C) ReportWriter Agent
- **입력**: 보유종목/후보군/비중/근거
- **output_type**: `PortfolioReportData`
- **출력 섹션 권장**
  - 기보유 종목 현황(성공/실패 사유 포함)
  - 추천 프로세스(큰 기능만)
  - 최종 추천 포트폴리오 표
  - 한계/주의사항(데이터 소스, 모의투자 한계, 계좌 오류 시 영향)

### D) Verifier Agent(선택)
- **목적**: 리포트 내부 모순/미출처 주장/수치 정합성 점검
- **output_type**: `VerificationResult`

### Agents as tools vs Handoffs
- 추천 파이프라인은 **매니저(코드)**가 흐름을 고정하므로,
  - 서브 에이전트는 `agent.as_tool(...)`로 **도구 형태**로 붙이는 방식을 권장(`docs/ko/tools.md#도구로서의-에이전트`)
  - “대화형 분기”가 필요할 때만 `handoffs` 사용(`docs/ko/handoffs.md`)

---

## Guardrails(입력/출력/도구)

### 입력 가드레일(Input)
- `portfolio_size` 범위(1~20)
- `rank_weight` 범위/정규화 정책(0~1, 합이 0이면 디폴트)
- 모의투자 고정(base_url/TR ID) 강제

### 출력 가드레일(Output)
- 비중 합 검증(1±ε)
- 개별 비중 상한(≤30%)
- ETF/ETN 포함 여부 재검증(2중 방어)

### 도구 가드레일(Tool)
- 민감정보 마스킹(app_secret, access_token)
- 레이트리밋 보호(세마포어/재시도/백오프)

---

## Tracing/Observability(운영)

Agents SDK 트레이싱을 기본 사용(`docs/ko/tracing.md`).

- 권장 `workflow_name`: `"Stockelper Portfolio Agent"`
- 권장 `group_id`: `user_id` 또는 “요청 단위 request_id”
- 메타데이터:
  - `portfolio_size`, `risk_level`, `universe_size`, `holdings_count`
  - `holdings_error_code`(예: INVALID_CHECK_ACNO)
  - `retry_count`
- 민감정보:
  - `RunConfig(trace_include_sensitive_data=False)` 권장

---

## 오류/재시도 정책(권장)

- **토큰 만료/무효**: 1회 재발급 후 재시도(성공 시 DB 업데이트)
- **KIS 초당 제한**: 지수 백오프 + 균등 간격 제한
- **`INVALID_CHECK_ACNO`**:
  - “코드 재시도”로 해결되지 않는 경우가 대부분 → **계좌/키 매칭 문제**로 분류
  - 리포트에는 “보유종목 조회 실패 사유”를 명확히 표기하고, 추천 자체는 “보유종목 미반영 모드”로 계속 진행(옵션)

---

## 마이그레이션(권장)

### 단계적 도입
- M1. `src/portfolio_agents/`에 최소 기능(LoadUserContext Tool + ReportWriter Agent)부터 구축
- M2. 추천 API에 기능 플래그로 Agents 파이프라인 추가(기존 로직과 병행)
- M3. 신호 수집(재무/기술/웹) 및 BL 최적화까지 확장
- M4. 매수/매도 워크플로 확장(주문 도구 + sell agent)

---

## 테스트 전략(권장)

- **유닛 테스트**
  - DB 조회/토큰 갱신(모킹)
  - ETF/ETN 필터 규칙
  - 비중 제약/합 검증(optimizer)
  - 리포트 섹션(기보유 종목 표기, 오류 표기)
- **통합 테스트**
  - 스테이징 DB + KIS VTS 환경에서 user_id별 케이스
  - “보유종목 조회 성공(user_id=4)” vs “실패(user_id=2)” 비교 리포트 검증

---

## 부록: 모델/구성 권장

- 모델 혼합(비용/지연 최적화)
  - View/Writer: 품질 우선 모델
  - Verifier: 빠른 모델
- run 단위 구성
  - `RunConfig(model=..., workflow_name=..., trace_include_sensitive_data=False)`

