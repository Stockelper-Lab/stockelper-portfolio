## Stockelper Portfolio Service

OpenAI **Agents SDK(`openai-agents`)** 기반 포트폴리오 추천/매수/매도 FastAPI 서비스입니다.

- **user_id 기반**으로 DB(`stockelper_web`)에서 사용자 KIS 자격증명을 조회하고,
- 결정적 단계(외부 API/계산)는 **코드로 고정**하며,
- LLM은 (선택적으로) **WebSearch / Investor View / Report / Verify**에만 사용합니다.

---

## 주요 기능

- **포트폴리오 추천**: 기보유 종목 + 시가총액 상위 후보군 → 신호 수집 → Black-Litterman 최적화 → 마크다운 리포트
- **매수/매도 실행**: KIS API 주문 실행(기본 VTS/모의투자 도메인 사용)
- **레이트리밋 방어**: KIS 시총랭킹 호출에 캐시/직렬화/백오프 적용(동시 요청에서 안정성 개선)

---

## 기술 스택

- Python 3.12+
- FastAPI / Uvicorn
- OpenAI Agents SDK (`openai-agents`)
- PostgreSQL (`asyncpg`, `psycopg`)
- KIS API, DART(OpenDartReader)

---

## 디렉토리 구조(핵심)

- `src/main.py`: FastAPI 앱 엔트리포인트
- `src/routers/portfolio.py`: `/portfolio/*` 엔드포인트
- `src/portfolio_agents/`: Agents SDK 기반 오케스트레이션(추천/매수/매도)
- `src/portfolio_multi_agent/`: 결정적 계산/외부 API 호출 모듈(랭킹/재무/기술/최적화/주문)
- `src/multi_agent/utils.py`: DB 접근(사용자/설문/추천결과 적재) 및 KIS 토큰 보장 유틸
- `ref-open-trading-api/`: KIS OpenAPI 참고 자료(서비스 런타임과 무관)

---

## 실행 전 필수 조건

### 1) 데이터베이스

- **DB 이름**: `stockelper_web`
- **스키마**: 기본 `public` (다르면 `STOCKELPER_WEB_SCHEMA`로 지정)
- **사용 테이블**
  - `public.users`: `id`, `kis_app_key`, `kis_app_secret`, `kis_access_token`, `account_no`
  - `public.survey`: `user_id`, `answer(JSON)`
  - `public.portfolio_recommendations`: 결과 저장(요청 즉시 placeholder 생성 후 업데이트)

### 2) 외부 API 키

- **KIS 자격증명**: API 요청 파라미터로 받지 않고 **DB(users)** 에 저장된 값을 사용
- **DART 키**: `OPEN_DART_API_KEY` 또는 `OPEN_DART_API_KEYS`
- **OpenAI 키**: `OPENAI_API_KEY`
  - 미설정 시 LLM 단계는 자동으로 비활성화/폴백됩니다(결정적 단계는 수행).

---

## 환경 변수(.env)

`src/main.py`에서 `python-dotenv`로 `.env`를 자동 로딩합니다.

### 서버

```bash
HOST=0.0.0.0
# 기본값(코드): 21010, 운영/로컬에서는 21008 사용을 권장합니다.
PORT=21008
DEBUG=false
```

### DB

```bash
DATABASE_URL=postgresql://user:pass@host:5432/stockelper_web
# 선택(없으면 DATABASE_URL 기반으로 내부 변환)
# ASYNC_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/stockelper_web
STOCKELPER_WEB_SCHEMA=public
```

### OpenAI Agents SDK

```bash
OPENAI_API_KEY=
# OPENAI_AGENTS_DISABLE_TRACING=1
```

### DART

```bash
OPEN_DART_API_KEY=
# 여러 개 설정 시 자동 로테이션(쿼터 초과 시 다음 키 사용)
# OPEN_DART_API_KEYS=key1,key2,key3
```

### KIS 호출 옵션(권장)

```bash
# 모의투자 기본값: openapivts
# KIS_API_BASE_URL=https://openapivts.koreainvestment.com:29443

# 초당 최대 요청 수(전체)
KIS_MAX_REQUESTS_PER_SECOND=20
# 분석/랭킹처럼 burst가 생기기 쉬운 경로는 더 낮게 권장(모의투자: 1~2)
KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND=1

# (선택) 시총랭킹 호출 안정화 옵션
KIS_RANK_CACHE_TTL_SECONDS=30
KIS_RANK_CACHE_MAX_AGE_SECONDS=300
KIS_RANK_RETRY_MAX=3
```

### Agents 오케스트레이션 옵션(선택)

```bash
PORTFOLIO_ENABLE_WEBSEARCH=true
PORTFOLIO_ENABLE_LLM_VIEWS=true
PORTFOLIO_ENABLE_LLM_REPORT=true
PORTFOLIO_ENABLE_LLM_VERIFY=true

PORTFOLIO_ANALYSIS_TOP_N=20
PORTFOLIO_DEFAULT_SIZE=10
PORTFOLIO_WEBSEARCH_MAX_STOCKS=6
PORTFOLIO_VIEW_MAX_STOCKS=10
PORTFOLIO_UNIVERSE_MAX=30
PORTFOLIO_SELL_MAX_STOCKS=30

# 모델 선택(Agents SDK)
# OPENAI_AGENTS_SEARCH_MODEL=gpt-4.1-mini
# OPENAI_AGENTS_VIEW_MODEL=gpt-4.1-mini
# OPENAI_AGENTS_REPORT_MODEL=gpt-4.1-mini
# OPENAI_AGENTS_VERIFY_MODEL=gpt-4.1-mini
```

---

## 빠른 시작(uv)

```bash
uv sync --dev
PORT=21008 uv run python src/main.py
```

---

## Docker 실행

### 1) 네트워크 생성(최초 1회)

`docker-compose.yml`은 외부 네트워크 `stockelper`를 사용합니다.

```bash
docker network create stockelper
```

### 2) 실행

```bash
docker compose up -d --build
docker compose logs -f portfolio-server
```

---

## API

### 공통

- `GET /` / `GET /health`

### 1) 포트폴리오 추천

```bash
curl -X POST http://localhost:21008/portfolio/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "portfolio_size": 10, "include_web_search": false, "risk_free_rate": 0.03}'
```

응답(`result`)은 **마크다운 보고서**이며, DB(`public.portfolio_recommendations`)에 저장됩니다.

### 2) 매수(주의: 주문 실행)

`/portfolio/buy`는 KIS 주문을 실행합니다(기본 VTS 도메인).

```bash
curl -X POST http://localhost:21008/portfolio/buy \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
    "max_portfolio_size": 10,
    "rank_weight": {"market_cap": 1.0},
    "risk_free_rate": 0.03
  }'
```

### 3) 매도(주의: 주문 실행)

```bash
curl -X POST http://localhost:21008/portfolio/sell \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 2,
    "loss_threshold": -0.10,
    "profit_threshold": 0.20
  }'
```

---

## 트러블슈팅

### 1) `400 Bad Request` + “KIS 자격증명이 유효하지 않습니다”

- 의미: 해당 `user_id`의 `public.users.kis_app_key/kis_app_secret`이 비어있거나 placeholder 입니다.
- 조치: DB에서 값을 확인/수정하세요.

### 2) `429 Too Many Requests` 또는 KIS “초당 거래건수 초과”

- 의미: KIS 호출이 burst로 몰렸습니다(동시 요청/여러 인스턴스/다른 서비스가 같은 AppKey 공유 등).
- 조치:
  - `KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND`를 낮추고,
  - `KIS_RANK_CACHE_TTL_SECONDS`를 늘리거나,
  - 다중 레플리카라면 Redis 기반 공유 캐시/락 도입을 권장합니다.

### 3) DART “사용한도 초과(status=020)”

- `OPEN_DART_API_KEYS`를 여러 개 설정하면 자동으로 다음 키로 전환합니다.

---

## 개발

```bash
uv sync --dev
uv run pytest -q
```

---

## 라이선스

MIT
