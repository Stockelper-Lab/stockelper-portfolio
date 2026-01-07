# Stockelper Portfolio Service

LangGraph 기반 포트폴리오 추천 및 자동 매매 서비스입니다.

## 🚀 주요 기능

- 투자 성향 기반 포트폴리오 추천
- LangGraph 다중 에이전트 매수/매도 워크플로우
- Black-Litterman 모델 기반 포트폴리오 최적화
- 한국투자증권 (KIS) API 연동 실제 거래
- 다중 지표 종목 랭킹 시스템

## 📋 기술 스택

- Python 3.12+
- FastAPI 0.111
- LangGraph (상태 그래프 기반 워크플로우)
- LangChain 1.0+
- PostgreSQL (asyncpg, psycopg)
- OpenRouter API (Perplexity, GPT-4.5.1)
- Korea Investment & Securities (KIS) API
- OpenDartReader (한국 금융감독원 DART)

## 🔌 API 엔드포인트

### 기본
- `GET /` - 루트 엔드포인트
- `GET /health` - 헬스 체크

### 포트폴리오
- `POST /portfolio/recommendations` - 투자 성향 기반 추천
- `POST /portfolio/buy` - 매수 워크플로우 (LangGraph)
- `POST /portfolio/sell` - 매도 워크플로우 (LangGraph)

## 🤖 LangGraph 워크플로우

### 매수 워크플로우

```
Ranking (11개 지표 기반)
  ↓
Analysis (병렬 3개)
  ├─ WebSearch (Perplexity)
  ├─ FinancialStatement (재무제표)
  └─ TechnicalIndicator (기술적 지표)
  ↓
ViewGenerator (Black-Litterman 뷰 생성)
  ↓
PortfolioBuilder (포트폴리오 최적화)
  ↓
PortfolioTrader (매수 주문 실행)
```

### 매도 워크플로우

```
GetPortfolioHoldings (보유 종목 조회)
  ↓
Analysis (병렬 3개)
  ├─ WebSearch
  ├─ FinancialStatement
  └─ TechnicalIndicator
  ↓
SellDecisionMaker (매도 결정)
  ↓
PortfolioSeller (매도 주문 실행)
```

## 📊 종목 랭킹 시스템

11개 랭킹 함수:
- 거래 활동성
- 영업 이익률
- 성장률
- 부채 수준
- 상승률
- 안정성
- 순이익
- 하락률
- 시가총액

## ⚙️ 환경 변수

```bash
# 서버 설정
HOST=0.0.0.0
PORT=21008
DEBUG=false

# 데이터베이스 (필수)
# - stockelper_web DB를 가리켜야 합니다.
# - /portfolio/* 에서 user_id 기반으로 public.users / public.survey 를 조회합니다.
DATABASE_URL=postgresql://user:pass@host:5432/stockelper_web
ASYNC_DATABASE_URL=

# (선택) 기본 schema는 public 입니다. 다르면 지정
STOCKELPER_WEB_SCHEMA=public

ASYNC_DATABASE_URL_KSIC=postgresql+asyncpg://user:pass@host:5432/ksic  # 선택

# 외부 API
# - DART 키는 1개 또는 여러 개를 설정할 수 있습니다.
# - 여러 개를 쓸 경우, `status=020(사용한도 초과)`가 나면 다음 키로 자동 전환합니다.
#   예) OPEN_DART_API_KEYS=key1,key2,key3
OPEN_DART_API_KEY=
OPEN_DART_API_KEYS=
OPENROUTER_API_KEY=

# (옵션) Langfuse 트레이싱
# - 설정 시 /portfolio/* 요청의 LangChain/LangGraph 실행이 Langfuse로 트레이싱됩니다.
# - 가이드: [Langfuse Get Started](https://langfuse.com/docs/observability/get-started)
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# (선택) KIS 호출 Rate Limit (초당 최대 요청 수)
KIS_MAX_REQUESTS_PER_SECOND=20

# (추천/분석 전용) KIS 호출 RPS 제한 (모의투자는 특히 낮게 권장: 1~2)
KIS_ANALYSIS_MAX_REQUESTS_PER_SECOND=1
```

## 🚀 빠른 시작

### 로컬 실행

```bash
# 의존성 설치
uv sync --dev

# 서버 실행
PORT=21008 uv run python src/main.py
```

### Docker 실행

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f stockelper-portfolio-server
```

## 📝 API 사용 예시

### 포트폴리오 추천

```bash
curl -X POST http://localhost:21008/portfolio/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "portfolio_size": 10}'
```

- 응답에는 `id`(PK)와 `job_id`(UUID)가 포함됩니다.
- 서버는 요청을 받는 즉시 `public.portfolio_recommendations`에 **빈 레코드(placeholder)** 를 먼저 저장한 뒤,
  추천 생성이 완료되면 해당 레코드를 업데이트합니다.
- `result`는 **보고서 형태(Markdown)** 로 저장되며, 기보유 종목이 있으면 이를 포함하고 추천 프로세스 요약과 최종 포트폴리오를 구조화해 제공합니다.

### 매수 워크플로우

```bash
curl -X POST http://localhost:21008/portfolio/buy \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "max_portfolio_size": 10,
    "rank_weight": {"market_cap": 1.0},
    "portfolio_list": [],
    "risk_free_rate": 0.03
  }'
```

### 매도 워크플로우

```bash
curl -X POST http://localhost:21008/portfolio/sell \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "loss_threshold": -0.05,
    "profit_threshold": 0.15
  }'
```

## 🗄️ 데이터베이스

### public.users 테이블
- id
- kis_app_key, kis_app_secret
- kis_access_token (요청 시 발급 후 저장)
- account_no

### public.survey 테이블
- user_id
- answer (JSON) 예: {"q1": 3, "q2": 5, ...}

### public.portfolio_recommendations 테이블
- id (PK, text)
- job_id (UUID string)
- user_id (FK → public.users.id)
- investor_type (text)
- result (text)
- created_at, updated_at

### industy 테이블 (KSIC DB)
- industy_code (5자리 코드)
- industy_name (산업 분류명)

## 🔒 보안

- 모든 API 키 환경 변수 관리
- KIS 토큰 자동 갱신 (DB 저장)
- Rate limiting (초당 20 요청)
- `.env` 파일 커밋 금지

## 🐳 Docker 구성

### 서비스
- **stockelper-portfolio-server** (포트: 21008)
  - FastAPI 애플리케이션
  - 헬스체크: `/health`

### 네트워크
- `stockelper` 브리지 네트워크

## 📄 라이선스

MIT License
